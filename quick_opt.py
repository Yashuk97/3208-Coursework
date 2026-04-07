"""Quick targeted optimization - focus on ITEM_WEIGHT only (most impactful)."""
import csv
import math

TRAIN_FILE = "train_100k_withratings.csv"


def read_training_data(path):
    ratings = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            ratings.append((int(row[0]), int(row[1]), float(row[2])))
    return ratings


def create_validation_split(ratings):
    train, valid = [], []
    for idx, r in enumerate(ratings):
        if idx % 10 == 0:
            valid.append(r)
        else:
            train.append(r)
    return train, valid


def build_model(iw):
    """Build model with specific ITEM_WEIGHT."""
    u_ratings, i_ratings = {}, {}
    u_avg, i_avg, u_bias, i_bias = {}, {}, {}, {}
    i_sim_cache, u_sim_cache = {}, {}
    global_avg = 3.5

    def clamp(x):
        return max(0.5, min(5.0, x)) if x == x else global_avg

    def get_base(u, i):
        res = global_avg
        if u in u_bias:
            res += u_bias[u]
        if i in i_bias:
            res += i_bias[i]
        return res

    def get_isim(i1, i2):
        key = tuple(sorted((i1, i2)))
        if key in i_sim_cache:
            return i_sim_cache[key]
        r1, r2 = i_ratings.get(i1, {}), i_ratings.get(i2, {})
        com = set(r1.keys()) & set(r2.keys())
        if len(com) < 5:
            i_sim_cache[key] = 0.0
            return 0.0
        num, d1, d2 = 0.0, 0.0, 0.0
        for u in com:
            mu = u_avg[u]
            v1, v2 = r1[u] - mu, r2[u] - mu
            num += v1 * v2
            d1 += v1**2
            d2 += v2**2
        if d1 == 0 or d2 == 0:
            i_sim_cache[key] = 0.0
            return 0.0
        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (len(com) / (len(com) + 90.0))
        sim = math.pow(sim, 2.0) if sim > 0 else 0
        i_sim_cache[key] = sim
        return sim

    def get_usim(u1, u2):
        if u1 == u2:
            return 1.0
        key = tuple(sorted((u1, u2)))
        if key in u_sim_cache:
            return u_sim_cache[key]
        r1, r2 = u_ratings.get(u1, {}), u_ratings.get(u2, {})
        com = set(r1.keys()) & set(r2.keys())
        if len(com) < 4:
            u_sim_cache[key] = 0.0
            return 0.0
        mu1, mu2 = u_avg[u1], u_avg[u2]
        num, d1, d2 = 0.0, 0.0, 0.0
        for i in com:
            v1, v2 = r1[i] - mu1, r2[i] - mu2
            num += v1 * v2
            d1 += v1**2
            d2 += v2**2
        if d1 == 0 or d2 == 0:
            u_sim_cache[key] = 0.0
            return 0.0
        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (len(com) / (len(com) + 25.0))
        sim = math.pow(sim, 1.5) if sim > 0 else 0
        u_sim_cache[key] = sim
        return sim

    def pred_item(u, i):
        base = get_base(u, i)
        if u not in u_ratings or i not in i_ratings:
            return base, 0.0
        neighs = []
        for oi, r in u_ratings[u].items():
            if oi == i:
                continue
            sim = get_isim(i, oi)
            if sim > 0:
                dev = r - get_base(u, oi)
                neighs.append((sim, dev))
        if not neighs:
            return base, 0.0
        neighs.sort(key=lambda x: x[0], reverse=True)
        neighs = neighs[:80]
        ws = sum(s*d for s, d in neighs)
        ss = sum(s for s, d in neighs)
        return base + (ws/ss) if ss > 0 else base, ss

    def pred_user(u, i):
        base = get_base(u, i)
        if u not in u_ratings or i not in i_ratings:
            return base, 0.0
        neighs = []
        for ou, r in i_ratings[i].items():
            if ou == u:
                continue
            sim = get_usim(u, ou)
            if sim > 0:
                dev = r - get_base(ou, i)
                neighs.append((sim, dev))
        if not neighs:
            return base, 0.0
        neighs.sort(key=lambda x: x[0], reverse=True)
        neighs = neighs[:60]
        ws = sum(s*d for s, d in neighs)
        ss = sum(s for s, d in neighs)
        return base + (ws/ss) if ss > 0 else base, ss

    def predict(u, i):
        if u not in u_ratings and i not in i_ratings:
            return clamp(global_avg)
        if u not in u_ratings:
            return clamp(i_avg.get(i, get_base(u, i)))
        if i not in i_ratings:
            return clamp(u_avg.get(u, get_base(u, i)))
        ip, iss = pred_item(u, i)
        up, uss = pred_user(u, i)
        base = get_base(u, i)
        if iss == 0 and uss == 0:
            return clamp(base)
        if uss == 0:
            return clamp(ip)
        if iss == 0:
            return clamp(up)
        wi = iw * iss
        wu = (1 - iw) * uss
        return clamp((wi * ip + wu * up) / (wi + wu))

    def train(ratings):
        nonlocal u_ratings, i_ratings, u_avg, i_avg, u_bias, i_bias, i_sim_cache, u_sim_cache, global_avg
        u_ratings, i_ratings, i_sim_cache, u_sim_cache = {}, {}, {}, {}
        ts = 0
        for u, i, r in ratings:
            ts += r
            u_ratings.setdefault(u, {})[i] = r
            i_ratings.setdefault(i, {})[u] = r
        global_avg = ts / len(ratings) if ratings else 3.5
        for u in u_ratings:
            u_avg[u] = sum(u_ratings[u].values()) / len(u_ratings[u])
        for i in i_ratings:
            i_avg[i] = sum(i_ratings[i].values()) / len(i_ratings[i])
        for u in u_ratings:
            u_bias[u] = sum(
                r - global_avg for r in u_ratings[u].values()) / (len(u_ratings[u]) + 5.0)
        for i in i_ratings:
            diff = sum(r - global_avg - u_bias.get(u, 0)
                       for u, r in i_ratings[i].items())
            i_bias[i] = diff / (len(i_ratings[i]) + 5.0)

    return train, predict


def eval(train_fn, pred_fn, train_r, valid_r):
    train_fn(train_r)
    err = sum(abs(pred_fn(u, i) - r) for u, i, r in valid_r)
    return err / len(valid_r)


all_r = read_training_data(TRAIN_FILE)
tr, vr = create_validation_split(all_r)

best_mae = 0.71883
best_iw = 0.72

print("Testing ITEM_WEIGHT values:")
for iw_val in [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.74, 0.76, 0.78, 0.80, 0.85]:
    train_fn, pred_fn = build_model(iw_val)
    mae = eval(train_fn, pred_fn, tr, vr)
    marker = " <-- BETTER!" if mae < best_mae else ""
    print(f"ITEM_WEIGHT={iw_val:.2f}: {mae:.5f}{marker}")
    if mae < best_mae:
        best_mae = mae
        best_iw = iw_val

print(f"\nBest: ITEM_WEIGHT={best_iw:.2f} → MAE={best_mae:.5f}")
print(f"Improvement: {0.71883 - best_mae:.5f}")
