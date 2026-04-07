"""
Test which of the 3 grid configs is best, then micro-optimize.
"""
import csv
import math

TRAIN_FILE = "train_100k_withratings.csv"

# The 3 configs from assignment1.py
CONFIGS = [
    {
        "TOP_K_ITEM": 80, "TOP_K_USER": 60, "BIAS_REG": 5.0,
        "SHRINKAGE_ITEM": 90.0, "SHRINKAGE_USER": 25.0,
        "MIN_COMMON_USERS": 5, "MIN_COMMON_ITEMS": 4,
        "SIM_POWER_ITEM": 2.0, "SIM_POWER_USER": 1.5, "ITEM_WEIGHT": 0.72
    },
    {
        "TOP_K_ITEM": 100, "TOP_K_USER": 60, "BIAS_REG": 5.0,
        "SHRINKAGE_ITEM": 100.0, "SHRINKAGE_USER": 30.0,
        "MIN_COMMON_USERS": 5, "MIN_COMMON_ITEMS": 4,
        "SIM_POWER_ITEM": 1.8, "SIM_POWER_USER": 1.5, "ITEM_WEIGHT": 0.70
    },
    {
        "TOP_K_ITEM": 80, "TOP_K_USER": 80, "BIAS_REG": 8.0,
        "SHRINKAGE_ITEM": 90.0, "SHRINKAGE_USER": 25.0,
        "MIN_COMMON_USERS": 4, "MIN_COMMON_ITEMS": 4,
        "SIM_POWER_ITEM": 2.0, "SIM_POWER_USER": 1.3, "ITEM_WEIGHT": 0.68
    }
]


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


def build_model(cfg):
    u_r, i_r = {}, {}
    u_a, i_a, u_b, i_b = {}, {}, {}, {}
    ic, uc = {}, {}
    ga = 3.5

    def clamp(x):
        return max(0.5, min(5.0, x)) if x == x else ga

    def base(u, i):
        r = ga
        if u in u_b:
            r += u_b[u]
        if i in i_b:
            r += i_b[i]
        return r

    def isim(i1, i2):
        k = tuple(sorted((i1, i2)))
        if k in ic:
            return ic[k]
        r1, r2 = i_r.get(i1, {}), i_r.get(i2, {})
        co = set(r1.keys()) & set(r2.keys())
        if len(co) < cfg["MIN_COMMON_USERS"]:
            ic[k] = 0.0
            return 0.0
        n, d1, d2 = 0.0, 0.0, 0.0
        for u in co:
            mu = u_a[u]
            v1, v2 = r1[u] - mu, r2[u] - mu
            n += v1 * v2
            d1 += v1**2
            d2 += v2**2
        if d1 == 0 or d2 == 0:
            ic[k] = 0.0
            return 0.0
        s = n / (math.sqrt(d1) * math.sqrt(d2))
        s *= (len(co) / (len(co) + cfg["SHRINKAGE_ITEM"]))
        s = math.pow(s, cfg["SIM_POWER_ITEM"]) if s > 0 else 0
        ic[k] = s
        return s

    def usim(u1, u2):
        if u1 == u2:
            return 1.0
        k = tuple(sorted((u1, u2)))
        if k in uc:
            return uc[k]
        r1, r2 = u_r.get(u1, {}), u_r.get(u2, {})
        co = set(r1.keys()) & set(r2.keys())
        if len(co) < cfg["MIN_COMMON_ITEMS"]:
            uc[k] = 0.0
            return 0.0
        mu1, mu2 = u_a[u1], u_a[u2]
        n, d1, d2 = 0.0, 0.0, 0.0
        for i in co:
            v1, v2 = r1[i] - mu1, r2[i] - mu2
            n += v1 * v2
            d1 += v1**2
            d2 += v2**2
        if d1 == 0 or d2 == 0:
            uc[k] = 0.0
            return 0.0
        s = n / (math.sqrt(d1) * math.sqrt(d2))
        s *= (len(co) / (len(co) + cfg["SHRINKAGE_USER"]))
        s = math.pow(s, cfg["SIM_POWER_USER"]) if s > 0 else 0
        uc[k] = s
        return s

    def pi(u, i):
        ba = base(u, i)
        if u not in u_r or i not in i_r:
            return ba, 0.0
        ns = []
        for oi, r in u_r[u].items():
            if oi == i:
                continue
            s = isim(i, oi)
            if s > 0:
                d = r - base(u, oi)
                ns.append((s, d))
        if not ns:
            return ba, 0.0
        ns.sort(reverse=True)
        ns = ns[:cfg["TOP_K_ITEM"]]
        ws = sum(s*d for s, d in ns)
        ss = sum(s for s, d in ns)
        return ba + (ws/ss) if ss > 0 else ba, ss

    def pu(u, i):
        ba = base(u, i)
        if u not in u_r or i not in i_r:
            return ba, 0.0
        ns = []
        for ou, r in i_r[i].items():
            if ou == u:
                continue
            s = usim(u, ou)
            if s > 0:
                d = r - base(ou, i)
                ns.append((s, d))
        if not ns:
            return ba, 0.0
        ns.sort(reverse=True)
        ns = ns[:cfg["TOP_K_USER"]]
        ws = sum(s*d for s, d in ns)
        ss = sum(s for s, d in ns)
        return ba + (ws/ss) if ss > 0 else ba, ss

    def pred(u, i):
        if u not in u_r and i not in i_r:
            return clamp(ga)
        if u not in u_r:
            return clamp(i_a.get(i, base(u, i)))
        if i not in i_r:
            return clamp(u_a.get(u, base(u, i)))
        ip, iss = pi(u, i)
        up, uss = pu(u, i)
        ba = base(u, i)
        if iss == 0 and uss == 0:
            return clamp(ba)
        if uss == 0:
            return clamp(ip)
        if iss == 0:
            return clamp(up)
        wi = cfg["ITEM_WEIGHT"] * iss
        wu = (1 - cfg["ITEM_WEIGHT"]) * uss
        return clamp((wi * ip + wu * up) / (wi + wu))

    def train(ratings):
        nonlocal u_r, i_r, u_a, i_a, u_b, i_b, ic, uc, ga
        u_r, i_r, ic, uc = {}, {}, {}, {}
        ts = 0
        for u, i, r in ratings:
            ts += r
            u_r.setdefault(u, {})[i] = r
            i_r.setdefault(i, {})[u] = r
        ga = ts / len(ratings) if ratings else 3.5
        for u in u_r:
            u_a[u] = sum(u_r[u].values()) / len(u_r[u])
        for i in i_r:
            i_a[i] = sum(i_r[i].values()) / len(i_r[i])
        for u in u_r:
            u_b[u] = sum(r - ga for r in u_r[u].values()) / \
                (len(u_r[u]) + cfg["BIAS_REG"])
        for i in i_r:
            d = sum(r - ga - u_b.get(u, 0) for u, r in i_r[i].items())
            i_b[i] = d / (len(i_r[i]) + cfg["BIAS_REG"])

    return train, pred


def eval(tf, pf, tr, vr):
    tf(tr)
    e = sum(abs(pf(u, i) - r) for u, i, r in vr)
    return e / len(vr)


all_r = read_training_data(TRAIN_FILE)
tr, vr = create_validation_split(all_r)

print("Testing current 3 configs:")
best_mae = 999
best_cfg_idx = 0
for idx, cfg in enumerate(CONFIGS, 1):
    tf, pf = build_model(cfg)
    mae = eval(tf, pf, tr, vr)
    marker = " <-- BEST" if mae < best_mae else ""
    print(f"Config {idx}: MAE={mae:.5f}{marker}")
    if mae < best_mae:
        best_mae = mae
        best_cfg_idx = idx - 1

print(f"\nBest config: #{best_cfg_idx + 1} with MAE={best_mae:.5f}")
print(f"Current baseline: 0.71883")
print(f"Gain: {0.71883 - best_mae:+.5f}")
