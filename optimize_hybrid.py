"""
Fine-grained optimization for hybrid model.
Focus: ITEM_WEIGHT, BIAS_REG, and per-method TOP_K.
"""

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


def build_hybrid_model(
    top_k_item=80, top_k_user=60, bias_reg=5.0,
    shrinkage_item=90.0, shrinkage_user=25.0,
    min_common_users=5, min_common_items=4,
    sim_power_item=2.0, sim_power_user=1.5,
    item_weight=0.72
):
    """Factory to build a hybrid CF model variant."""

    user_ratings = {}
    item_ratings = {}
    user_avg = {}
    item_avg = {}
    user_bias = {}
    item_bias = {}
    item_sim_cache = {}
    user_sim_cache = {}
    global_avg = 3.5

    def clamp(x):
        if x != x:
            return global_avg
        return max(0.5, min(5.0, x))

    def get_baseline(u, i):
        res = global_avg
        if u in user_bias:
            res += user_bias[u]
        if i in item_bias:
            res += item_bias[i]
        return res

    def get_item_sim(item1, item2):
        key = tuple(sorted((item1, item2)))
        if key in item_sim_cache:
            return item_sim_cache[key]

        r1 = item_ratings.get(item1, {})
        r2 = item_ratings.get(item2, {})
        common = set(r1.keys()) & set(r2.keys())
        n_common = len(common)

        if n_common < min_common_users:
            item_sim_cache[key] = 0.0
            return 0.0

        num, d1, d2 = 0.0, 0.0, 0.0
        for u in common:
            mu = user_avg[u]
            v1, v2 = r1[u] - mu, r2[u] - mu
            num += v1 * v2
            d1 += v1 * v1
            d2 += v2 * v2

        if d1 == 0.0 or d2 == 0.0:
            item_sim_cache[key] = 0.0
            return 0.0

        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (n_common / (n_common + shrinkage_item))
        if sim > 0.0:
            sim = math.pow(sim, sim_power_item)
        else:
            sim = 0.0
        item_sim_cache[key] = sim
        return sim

    def get_user_sim(u1, u2):
        if u1 == u2:
            return 1.0
        key = tuple(sorted((u1, u2)))
        if key in user_sim_cache:
            return user_sim_cache[key]

        r1 = user_ratings.get(u1, {})
        r2 = user_ratings.get(u2, {})
        common = set(r1.keys()) & set(r2.keys())
        n_common = len(common)

        if n_common < min_common_items:
            user_sim_cache[key] = 0.0
            return 0.0

        mu1, mu2 = user_avg[u1], user_avg[u2]
        num, d1, d2 = 0.0, 0.0, 0.0
        for i in common:
            v1, v2 = r1[i] - mu1, r2[i] - mu2
            num += v1 * v2
            d1 += v1 * v1
            d2 += v2 * v2

        if d1 == 0.0 or d2 == 0.0:
            user_sim_cache[key] = 0.0
            return 0.0

        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (n_common / (n_common + shrinkage_user))
        if sim > 0.0:
            sim = math.pow(sim, sim_power_user)
        else:
            sim = 0.0
        user_sim_cache[key] = sim
        return sim

    def predict_item(user, item):
        base = get_baseline(user, item)
        if user not in user_ratings or item not in item_ratings:
            return base, 0.0

        neighbours = []
        for other_item, rating in user_ratings[user].items():
            if other_item == item:
                continue
            sim = get_item_sim(item, other_item)
            if sim > 0.0:
                dev = rating - get_baseline(user, other_item)
                neighbours.append((sim, dev))

        if not neighbours:
            return base, 0.0
        neighbours.sort(key=lambda x: x[0], reverse=True)
        neighbours = neighbours[:top_k_item]

        w_sum = sum(sim * dev for sim, dev in neighbours)
        s_sum = sum(sim for sim, dev in neighbours)
        if s_sum == 0.0:
            return base, 0.0
        return base + (w_sum / s_sum), s_sum

    def predict_user(user, item):
        base = get_baseline(user, item)
        if user not in user_ratings or item not in item_ratings:
            return base, 0.0

        neighbours = []
        for other_user, rating in item_ratings[item].items():
            if other_user == user:
                continue
            sim = get_user_sim(user, other_user)
            if sim > 0.0:
                dev = rating - get_baseline(other_user, item)
                neighbours.append((sim, dev))

        if not neighbours:
            return base, 0.0
        neighbours.sort(key=lambda x: x[0], reverse=True)
        neighbours = neighbours[:top_k_user]

        w_sum = sum(sim * dev for sim, dev in neighbours)
        s_sum = sum(sim for sim, dev in neighbours)
        if s_sum == 0.0:
            return base, 0.0
        return base + (w_sum / s_sum), s_sum

    def predict(user, item):
        if user not in user_ratings and item not in item_ratings:
            return clamp(global_avg)
        if user not in user_ratings:
            return clamp(item_avg.get(item, get_baseline(user, item)))
        if item not in item_ratings:
            return clamp(user_avg.get(user, get_baseline(user, item)))

        item_pred, item_str = predict_item(user, item)
        user_pred, user_str = predict_user(user, item)
        base = get_baseline(user, item)

        if item_str == 0.0 and user_str == 0.0:
            return clamp(base)
        if user_str == 0.0:
            return clamp(item_pred)
        if item_str == 0.0:
            return clamp(user_pred)

        w_item = item_weight * item_str
        w_user = (1.0 - item_weight) * user_str
        pred = (w_item * item_pred + w_user * user_pred) / (w_item + w_user)
        return clamp(pred)

    def train(ratings):
        nonlocal user_ratings, item_ratings, user_avg, item_avg, user_bias, item_bias
        nonlocal item_sim_cache, user_sim_cache, global_avg

        user_ratings, item_ratings, item_sim_cache, user_sim_cache = {}, {}, {}, {}
        total_sum = 0

        for u, i, r in ratings:
            total_sum += r
            user_ratings.setdefault(u, {})[i] = r
            item_ratings.setdefault(i, {})[u] = r

        global_avg = total_sum / len(ratings) if ratings else 3.5

        for u in user_ratings:
            user_avg[u] = sum(user_ratings[u].values()) / len(user_ratings[u])
        for i in item_ratings:
            item_avg[i] = sum(item_ratings[i].values()) / len(item_ratings[i])

        for u in user_ratings:
            user_bias[u] = sum(
                r - global_avg for r in user_ratings[u].values()) / (len(user_ratings[u]) + bias_reg)

        for i in item_ratings:
            diff = sum(r - global_avg - user_bias.get(u, 0)
                       for u, r in item_ratings[i].items())
            item_bias[i] = diff / (len(item_ratings[i]) + bias_reg)

    return train, predict


def evaluate(train_fn, pred_fn, train_ratings, valid_ratings):
    train_fn(train_ratings)
    err = sum(abs(pred_fn(u, i) - r) for u, i, r in valid_ratings)
    return err / len(valid_ratings)


all_ratings = read_training_data(TRAIN_FILE)
train, valid = create_validation_split(all_ratings)

print("=" * 70)
print("HYBRID MODEL OPTIMIZATION")
print("=" * 70)

baseline_mae = 0.71883
best_mae = baseline_mae
best_config = {}

# Test ITEM_WEIGHT (most impactful)
print("\n1. ITEM_WEIGHT sweep (keeping other params at current best)")
for iw in [0.60, 0.65, 0.70, 0.72, 0.75, 0.80, 0.85]:
    train_fn, pred_fn = build_hybrid_model(
        top_k_item=80, top_k_user=60, bias_reg=5.0,
        shrinkage_item=90.0, shrinkage_user=25.0,
        min_common_users=5, min_common_items=4,
        sim_power_item=2.0, sim_power_user=1.5,
        item_weight=iw
    )
    mae = evaluate(train_fn, pred_fn, train, valid)
    is_best = " *** NEW BEST ***" if mae < best_mae else ""
    print(f"   ITEM_WEIGHT={iw:.2f}: MAE={mae:.5f}{is_best}")
    if mae < best_mae:
        best_mae = mae
        best_config = {'item_weight': iw}

print(
    f"\n   Best so far: ITEM_WEIGHT={best_config.get('item_weight', 0.72):.2f} → MAE={best_mae:.5f}")

# Test BIAS_REG around best
print("\n2. BIAS_REG sweep")
for br in [3, 4, 5, 6, 7]:
    train_fn, pred_fn = build_hybrid_model(
        top_k_item=80, top_k_user=60, bias_reg=float(br),
        shrinkage_item=90.0, shrinkage_user=25.0,
        min_common_users=5, min_common_items=4,
        sim_power_item=2.0, sim_power_user=1.5,
        item_weight=best_config.get('item_weight', 0.72)
    )
    mae = evaluate(train_fn, pred_fn, train, valid)
    is_best = " *** NEW BEST ***" if mae < best_mae else ""
    print(f"   BIAS_REG={br}.0: MAE={mae:.5f}{is_best}")
    if mae < best_mae:
        best_mae = mae
        best_config['bias_reg'] = float(br)

print(f"\n   Best so far: MAE={best_mae:.5f}")

# Test TOP_K combinations
print("\n3. TOP_K combo sweep")
for tk_i in [70, 80, 90]:
    for tk_u in [50, 60, 70]:
        train_fn, pred_fn = build_hybrid_model(
            top_k_item=tk_i, top_k_user=tk_u, bias_reg=best_config.get(
                'bias_reg', 5.0),
            shrinkage_item=90.0, shrinkage_user=25.0,
            min_common_users=5, min_common_items=4,
            sim_power_item=2.0, sim_power_user=1.5,
            item_weight=best_config.get('item_weight', 0.72)
        )
        mae = evaluate(train_fn, pred_fn, train, valid)
        is_best = " *** NEW BEST ***" if mae < best_mae else ""
        print(
            f"   TOP_K_ITEM={tk_i}, TOP_K_USER={tk_u}: MAE={mae:.5f}{is_best}")
        if mae < best_mae:
            best_mae = mae
            best_config['top_k_item'] = tk_i
            best_config['top_k_user'] = tk_u

print("\n" + "=" * 70)
print("FINAL RESULTS")
print("=" * 70)
print(f"Baseline MAE:       {baseline_mae:.5f}")
print(f"Optimized MAE:      {best_mae:.5f}")
print(
    f"Improvement:        {baseline_mae - best_mae:.5f} ({100*(baseline_mae - best_mae)/baseline_mae:.2f}%)")
print(f"\nBest config found:")
for k, v in best_config.items():
    print(f"  {k}: {v}")
print("=" * 70)
