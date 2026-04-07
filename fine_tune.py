"""
Fine-tuning around BIAS_REG=5 baseline. 
Test SHRINKAGE and SIM_POWER to see if we can push even further.
"""

import csv
import math

TRAIN_FILE = "train_100k_withratings.csv"


def read_training_data(path):
    ratings = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
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


def build_model(bias_reg=25.0, top_k=80, shrinkage=100.0, min_common=5, sim_power=2.5):
    user_ratings = {}
    item_ratings = {}
    user_avg = {}
    item_avg = {}
    user_bias = {}
    item_bias = {}
    similarity_cache = {}
    global_avg = 3.5

    def clamp(x):
        if x != x:
            return global_avg
        return max(0.5, min(5.0, x))

    def get_baseline(user, item):
        res = global_avg
        if user in user_bias:
            res += user_bias[user]
        if item in item_bias:
            res += item_bias[item]
        return res

    def get_similarity(item1, item2):
        key = tuple(sorted((item1, item2)))
        if key in similarity_cache:
            return similarity_cache[key]

        u1 = item_ratings.get(item1, {})
        u2 = item_ratings.get(item2, {})
        common_users = set(u1.keys()) & set(u2.keys())
        n_common = len(common_users)

        if n_common < min_common:
            similarity_cache[key] = 0.0
            return 0.0

        num, d1, d2 = 0.0, 0.0, 0.0
        for u in common_users:
            u_mean = user_avg[u]
            v1 = u1[u] - u_mean
            v2 = u2[u] - u_mean
            num += v1 * v2
            d1 += v1**2
            d2 += v2**2

        if d1 == 0 or d2 == 0:
            similarity_cache[key] = 0.0
            return 0.0

        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (n_common / (n_common + shrinkage))
        if sim > 0:
            sim = math.pow(sim, sim_power)
        else:
            sim = 0

        similarity_cache[key] = sim
        return sim

    def predict(user, item):
        base_ui = get_baseline(user, item)
        if user not in user_ratings:
            return clamp(item_avg.get(item, global_avg))
        if item not in item_ratings:
            return clamp(user_avg.get(user, global_avg))
        neighbours = []
        for other_item, rating in user_ratings[user].items():
            sim = get_similarity(item, other_item)
            if sim > 0:
                dev = rating - get_baseline(user, other_item)
                neighbours.append((sim, dev))
        if not neighbours:
            return clamp(base_ui)
        neighbours.sort(key=lambda x: x[0], reverse=True)
        neighbours = neighbours[:top_k]
        w_sum = sum(sim * dev for sim, dev in neighbours)
        s_sim = sum(sim for sim, dev in neighbours)
        pred = base_ui + (w_sum / s_sim)
        return clamp(pred)

    def train(ratings):
        nonlocal user_ratings, item_ratings, user_avg, item_avg, user_bias, item_bias, global_avg, similarity_cache
        user_ratings, item_ratings, similarity_cache = {}, {}, {}
        total_sum = 0

        for u, i, r in ratings:
            total_sum += r
            user_ratings.setdefault(u, {})[i] = r
            item_ratings.setdefault(i, {})[u] = r

        global_avg = total_sum / len(ratings) if ratings else 3.5

        for u, r_map in user_ratings.items():
            user_avg[u] = sum(r_map.values()) / len(r_map)
        for i, u_map in item_ratings.items():
            item_avg[i] = sum(u_map.values()) / len(u_map)

        for u, r_map in user_ratings.items():
            user_bias[u] = sum(
                r - global_avg for r in r_map.values()) / (len(r_map) + bias_reg)
        for i, u_map in item_ratings.items():
            diff = sum(r - global_avg - user_bias.get(u, 0)
                       for u, r in u_map.items())
            item_bias[i] = diff / (len(u_map) + bias_reg)

    return train, predict


def evaluate(train_fn, pred_fn, train_ratings, valid_ratings):
    train_fn(train_ratings)
    err = sum(abs(pred_fn(u, i) - r) for u, i, r in valid_ratings)
    return err / len(valid_ratings)


all_ratings = read_training_data(TRAIN_FILE)
train, valid = create_validation_split(all_ratings)

print("=" * 70)
print("FINE-TUNING: SHRINKAGE + SIM_POWER around BIAS_REG=5 baseline")
print("=" * 70)
print()

best_mae = 0.72890
best_params = {'bias_reg': 5, 'top_k': 80, 'shrinkage': 100, 'sim_power': 2.5}

# Test SHRINKAGE values
print("1. SHRINKAGE sweep (BIAS_REG=5, TOP_K=80)")
shrinkages = [50, 75, 80, 90, 100, 120, 150, 200]
results_shrink = []

for sh in shrinkages:
    train_fn, pred_fn = build_model(
        bias_reg=5, top_k=80, shrinkage=sh, min_common=5, sim_power=2.5)
    mae = evaluate(train_fn, pred_fn, train, valid)
    results_shrink.append((sh, mae))
    is_best = " *** BEST ***" if mae < best_mae else ""
    print(f"   SHRINKAGE={sh:3d}: MAE={mae:.5f}{is_best}")
    if mae < best_mae:
        best_mae = mae
        best_params = {'bias_reg': 5, 'top_k': 80,
                       'shrinkage': sh, 'sim_power': 2.5}

best_shrink = min(results_shrink, key=lambda x: x[1])
print(f"\n   BEST SHRINKAGE: {best_shrink[0]} with MAE={best_shrink[1]:.5f}")

# Test SIM_POWER values around best shrinkage
print(
    f"\n2. SIM_POWER sweep (BIAS_REG=5, TOP_K=80, SHRINKAGE={best_shrink[0]})")
sim_powers = [1.5, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0, 3.5]
results_sim = []

for sp in sim_powers:
    train_fn, pred_fn = build_model(
        bias_reg=5, top_k=80, shrinkage=best_shrink[0], min_common=5, sim_power=sp)
    mae = evaluate(train_fn, pred_fn, train, valid)
    results_sim.append((sp, mae))
    is_best = " *** BEST ***" if mae < best_mae else ""
    print(f"   SIM_POWER={sp:.1f}: MAE={mae:.5f}{is_best}")
    if mae < best_mae:
        best_mae = mae
        best_params = {'bias_reg': 5, 'top_k': 80,
                       'shrinkage': best_shrink[0], 'sim_power': sp}

best_sim = min(results_sim, key=lambda x: x[1])
print(f"\n   BEST SIM_POWER: {best_sim[0]} with MAE={best_sim[1]:.5f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
baseline = 0.73359
print(f"Original baseline:        MAE={baseline:.5f}")
print(f"After BIAS_REG=5:         MAE=0.72890")
print(f"Best after fine-tuning:   MAE={best_mae:.5f}")
improvement = baseline - best_mae
pct = 100 * improvement / baseline
print(f"Total improvement:        {improvement:.5f} ({pct:+.2f}%)")
print(f"\nOptimal params found:")
for k, v in best_params.items():
    print(f"  {k}: {v}")
print(f"\nEstimated server MAE: {0.714 * (1 - improvement/baseline):.5f}")
print("=" * 70)
