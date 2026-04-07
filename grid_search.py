"""
Hyperparameter search focusing on BIAS_REG and other promising params.
Since most fixes showed no improvement, search for the truly optimal values.
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
print("HYPERPARAMETER GRID SEARCH")
print("=" * 70)
print()

# Test different BIAS_REG values
print("1. BIAS_REG sweep (keeping other params at optimal from original)")
bias_regs = [5, 8, 10, 12, 15, 20, 25, 30, 35, 40]
results_bias_reg = []

for br in bias_regs:
    train_fn, pred_fn = build_model(
        bias_reg=br, top_k=80, shrinkage=100.0, min_common=5, sim_power=2.5)
    mae = evaluate(train_fn, pred_fn, train, valid)
    results_bias_reg.append((br, mae))
    print(f"   BIAS_REG={br:2d}: MAE={mae:.5f}")

best_bias_reg = min(results_bias_reg, key=lambda x: x[1])
print(
    f"\n   BEST: BIAS_REG={best_bias_reg[0]} with MAE={best_bias_reg[1]:.5f}")

# Test combinations of top_k + bias_reg
print("\n2. TOP_K + BIAS_REG combination (around best BIAS_REG)")
top_ks = [60, 80, 100, 120]
results_combo = []

for tk in top_ks:
    train_fn, pred_fn = build_model(
        bias_reg=best_bias_reg[0], top_k=tk, shrinkage=100.0, min_common=5, sim_power=2.5)
    mae = evaluate(train_fn, pred_fn, train, valid)
    results_combo.append((tk, mae))
    print(f"   TOP_K={tk:3d}, BIAS_REG={best_bias_reg[0]:2d}: MAE={mae:.5f}")

best_combo = min(results_combo, key=lambda x: x[1])
print(
    f"\n   BEST: TOP_K={best_combo[0]}, BIAS_REG={best_bias_reg[0]} with MAE={best_combo[1]:.5f}")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
baseline_mae = 0.73359
best_mae = best_combo[1]
improvement = baseline_mae - best_mae
pct_improvement = 100 * improvement / baseline_mae

print(f"Baseline (original):      MAE={baseline_mae:.5f}")
print(f"Best found:               MAE={best_mae:.5f}")
print(f"Improvement:              {improvement:.5f} ({pct_improvement:+.2f}%)")
print(
    f"Estimated server impact:  {0.714 * (1 - improvement/baseline_mae):.5f}")
print("=" * 70)
