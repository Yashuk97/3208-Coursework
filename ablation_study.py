"""
Ablation study: Measure impact of each improvement independently and combined.
"""

import csv
import math

TRAIN_FILE = "train_100k_withratings.csv"
TEST_FILE = "test_100k_withoutratings.csv"


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

# ============ Model Variants ============


def build_model_variant(use_pearson=False, sim_power=2.5, bias_reg=25.0, clip_outliers=False):
    """Factory function to create a model variant."""

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

        if n_common < 5:
            similarity_cache[key] = 0.0
            return 0.0

        if use_pearson:
            # Pearson correlation
            ratings_1 = [u1[u] for u in common_users]
            ratings_2 = [u2[u] for u in common_users]
            mean_1 = sum(ratings_1) / n_common
            mean_2 = sum(ratings_2) / n_common
            num = sum((ratings_1[i] - mean_1) * (ratings_2[i] - mean_2)
                      for i in range(n_common))
            d1 = sum((r - mean_1) ** 2 for r in ratings_1)
            d2 = sum((r - mean_2) ** 2 for r in ratings_2)
        else:
            # Adjusted cosine
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
        sim *= (n_common / (n_common + 100.0))
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
        neighbours = neighbours[:80]
        w_sum = sum(sim * dev for sim, dev in neighbours)
        s_sim = sum(sim for sim, dev in neighbours)
        pred = base_ui + (w_sum / s_sim)
        return clamp(pred)

    def train(ratings):
        nonlocal user_ratings, item_ratings, user_avg, item_avg, user_bias, item_bias, global_avg, similarity_cache
        user_ratings, item_ratings, similarity_cache = {}, {}, {}
        total_sum = 0
        n_clipped = 0

        # FIX: Outlier clipping
        if clip_outliers:
            user_sums = {}
            user_counts = {}
            for u, i, r in ratings:
                user_sums[u] = user_sums.get(u, 0) + r
                user_counts[u] = user_counts.get(u, 0) + 1
            user_means = {u: user_sums[u] / user_counts[u] for u in user_sums}
            user_sq_diffs = {}
            for u, i, r in ratings:
                user_sq_diffs[u] = user_sq_diffs.get(
                    u, 0) + (r - user_means[u]) ** 2
            user_stds = {u: math.sqrt(
                user_sq_diffs[u] / user_counts[u]) for u in user_sq_diffs}

            filtered = []
            for u, i, r in ratings:
                if user_stds[u] == 0 or abs(r - user_means[u]) <= 3.0 * user_stds[u]:
                    filtered.append((u, i, r))
                else:
                    n_clipped += 1
            ratings = filtered

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

# ============ Evaluation ============


def evaluate_mae(model_train, model_predict, train_ratings, valid_ratings):
    model_train(train_ratings)
    err = sum(abs(model_predict(u, i) - r) for u, i, r in valid_ratings)
    return err / len(valid_ratings)

# ============ Main ============


all_ratings = read_training_data(TRAIN_FILE)
train, valid = create_validation_split(all_ratings)

print("=" * 60)
print("ABLATION STUDY: Impact of each improvement")
print("=" * 60)

# Baseline
print("\n1. BASELINE (original settings)")
train_fn, pred_fn = build_model_variant(
    use_pearson=False, sim_power=2.5, bias_reg=25.0, clip_outliers=False)
mae_baseline = evaluate_mae(train_fn, pred_fn, train, valid)
print(f"   MAE: {mae_baseline:.5f}")

# Fix 1: Lower SIM_POWER
print("\n2. + Lower SIM_POWER (2.5 → 1.8)")
train_fn, pred_fn = build_model_variant(
    use_pearson=False, sim_power=1.8, bias_reg=25.0, clip_outliers=False)
mae_sim_power = evaluate_mae(train_fn, pred_fn, train, valid)
delta = mae_baseline - mae_sim_power
print(
    f"   MAE: {mae_sim_power:.5f} (Δ: {delta:+.5f}, {100*delta/mae_baseline:+.2f}%)")

# Fix 2: Lower BIAS_REG
print("\n3. + Lower BIAS_REG (25.0 → 12.0)")
train_fn, pred_fn = build_model_variant(
    use_pearson=False, sim_power=2.5, bias_reg=12.0, clip_outliers=False)
mae_bias_reg = evaluate_mae(train_fn, pred_fn, train, valid)
delta = mae_baseline - mae_bias_reg
print(
    f"   MAE: {mae_bias_reg:.5f} (Δ: {delta:+.5f}, {100*delta/mae_baseline:+.2f}%)")

# Fix 3: Outlier Clipping
print("\n4. + Outlier Clipping (3σ)")
train_fn, pred_fn = build_model_variant(
    use_pearson=False, sim_power=2.5, bias_reg=25.0, clip_outliers=True)
mae_clip = evaluate_mae(train_fn, pred_fn, train, valid)
delta = mae_baseline - mae_clip
print(
    f"   MAE: {mae_clip:.5f} (Δ: {delta:+.5f}, {100*delta/mae_baseline:+.2f}%)")

# Fix 4: Pearson Correlation
print("\n5. + Pearson Correlation")
train_fn, pred_fn = build_model_variant(
    use_pearson=True, sim_power=2.5, bias_reg=25.0, clip_outliers=False)
mae_pearson = evaluate_mae(train_fn, pred_fn, train, valid)
delta = mae_baseline - mae_pearson
print(
    f"   MAE: {mae_pearson:.5f} (Δ: {delta:+.5f}, {100*delta/mae_baseline:+.2f}%)")

# All 4 combined
print("\n6. ALL 4 FIXES COMBINED")
train_fn, pred_fn = build_model_variant(
    use_pearson=True, sim_power=1.8, bias_reg=12.0, clip_outliers=True)
mae_combined = evaluate_mae(train_fn, pred_fn, train, valid)
delta = mae_baseline - mae_combined
pct_delta = 100 * delta / mae_baseline
print(f"   MAE: {mae_combined:.5f} (Δ: {delta:+.5f}, {pct_delta:+.2f}%)")

# Summary
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Baseline:           {mae_baseline:.5f}")
print(f"Combined (all 4):   {mae_combined:.5f}")
print(f"Improvement:        {delta:.5f} ({pct_delta:.2f}%)")
print(f"\nEstimated server MAE (0.714 * improvement):")
print(f"  {0.714 * (1 - abs(pct_delta/100)):.5f} (if proportional)")
print("=" * 60)
