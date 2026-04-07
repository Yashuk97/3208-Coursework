"""
Improved recommendation system with 4 targeted fixes:
1. Lower SIM_POWER (2.5 → 1.8) to reduce underdispersion
2. Lower BIAS_REG (25.0 → 12.0) for better personalization
3. Outlier clipping in training data
4. Pearson correlation instead of adjusted cosine
"""

import csv
import math

TRAIN_FILE = "train_100k_withratings.csv"
TEST_FILE = "test_100k_withoutratings.csv"
OUTPUT_FILE = "results.csv"

# Improved hyper-parameters
PARAM_GRID = [
    {"TOP_K": 80, "BIAS_REG": 12.0, "SHRINKAGE": 100.0,
        "MIN_COMMON_USERS": 5, "SIM_POWER": 1.8},
    {"TOP_K": 100, "BIAS_REG": 12.0, "SHRINKAGE": 120.0,
        "MIN_COMMON_USERS": 5, "SIM_POWER": 1.8}
]

SIM_THRESHOLD = 0.0
OUTLIER_SIGMA = 3.0  # NEW: Clip deviations > 3 sigma from user mean

# Global storage
user_ratings = {}
item_ratings = {}
user_avg = {}
item_avg = {}
user_bias = {}
item_bias = {}
similarity_cache = {}
global_avg = 0.0

# Current active parameters
TOP_K = 80
BIAS_REG = 12.0
SHRINKAGE = 100.0
MIN_COMMON_USERS = 5
SIM_POWER = 1.8


def read_training_data(path):
    ratings = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            ratings.append((int(row[0]), int(row[1]), float(row[2])))
    return ratings


def read_test_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            user, item = int(row[0]), int(row[1])
            ts = row[2] if len(row) > 2 else "0"
            rows.append((user, item, ts))
    return rows


def create_validation_split(ratings):
    train, valid = [], []
    for idx, r in enumerate(ratings):
        if idx % 10 == 0:
            valid.append(r)
        else:
            train.append(r)
    return train, valid


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


def get_pearson_similarity(item1, item2):
    """
    Pearson correlation-based similarity.
    More robust to outliers and scale variance than adjusted cosine.
    """
    key = tuple(sorted((item1, item2)))
    if key in similarity_cache:
        return similarity_cache[key]

    u1 = item_ratings.get(item1, {})
    u2 = item_ratings.get(item2, {})

    common_users = set(u1.keys()) & set(u2.keys())
    n_common = len(common_users)

    if n_common < MIN_COMMON_USERS:
        similarity_cache[key] = 0.0
        return 0.0

    # Compute Pearson correlation
    ratings_1 = [u1[u] for u in common_users]
    ratings_2 = [u2[u] for u in common_users]

    mean_1 = sum(ratings_1) / n_common
    mean_2 = sum(ratings_2) / n_common

    num = sum((ratings_1[i] - mean_1) * (ratings_2[i] - mean_2)
              for i in range(n_common))
    d1 = sum((r - mean_1) ** 2 for r in ratings_1)
    d2 = sum((r - mean_2) ** 2 for r in ratings_2)

    if d1 == 0 or d2 == 0:
        similarity_cache[key] = 0.0
        return 0.0

    sim = num / (math.sqrt(d1) * math.sqrt(d2))

    # Significance weighting (same as before)
    sim *= (n_common / (n_common + SHRINKAGE))

    # Apply similarity power (now 1.8 instead of 2.5)
    if sim > 0:
        sim = math.pow(sim, SIM_POWER)
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
        sim = get_pearson_similarity(item, other_item)
        if sim > SIM_THRESHOLD:
            dev = rating - get_baseline(user, other_item)
            neighbours.append((sim, dev))

    if not neighbours:
        return clamp(base_ui)

    neighbours.sort(key=lambda x: x[0], reverse=True)
    neighbours = neighbours[:TOP_K]

    w_sum = sum(sim * dev for sim, dev in neighbours)
    s_sim = sum(sim for sim, dev in neighbours)

    pred = base_ui + (w_sum / s_sim)
    return clamp(pred)


def train_model(ratings, params):
    global user_ratings, item_ratings, user_avg, item_avg
    global user_bias, item_bias, global_avg, similarity_cache
    global TOP_K, BIAS_REG, SHRINKAGE, MIN_COMMON_USERS, SIM_POWER

    TOP_K = params["TOP_K"]
    BIAS_REG = params["BIAS_REG"]
    SHRINKAGE = params["SHRINKAGE"]
    MIN_COMMON_USERS = params["MIN_COMMON_USERS"]
    SIM_POWER = params["SIM_POWER"]

    user_ratings, item_ratings, similarity_cache = {}, {}, {}
    total_sum = 0
    n_clipped = 0

    # FIX #3: Outlier clipping in training data
    # First pass: compute user means
    user_sums = {}
    user_counts = {}
    for u, i, r in ratings:
        user_sums[u] = user_sums.get(u, 0) + r
        user_counts[u] = user_counts.get(u, 0) + 1

    user_means = {u: user_sums[u] / user_counts[u] for u in user_sums}

    # Compute per-user std
    user_sq_diffs = {}
    for u, i, r in ratings:
        user_sq_diffs[u] = user_sq_diffs.get(u, 0) + (r - user_means[u]) ** 2

    user_stds = {u: math.sqrt(
        user_sq_diffs[u] / user_counts[u]) for u in user_sq_diffs}

    # Filter ratings: keep those within 3 sigma of user mean
    filtered_ratings = []
    for u, i, r in ratings:
        if user_stds[u] == 0:
            filtered_ratings.append((u, i, r))
        elif abs(r - user_means[u]) <= OUTLIER_SIGMA * user_stds[u]:
            filtered_ratings.append((u, i, r))
        else:
            n_clipped += 1

    print(
        f"[Outlier clipping] Removed {n_clipped} ratings (~{100*n_clipped/len(ratings):.2f}%)")
    ratings = filtered_ratings

    # Build data structures
    for u, i, r in ratings:
        total_sum += r
        user_ratings.setdefault(u, {})[i] = r
        item_ratings.setdefault(i, {})[u] = r

    global_avg = total_sum / len(ratings) if ratings else 3.5

    for u, r_map in user_ratings.items():
        user_avg[u] = sum(r_map.values()) / len(r_map)

    for i, u_map in item_ratings.items():
        item_avg[i] = sum(u_map.values()) / len(u_map)

    # FIX #2: Lower bias regularization for better personalization
    for u, r_map in user_ratings.items():
        user_bias[u] = sum(
            r - global_avg for r in r_map.values()) / (len(r_map) + BIAS_REG)

    for i, u_map in item_ratings.items():
        diff = sum(r - global_avg - user_bias.get(u, 0)
                   for u, r in u_map.items())
        item_bias[i] = diff / (len(u_map) + BIAS_REG)


def main():
    print("Loading data...")
    all_ratings = read_training_data(TRAIN_FILE)
    train, valid = create_validation_split(all_ratings)

    best_mae = 1.0
    best_p = PARAM_GRID[0]

    for p in PARAM_GRID:
        print(f"Testing: {p}")
        train_model(train, p)

        err = 0
        for u, i, r in valid:
            err += abs(predict(u, i) - r)
        curr_mae = err / len(valid)
        print(f"Local MAE: {curr_mae:.5f}")

        if curr_mae < best_mae:
            best_mae = curr_mae
            best_p = p

    print(f"\nFinal training on best params: {best_p}")
    print(f"Best validation MAE: {best_mae:.5f}")
    train_model(all_ratings, best_p)
    test_rows = read_test_data(TEST_FILE)

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["userid", "itemid", "rating", "timestamp"])
        for u, i, ts in test_rows:
            p = predict(u, i)
            writer.writerow([u, i, f"{p:.4f}", ts])

    print(f"File {OUTPUT_FILE} is ready for submission.")


if __name__ == "__main__":
    main()
