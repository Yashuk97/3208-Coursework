import csv
import math

# Run the model once to get predictions
TRAIN_FILE = "train_100k_withratings.csv"
TEST_FILE = "test_100k_withoutratings.csv"
OUTPUT_FILE = "results.csv"

# Load model code
user_ratings = {}
item_ratings = {}
user_avg = {}
item_avg = {}
user_bias = {}
item_bias = {}
similarity_cache = {}
global_avg = 0.0

TOP_K = 80
BIAS_REG = 25.0
SHRINKAGE = 100.0
MIN_COMMON_USERS = 5
SIM_POWER = 2.5


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
    if n_common < MIN_COMMON_USERS:
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
    sim *= (n_common / (n_common + SHRINKAGE))
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
        sim = get_similarity(item, other_item)
        if sim > 0:
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


def train_model(ratings):
    global user_ratings, item_ratings, user_avg, item_avg
    global user_bias, item_bias, global_avg, similarity_cache
    user_ratings, item_ratings, similarity_cache = {}, {}, {}
    total_sum = 0
    for u, i, r in ratings:
        total_sum += r
        user_ratings.setdefault(u, {})[i] = r
        item_ratings.setdefault(i, {})[u] = r
    global_avg = total_sum / len(ratings)
    for u, r_map in user_ratings.items():
        user_avg[u] = sum(r_map.values()) / len(r_map)
    for i, u_map in item_ratings.items():
        item_avg[i] = sum(u_map.values()) / len(u_map)
    for u, r_map in user_ratings.items():
        user_bias[u] = sum(
            r - global_avg for r in r_map.values()) / (len(r_map) + BIAS_REG)
    for i, u_map in item_ratings.items():
        diff = sum(r - global_avg - user_bias.get(u, 0)
                   for u, r in u_map.items())
        item_bias[i] = diff / (len(u_map) + BIAS_REG)


# Train on full set
all_ratings = read_training_data(TRAIN_FILE)
train_model(all_ratings)

# Get predictions
test_rows = read_test_data(TEST_FILE)
predictions = []
for u, i, ts in test_rows:
    p = predict(u, i)
    predictions.append(p)

# Analyze predictions
print(f"Prediction distribution:")
pred_counts = {k: 0 for k in [
    0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]}
for p in predictions:
    bucket = round(p * 2) / 2
    if bucket not in pred_counts:
        pred_counts[bucket] = 0
    pred_counts[bucket] += 1

print(f"  Min: {min(predictions):.4f}, Max: {max(predictions):.4f}, Mean: {sum(predictions)/len(predictions):.4f}")
print(f"  Std: {math.sqrt(sum((p - sum(predictions)/len(predictions))**2 for p in predictions)/len(predictions)):.4f}")
print(f"\nPrediction buckets (rounded to 0.5):")
for k in sorted(pred_counts.keys()):
    if pred_counts[k] > 0:
        print(f"  {k}: {pred_counts[k]}")

# Check for mode prediction
freq = {}
for p in predictions:
    p_round = round(p * 2) / 2
    freq[p_round] = freq.get(p_round, 0) + 1
top_pred = max(freq.items(), key=lambda x: x[1])
print(
    f"\nMost common prediction: {top_pred[0]} ({top_pred[1]} times, {100*top_pred[1]/len(predictions):.1f}%)")
