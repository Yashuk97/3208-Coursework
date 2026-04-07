import csv
import math

TRAIN_FILE = "train_100k_withratings.csv"
TEST_FILE = "test_100k_withoutratings.csv"

ratings = []
with open(TRAIN_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)
    for row in reader:
        ratings.append((int(row[0]), int(row[1]), float(row[2])))

print(f"Total training records: {len(ratings)}")
print(f"Rating distribution:")
rating_counts = {}
for u, i, r in ratings:
    rating_counts[r] = rating_counts.get(r, 0) + 1
for r in sorted(rating_counts.keys()):
    print(
        f"  {r}: {rating_counts[r]} ({100*rating_counts[r]/len(ratings):.1f}%)")

# User/item stats
users = set(u for u, i, r in ratings)
items = set(i for u, i, r in ratings)
print(f"\nUnique users: {len(users)}")
print(f"Unique items: {len(items)}")
print(f"Sparsity: {100*(1 - len(ratings)/(len(users)*len(items))):.4f}%")

# User activity distribution
user_counts = {}
for u, i, r in ratings:
    user_counts[u] = user_counts.get(u, 0) + 1
print(
    f"User ratings per user: min={min(user_counts.values())}, max={max(user_counts.values())}, avg={sum(user_counts.values())/len(user_counts):.1f}")

# Item popularity distribution
item_counts = {}
for u, i, r in ratings:
    item_counts[i] = item_counts.get(i, 0) + 1
print(
    f"Item ratings per item: min={min(item_counts.values())}, max={max(item_counts.values())}, avg={sum(item_counts.values())/len(item_counts):.1f}")

# Test set stats
test_rows = []
with open(TEST_FILE, "r") as f:
    reader = csv.reader(f)
    for row in reader:
        test_rows.append((int(row[0]), int(row[1])))

print(f"\nTest set: {len(test_rows)} predictions")
test_users = set(u for u, i in test_rows)
test_items = set(i for u, i in test_rows)
test_new_users = len(test_users - users)
test_new_items = len(test_items - items)
print(
    f"  Users in test: {len(test_users)} ({test_new_users} new, {len(test_users & users)} known)")
print(
    f"  Items in test: {len(test_items)} ({test_new_items} new, {len(test_items & items)} known)")

# Cold-start prediction fallback analysis
print(f"\nCold-start predictions in test:")
cold_user_only = sum(1 for u, i in test_rows if u not in users)
cold_item_only = sum(1 for u, i in test_rows if i not in items)
cold_both = sum(1 for u, i in test_rows if u not in users and i not in items)
warm = len(test_rows) - cold_user_only - cold_item_only - cold_both
print(f"  Warm (both known): {warm}")
print(f"  Cold (new user): {cold_user_only}")
print(f"  Cold (new item): {cold_item_only}")
print(f"  Cold (both new): {cold_both}")

# Compute global stats
global_avg = sum(r for u, i, r in ratings) / len(ratings)
print(f"\nGlobal average rating: {global_avg:.4f}")
print(
    f"Std dev: {math.sqrt(sum((r - global_avg)**2 for u, i, r in ratings) / len(ratings)):.4f}")
