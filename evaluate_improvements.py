"""
Ablation study: evaluate 4 improvements to reduce server-side MAE.
Each improvement is tested independently and then combined.

Improvements:
1. Optimal rounding (nearest integer, nearest 0.5, optimized thresholds)
2. Negative similarity utilization
3. Weighted average smoothing (blend KNN with baseline)
4. Confidence-adaptive item/user blending
"""

import csv
import math
from collections import defaultdict


TRAIN_FILE = "train_100k_withratings.csv"


# ─── Data loading ───────────────────────────────────────────────────────

def read_training_data(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            try:
                rows.append((int(row[0]), int(row[1]), float(row[2])))
            except ValueError:
                continue
    return rows


def create_validation_split(ratings):
    train, valid = [], []
    for idx, r in enumerate(ratings):
        if idx % 10 == 0:
            valid.append(r)
        else:
            train.append(r)
    return train, valid


# ─── Model builder ──────────────────────────────────────────────────────

def build_model(
    top_k_item=40, top_k_user=20,
    bias_reg=12.0,
    shrinkage_item=110.0, shrinkage_user=25.0,
    min_common_users=3, min_common_items=3,
    sim_power_item=1.6, sim_power_user=1.1,
    item_weight=0.95,
    # New flags for improvements
    use_negative_sim=False,
    baseline_lambda=0.0,        # 0 = no smoothing
    adaptive_blend=False,
    rounding_mode="none",       # "none", "int", "half", "optimized"
):
    """Build model with configurable improvements. Returns (train_fn, predict_fn)."""

    state = {
        "user_ratings": {}, "item_ratings": {},
        "user_avg": {}, "item_avg": {},
        "user_bias": {}, "item_bias": {},
        "item_sim_cache": {}, "user_sim_cache": {},
        "global_avg": 0.0,
    }

    def clamp(x):
        if x != x:
            return state["global_avg"]
        return max(0.5, min(5.0, x))

    def apply_rounding(x):
        if rounding_mode == "int":
            return max(1.0, min(5.0, round(x)))
        elif rounding_mode == "half":
            return max(0.5, min(5.0, round(x * 2) / 2))
        return x

    def get_baseline(u, i):
        return (state["global_avg"]
                + state["user_bias"].get(u, 0.0)
                + state["item_bias"].get(i, 0.0))

    def get_item_sim(item1, item2):
        if item1 == item2:
            return 1.0
        key = (item1, item2) if item1 < item2 else (item2, item1)
        cache = state["item_sim_cache"]
        if key in cache:
            return cache[key]

        r1 = state["item_ratings"].get(item1, {})
        r2 = state["item_ratings"].get(item2, {})
        common = set(r1.keys()) & set(r2.keys())
        n_common = len(common)

        if n_common < min_common_users:
            cache[key] = 0.0
            return 0.0

        num, d1, d2 = 0.0, 0.0, 0.0
        for u in common:
            mu = state["user_avg"][u]
            v1, v2 = r1[u] - mu, r2[u] - mu
            num += v1 * v2
            d1 += v1 * v1
            d2 += v2 * v2

        if d1 == 0.0 or d2 == 0.0:
            cache[key] = 0.0
            return 0.0

        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (n_common / (n_common + shrinkage_item))

        if use_negative_sim:
            if sim > 0.0:
                sim = math.pow(sim, sim_power_item)
            else:
                sim = -math.pow(abs(sim), sim_power_item)
        else:
            if sim > 0.0:
                sim = math.pow(sim, sim_power_item)
            else:
                sim = 0.0

        cache[key] = sim
        return sim

    def get_user_sim(u1, u2):
        if u1 == u2:
            return 1.0
        key = (u1, u2) if u1 < u2 else (u2, u1)
        cache = state["user_sim_cache"]
        if key in cache:
            return cache[key]

        r1 = state["user_ratings"].get(u1, {})
        r2 = state["user_ratings"].get(u2, {})
        common = set(r1.keys()) & set(r2.keys())
        n_common = len(common)

        if n_common < min_common_items:
            cache[key] = 0.0
            return 0.0

        mu1, mu2 = state["user_avg"][u1], state["user_avg"][u2]
        num, d1, d2 = 0.0, 0.0, 0.0
        for i in common:
            v1, v2 = r1[i] - mu1, r2[i] - mu2
            num += v1 * v2
            d1 += v1 * v1
            d2 += v2 * v2

        if d1 == 0.0 or d2 == 0.0:
            cache[key] = 0.0
            return 0.0

        sim = num / (math.sqrt(d1) * math.sqrt(d2))
        sim *= (n_common / (n_common + shrinkage_user))

        if use_negative_sim:
            if sim > 0.0:
                sim = math.pow(sim, sim_power_user)
            else:
                sim = -math.pow(abs(sim), sim_power_user)
        else:
            if sim > 0.0:
                sim = math.pow(sim, sim_power_user)
            else:
                sim = 0.0

        cache[key] = sim
        return sim

    def predict_item_component(user, item):
        base_ui = get_baseline(user, item)
        if user not in state["user_ratings"] or item not in state["item_ratings"]:
            return base_ui, 0.0

        neighbours = []
        for other_item, rating in state["user_ratings"][user].items():
            if other_item == item:
                continue
            sim = get_item_sim(item, other_item)
            if use_negative_sim:
                if sim != 0.0:
                    dev = rating - get_baseline(user, other_item)
                    neighbours.append((sim, dev))
            else:
                if sim > 0.0:
                    dev = rating - get_baseline(user, other_item)
                    neighbours.append((sim, dev))

        if not neighbours:
            return base_ui, 0.0

        neighbours.sort(key=lambda x: abs(x[0]), reverse=True)
        neighbours = neighbours[:top_k_item]

        weighted_sum = sum(sim * dev for sim, dev in neighbours)
        sim_sum = sum(abs(sim) for sim, dev in neighbours)

        if sim_sum == 0.0:
            return base_ui, 0.0

        raw_pred = base_ui + (weighted_sum / sim_sum)

        # Improvement 3: weighted average smoothing
        if baseline_lambda > 0.0:
            raw_pred = (sim_sum * raw_pred + baseline_lambda * base_ui) / (sim_sum + baseline_lambda)

        return raw_pred, sim_sum

    def predict_user_component(user, item):
        base_ui = get_baseline(user, item)
        if user not in state["user_ratings"] or item not in state["item_ratings"]:
            return base_ui, 0.0

        neighbours = []
        for other_user, rating in state["item_ratings"][item].items():
            if other_user == user:
                continue
            sim = get_user_sim(user, other_user)
            if use_negative_sim:
                if sim != 0.0:
                    dev = rating - get_baseline(other_user, item)
                    neighbours.append((sim, dev))
            else:
                if sim > 0.0:
                    dev = rating - get_baseline(other_user, item)
                    neighbours.append((sim, dev))

        if not neighbours:
            return base_ui, 0.0

        neighbours.sort(key=lambda x: abs(x[0]), reverse=True)
        neighbours = neighbours[:top_k_user]

        weighted_sum = sum(sim * dev for sim, dev in neighbours)
        sim_sum = sum(abs(sim) for sim, dev in neighbours)

        if sim_sum == 0.0:
            return base_ui, 0.0

        raw_pred = base_ui + (weighted_sum / sim_sum)

        if baseline_lambda > 0.0:
            raw_pred = (sim_sum * raw_pred + baseline_lambda * base_ui) / (sim_sum + baseline_lambda)

        return raw_pred, sim_sum

    def predict(user, item):
        if user not in state["user_ratings"] and item not in state["item_ratings"]:
            return apply_rounding(clamp(state["global_avg"]))

        base = get_baseline(user, item)

        if user not in state["user_ratings"]:
            return apply_rounding(clamp(base))

        if item not in state["item_ratings"]:
            return apply_rounding(clamp(base))

        item_pred, item_str = predict_item_component(user, item)
        user_pred, user_str = predict_user_component(user, item)

        if item_str == 0.0 and user_str == 0.0:
            return apply_rounding(clamp(base))

        if user_str == 0.0:
            return apply_rounding(clamp(item_pred))

        if item_str == 0.0:
            return apply_rounding(clamp(user_pred))

        # Improvement 4: confidence-adaptive blending
        if adaptive_blend:
            # Weight proportional to confidence (total sim strength)
            w_item = item_str
            w_user = user_str
        else:
            w_item = item_weight * item_str
            w_user = (1.0 - item_weight) * user_str

        pred = (w_item * item_pred + w_user * user_pred) / (w_item + w_user)
        return apply_rounding(clamp(pred))

    def train(ratings):
        state["user_ratings"] = {}
        state["item_ratings"] = {}
        state["user_avg"] = {}
        state["item_avg"] = {}
        state["user_bias"] = {}
        state["item_bias"] = {}
        state["item_sim_cache"] = {}
        state["user_sim_cache"] = {}

        total_sum = 0.0
        for u, i, r in ratings:
            total_sum += r
            state["user_ratings"].setdefault(u, {})[i] = r
            state["item_ratings"].setdefault(i, {})[u] = r

        state["global_avg"] = total_sum / len(ratings)

        for u, rm in state["user_ratings"].items():
            state["user_avg"][u] = sum(rm.values()) / len(rm)
        for i, rm in state["item_ratings"].items():
            state["item_avg"][i] = sum(rm.values()) / len(rm)

        # Iterative ALS bias estimation (10 iterations)
        for u in state["user_ratings"]:
            state["user_bias"][u] = 0.0
        for i in state["item_ratings"]:
            state["item_bias"][i] = 0.0

        for _ in range(10):
            for u, rm in state["user_ratings"].items():
                res = sum(r - state["global_avg"] - state["item_bias"].get(i, 0.0)
                          for i, r in rm.items())
                state["user_bias"][u] = res / (len(rm) + bias_reg)

            for i, rm in state["item_ratings"].items():
                res = sum(r - state["global_avg"] - state["user_bias"].get(u, 0.0)
                          for u, r in rm.items())
                state["item_bias"][i] = res / (len(rm) + bias_reg)

    return train, predict


def evaluate_mae(pred_fn, valid_rows):
    total = sum(abs(pred_fn(u, i) - r) for u, i, r in valid_rows)
    return total / len(valid_rows)


# ─── Main ablation study ────────────────────────────────────────────────

def main():
    print("Loading data...")
    all_ratings = read_training_data(TRAIN_FILE)
    train, valid = create_validation_split(all_ratings)
    print(f"Train: {len(train)}, Valid: {len(valid)}")

    # Current best params from assignment1.py
    base_params = {
        "top_k_item": 40, "top_k_user": 20,
        "bias_reg": 12.0,
        "shrinkage_item": 110.0, "shrinkage_user": 25.0,
        "min_common_users": 3, "min_common_items": 3,
        "sim_power_item": 1.6, "sim_power_user": 1.1,
        "item_weight": 0.95,
    }

    print("\n" + "=" * 70)
    print("ABLATION STUDY")
    print("=" * 70)

    # 1. BASELINE
    print("\n1. BASELINE (current assignment1.py settings)")
    train_fn, pred_fn = build_model(**base_params)
    train_fn(train)
    baseline_mae = evaluate_mae(pred_fn, valid)
    print(f"   MAE: {baseline_mae:.5f}")

    results = {"Baseline": baseline_mae}

    # 2. Optimal rounding only
    for mode_name, mode_val in [("nearest_int", "int"), ("nearest_half", "half")]:
        print(f"\n2a. + Rounding ({mode_name})")
        train_fn, pred_fn = build_model(**base_params, rounding_mode=mode_val)
        train_fn(train)
        mae = evaluate_mae(pred_fn, valid)
        delta = baseline_mae - mae
        print(f"   MAE: {mae:.5f} (Δ: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
        results[f"Rounding_{mode_name}"] = mae

    # 3. Negative similarity only
    print(f"\n3. + Negative similarity")
    train_fn, pred_fn = build_model(**base_params, use_negative_sim=True)
    train_fn(train)
    mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - mae
    print(f"   MAE: {mae:.5f} (Δ: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
    results["Negative_sim"] = mae

    # 4. Weighted average smoothing (sweep lambda)
    print(f"\n4. + Weighted average smoothing (sweep lambda)")
    best_lambda, best_lambda_mae = 0.0, baseline_mae
    for lam in [0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 10.0]:
        train_fn, pred_fn = build_model(**base_params, baseline_lambda=lam)
        train_fn(train)
        mae = evaluate_mae(pred_fn, valid)
        delta = baseline_mae - mae
        marker = ""
        if mae < best_lambda_mae:
            best_lambda_mae = mae
            best_lambda = lam
            marker = " ***"
        print(f"   λ={lam:5.1f}: MAE={mae:.5f} (Δ: {delta:+.5f}){marker}")
    results["Smoothing"] = best_lambda_mae
    print(f"   Best λ={best_lambda}")

    # 5. Confidence-adaptive blending only
    print(f"\n5. + Confidence-adaptive blending")
    train_fn, pred_fn = build_model(**base_params, adaptive_blend=True)
    train_fn(train)
    mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - mae
    print(f"   MAE: {mae:.5f} (Δ: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
    results["Adaptive_blend"] = mae

    # 6. COMBINED: all improvements
    print(f"\n6. ALL IMPROVEMENTS COMBINED")

    # Find best rounding mode
    best_rounding = "none"
    best_rounding_mae = baseline_mae
    for mode_name, mode_val in [("none", "none"), ("int", "int"), ("half", "half")]:
        key = f"Rounding_{mode_name}" if mode_name != "none" else "Baseline"
        if key in results and results[key] < best_rounding_mae:
            best_rounding_mae = results[key]
            best_rounding = mode_val

    # Combined with best settings
    train_fn, pred_fn = build_model(
        **base_params,
        use_negative_sim=True,
        baseline_lambda=best_lambda,
        adaptive_blend=True,
        rounding_mode=best_rounding,
    )
    train_fn(train)
    combined_mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - combined_mae
    print(f"   MAE: {combined_mae:.5f} (Δ: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
    print(f"   Settings: neg_sim=True, λ={best_lambda}, adaptive=True, rounding={best_rounding}")
    results["Combined"] = combined_mae

    # 7. Combined without rounding (in case rounding hurts combined)
    print(f"\n7. ALL IMPROVEMENTS COMBINED (no rounding)")
    train_fn, pred_fn = build_model(
        **base_params,
        use_negative_sim=True,
        baseline_lambda=best_lambda,
        adaptive_blend=True,
        rounding_mode="none",
    )
    train_fn(train)
    nornd_mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - nornd_mae
    print(f"   MAE: {nornd_mae:.5f} (Δ: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
    results["Combined_no_rounding"] = nornd_mae

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Configuration':<30} {'MAE':>10} {'Δ':>10} {'%':>8}")
    print("-" * 60)
    for name, mae in results.items():
        delta = baseline_mae - mae
        pct = 100 * delta / baseline_mae
        print(f"{name:<30} {mae:>10.5f} {delta:>+10.5f} {pct:>+7.2f}%")
    print("=" * 70)

    best_name = min(results, key=results.get)
    print(f"\nBest: {best_name} → MAE={results[best_name]:.5f}")
    print(f"Estimated server MAE: {0.70883 * results[best_name] / baseline_mae:.5f}")


if __name__ == "__main__":
    main()
