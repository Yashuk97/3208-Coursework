"""
Ablation study for 4 targeted improvements to reduce server-side MAE.

Tests each improvement independently and combined:
1. Remove integer rounding → continuous predictions
2. Tune ITEM_WEIGHT (grid sweep)
3. Confidence-weighted baseline smoothing
4. Use negative similarities
"""

import csv
import math
import time


TRAIN_FILE = "train_100k_withratings.csv"


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


def build_model(
    top_k_item=20, top_k_user=20,
    bias_reg=5.0,
    shrinkage_item=110.0, shrinkage_user=25.0,
    min_common_users=3, min_common_items=3,
    sim_power_item=1.6, sim_power_user=1.1,
    item_weight=0.95,
    # Improvement flags
    use_rounding=True,        # True = integer rounding (baseline), False = continuous
    use_negative_sim=False,
    baseline_lambda=0.0,      # 0 = no smoothing
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
        if x != x:  # NaN check
            ga = state["global_avg"]
            return round(ga) if use_rounding else ga
        if use_rounding:
            return max(1.0, min(5.0, round(max(0.5, min(5.0, x)))))
        else:
            return max(1.0, min(5.0, x))

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
            sim = math.pow(sim, sim_power_item) if sim > 0.0 else 0.0

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
            sim = math.pow(sim, sim_power_user) if sim > 0.0 else 0.0

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
                    neighbours.append((sim, rating - get_baseline(user, other_item)))
            else:
                if sim > 0.0:
                    neighbours.append((sim, rating - get_baseline(user, other_item)))

        if not neighbours:
            return base_ui, 0.0

        neighbours.sort(key=lambda x: abs(x[0]), reverse=True)
        neighbours = neighbours[:top_k_item]

        weighted_sum = sum(s * d for s, d in neighbours)
        sim_sum = sum(abs(s) for s, d in neighbours)

        if sim_sum == 0.0:
            return base_ui, 0.0

        raw_pred = base_ui + (weighted_sum / sim_sum)

        # Confidence-weighted baseline smoothing
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
                    neighbours.append((sim, rating - get_baseline(other_user, item)))
            else:
                if sim > 0.0:
                    neighbours.append((sim, rating - get_baseline(other_user, item)))

        if not neighbours:
            return base_ui, 0.0

        neighbours.sort(key=lambda x: abs(x[0]), reverse=True)
        neighbours = neighbours[:top_k_user]

        weighted_sum = sum(s * d for s, d in neighbours)
        sim_sum = sum(abs(s) for s, d in neighbours)

        if sim_sum == 0.0:
            return base_ui, 0.0

        raw_pred = base_ui + (weighted_sum / sim_sum)

        if baseline_lambda > 0.0:
            raw_pred = (sim_sum * raw_pred + baseline_lambda * base_ui) / (sim_sum + baseline_lambda)

        return raw_pred, sim_sum

    def predict(user, item):
        if user not in state["user_ratings"] and item not in state["item_ratings"]:
            return clamp(state["global_avg"])

        base = get_baseline(user, item)

        if user not in state["user_ratings"]:
            return clamp(base)

        if item not in state["item_ratings"]:
            return clamp(base)

        item_pred, item_str = predict_item_component(user, item)
        user_pred, user_str = predict_user_component(user, item)

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

        state["global_avg"] = total_sum / len(ratings) if ratings else 3.5

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


def main():
    print("Loading data...")
    all_ratings = read_training_data(TRAIN_FILE)
    train, valid = create_validation_split(all_ratings)
    print(f"Train: {len(train)}, Valid: {len(valid)}")

    # Baseline params (current assignment1.py)
    base = {
        "top_k_item": 20, "top_k_user": 20,
        "bias_reg": 5.0,
        "shrinkage_item": 110.0, "shrinkage_user": 25.0,
        "min_common_users": 3, "min_common_items": 3,
        "sim_power_item": 1.6, "sim_power_user": 1.1,
        "item_weight": 0.95,
    }

    print("\n" + "=" * 70)
    print("ABLATION STUDY: 4 Improvements")
    print("=" * 70)

    # 0. BASELINE (integer rounding ON)
    print("\n0. BASELINE (integer rounding, ITEM_WEIGHT=0.95, no smoothing, no neg sim)")
    t0 = time.time()
    train_fn, pred_fn = build_model(**base, use_rounding=True)
    train_fn(train)
    baseline_mae = evaluate_mae(pred_fn, valid)
    print(f"   MAE: {baseline_mae:.5f} ({time.time()-t0:.1f}s)")

    results = {"0_Baseline": baseline_mae}

    # 1. Remove integer rounding only
    print("\n1. REMOVE INTEGER ROUNDING (continuous output)")
    t0 = time.time()
    train_fn, pred_fn = build_model(**base, use_rounding=False)
    train_fn(train)
    mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - mae
    print(f"   MAE: {mae:.5f} (Δ: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%) ({time.time()-t0:.1f}s)")
    results["1_No_rounding"] = mae

    # 2. Tune ITEM_WEIGHT (with rounding removed)
    print("\n2. TUNE ITEM_WEIGHT (with no rounding)")
    best_iw, best_iw_mae = 0.95, mae
    for iw in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
        train_fn, pred_fn = build_model(**{**base, "item_weight": iw}, use_rounding=False)
        train_fn(train)
        m = evaluate_mae(pred_fn, valid)
        marker = " ***" if m < best_iw_mae else ""
        print(f"   IW={iw:.2f}: MAE={m:.5f}{marker}")
        if m < best_iw_mae:
            best_iw_mae = m
            best_iw = iw
    delta = baseline_mae - best_iw_mae
    print(f"   Best ITEM_WEIGHT={best_iw:.2f}, MAE={best_iw_mae:.5f} (Δ from baseline: {delta:+.5f})")
    results["2_Best_IW"] = best_iw_mae

    # 3. Confidence-weighted baseline smoothing (with no rounding + best IW)
    print(f"\n3. BASELINE SMOOTHING (with no rounding + IW={best_iw})")
    best_lam, best_lam_mae = 0.0, best_iw_mae
    for lam in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0, 10.0]:
        train_fn, pred_fn = build_model(
            **{**base, "item_weight": best_iw},
            use_rounding=False, baseline_lambda=lam)
        train_fn(train)
        m = evaluate_mae(pred_fn, valid)
        marker = " ***" if m < best_lam_mae else ""
        print(f"   λ={lam:5.1f}: MAE={m:.5f}{marker}")
        if m < best_lam_mae:
            best_lam_mae = m
            best_lam = lam
    delta = baseline_mae - best_lam_mae
    print(f"   Best λ={best_lam:.1f}, MAE={best_lam_mae:.5f} (Δ from baseline: {delta:+.5f})")
    results["3_Best_smoothing"] = best_lam_mae

    # 4. Negative similarities (with no rounding + best IW + best smoothing)
    print(f"\n4. NEGATIVE SIMILARITIES (with no rounding + IW={best_iw} + λ={best_lam})")
    train_fn, pred_fn = build_model(
        **{**base, "item_weight": best_iw},
        use_rounding=False, baseline_lambda=best_lam, use_negative_sim=True)
    train_fn(train)
    neg_mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - neg_mae
    print(f"   MAE: {neg_mae:.5f} (Δ from baseline: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
    results["4_Neg_sim"] = neg_mae

    # Also test without negative sim for the combined (in case it hurts)
    print(f"\n5. COMBINED WITHOUT NEG SIM (no rounding + IW={best_iw} + λ={best_lam})")
    train_fn, pred_fn = build_model(
        **{**base, "item_weight": best_iw},
        use_rounding=False, baseline_lambda=best_lam, use_negative_sim=False)
    train_fn(train)
    comb_mae = evaluate_mae(pred_fn, valid)
    delta = baseline_mae - comb_mae
    print(f"   MAE: {comb_mae:.5f} (Δ from baseline: {delta:+.5f}, {100*delta/baseline_mae:+.2f}%)")
    results["5_Combined_no_neg"] = comb_mae

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"{'Config':<30} {'MAE':>10} {'Δ':>10} {'%':>8}")
    print("-" * 60)
    for name, mae in results.items():
        delta = baseline_mae - mae
        pct = 100 * delta / baseline_mae
        print(f"{name:<30} {mae:>10.5f} {delta:>+10.5f} {pct:>+7.2f}%")
    print("=" * 70)

    best_name = min(results, key=results.get)
    best_val = results[best_name]
    print(f"\nBest: {best_name} → MAE={best_val:.5f}")
    print(f"Projected server MAE: {0.6936 * best_val / baseline_mae:.5f}")
    print(f"\nOptimal settings for assignment1.py:")
    print(f"  use_rounding = False")
    print(f"  ITEM_WEIGHT = {best_iw}")
    print(f"  baseline_lambda = {best_lam}")
    use_neg = results.get("4_Neg_sim", 999) < results.get("5_Combined_no_neg", 999)
    print(f"  use_negative_sim = {use_neg}")


if __name__ == "__main__":
    main()
