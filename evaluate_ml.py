"""
Model 3 (Linear Regression) evaluation.

Compares the pipeline's Linear Regression model against a naive
baseline, and reports per-position breakdowns.

Run with:
    python evaluate_ml.py
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr

import prototype_fpl as fpl


# ==============================================================
# DATA PREPARATION
# ==============================================================
def load_dataset():
    """Fetch FPL data and prepare it for modelling."""
    players, teams = fpl.fetch_fpl_data_direct()

    df = pd.DataFrame(players)
    df["form"] = pd.to_numeric(df["form"], errors="coerce")
    df["minutes"] = pd.to_numeric(df["minutes"], errors="coerce")
    df["points_per_game"] = pd.to_numeric(df["points_per_game"], errors="coerce")
    df["total_points"] = pd.to_numeric(df["total_points"], errors="coerce")

    df = df.dropna(subset=["form", "minutes", "points_per_game"])
    df = df[df["minutes"] > 90].copy()

    df["team_name"] = df["team"].map(teams).fillna("Unknown")
    df["name"] = df["web_name"].fillna(df["second_name"]).astype(str)

    return df


# ==============================================================
# METRICS
# ==============================================================
def compute_metrics(y_true, y_pred, name="Model"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r = np.corrcoef(y_true, y_pred)[0, 1]
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "name": name,
        "MAE": mae,
        "RMSE": rmse,
        "Pearson_r": r,
        "Spearman_rho": rho,
        "n": len(y_true),
    }


def print_metrics(m):
    print(f"\n  {m['name']} (n={m['n']})")
    print(f"    MAE           = {m['MAE']:.3f}")
    print(f"    RMSE          = {m['RMSE']:.3f}")
    print(f"    Pearson r     = {m['Pearson_r']:.3f}")
    print(f"    Spearman rho  = {m['Spearman_rho']:.3f}")


# ==============================================================
# MAIN EVALUATION
# ==============================================================
def evaluate():
    print("=" * 60)
    print("📊 MODEL 3 — LINEAR REGRESSION EVALUATION")
    print("=" * 60)

    df = load_dataset()
    print(f"\nDataset: {len(df)} players (with > 90 minutes played)")

    X = df[["form", "minutes"]]
    y = df["points_per_game"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")

    # --- Model 1: Linear Regression ---
    lr = LinearRegression().fit(X_train, y_train)
    pred_lr = lr.predict(X_test)

    # --- Model 2: Naive baseline (predict training mean) ---
    pred_baseline = np.full_like(y_test, y_train.mean(), dtype=float)

    # --- Model 3: Form-only baseline (predict just from form) ---
    form_only = LinearRegression().fit(X_train[["form"]], y_train)
    pred_form = form_only.predict(X_test[["form"]])

    print("-" * 60)
    print("Regression metrics (test set):")
    print("-" * 60)
    metrics = [
        compute_metrics(y_test, pred_lr, "Linear Regression (form + minutes)"),
        compute_metrics(y_test, pred_form, "Form-only Linear Regression"),
        compute_metrics(y_test, pred_baseline, "Naive baseline (mean)"),
    ]
    for m in metrics:
        print_metrics(m)

    # --- Top-20 hit rate: how often do predicted top players beat the median? ---
    print("\n" + "-" * 60)
    print("Top-20 ranking evaluation:")
    print("-" * 60)
    test_df = X_test.copy()
    test_df["actual"] = y_test
    test_df["predicted"] = pred_lr
    median_actual = y_test.median()

    top20 = test_df.nlargest(20, "predicted")
    hit_rate = (top20["actual"] > median_actual).mean()
    print(f"  Median actual points/GW on test set: {median_actual:.2f}")
    print(f"  Top-20 hit rate (predicted top players above median): "
          f"{hit_rate:.1%}")

    # --- Per-position breakdown ---
    print("\n" + "-" * 60)
    print("Per-position breakdown:")
    print("-" * 60)
    # FPL position codes: 1=GK, 2=DEF, 3=MID, 4=FWD
    pos_labels = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
    test_pos = df.loc[X_test.index, "element_type"] if "element_type" in df.columns \
               else None

    if test_pos is not None:
        for pos_code, pos_name in pos_labels.items():
            mask = test_pos == pos_code
            if mask.sum() < 3:
                continue
            m = compute_metrics(
                y_test[mask], pred_lr[mask.values],
                f"{pos_name} (n={mask.sum()})"
            )
            print_metrics(m)
    else:
        print("  (element_type column not present — skipping)")

    # --- Baseline comparison summary ---
    print("\n" + "=" * 60)
    print("📋 SUMMARY")
    print("=" * 60)
    lr_m = metrics[0]
    bl_m = metrics[2]
    mae_improvement = (bl_m["MAE"] - lr_m["MAE"]) / bl_m["MAE"] * 100
    rmse_improvement = (bl_m["RMSE"] - lr_m["RMSE"]) / bl_m["RMSE"] * 100
    print(f"  LR MAE improvement over baseline:  {mae_improvement:+.1f}%")
    print(f"  LR RMSE improvement over baseline: {rmse_improvement:+.1f}%")
    print(f"  LR Spearman rank correlation:      {lr_m['Spearman_rho']:.3f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate()