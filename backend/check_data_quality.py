"""
check_data_quality.py
Run this after prepare_data.py to see if your dataset is worth training on.
Usage: python check_data_quality.py
"""

import pandas as pd
import numpy as np
from scipy import stats

def analyze_dataset(csv_file="training_data.csv"):
    df = pd.read_csv(csv_file)
    
    real = df[df['label'] == 0]
    fake = df[df['label'] == 1]
    
    print("=" * 65)
    print("  DATASET QUALITY REPORT")
    print("=" * 65)

    # ── 1. Basic counts ───────────────────────────────────────────────────
    print(f"\n📦 SAMPLE COUNTS")
    print(f"   Real images : {len(real)}")
    print(f"   Fake images : {len(fake)}")
    ratio = min(len(real), len(fake)) / max(len(real), len(fake))
    balance_status = "✅ Balanced" if ratio > 0.8 else "⚠️  Imbalanced — consider equalizing"
    print(f"   Balance     : {ratio:.2f}  →  {balance_status}")

    feature_cols = [c for c in df.columns if c != 'label']

    # ── 2. Per-feature discrimination check ──────────────────────────────
    print(f"\n📊 FEATURE DISCRIMINATION (the most important check)")
    print(f"   {'Feature':<22} {'Real mean':>10} {'Fake mean':>10} {'Overlap':>10}  {'Signal?':>9}  {'p-value':>9}")
    print("   " + "-" * 63)

    good_features = []
    bad_features = []

    for col in feature_cols:
        r_vals = real[col].values
        f_vals = fake[col].values

        r_mean = r_vals.mean()
        f_mean = f_vals.mean()

        # Overlap: what fraction of values are in the shared range?
        shared_min = max(r_vals.min(), f_vals.min())
        shared_max = min(r_vals.max(), f_vals.max())
        r_overlap = ((r_vals >= shared_min) & (r_vals <= shared_max)).mean()
        f_overlap = ((f_vals >= shared_min) & (f_vals <= shared_max)).mean()
        overlap = (r_overlap + f_overlap) / 2

        # Statistical test: can we separate the distributions?
        t_stat, p_val = stats.ttest_ind(r_vals, f_vals)

        # Cohen's d: effect size (how separated are the means relative to spread?)
        pooled_std = np.sqrt((r_vals.std()**2 + f_vals.std()**2) / 2) + 1e-9
        cohens_d = abs(r_mean - f_mean) / pooled_std

        if cohens_d > 0.5 and p_val < 0.05:
            signal = "✅ GOOD"
            good_features.append(col)
        elif cohens_d > 0.2:
            signal = "⚠️  WEAK"
        else:
            signal = "❌ USELESS"
            bad_features.append(col)

        print(f"   {col:<22} {r_mean:>10.1f} {f_mean:>10.1f} {overlap:>10.0%}  {signal:>9}  {p_val:>9.4f}")

    # ── 3. Saturation check ───────────────────────────────────────────────
    print(f"\n🔴 SATURATION CHECK (features stuck at min/max)")
    any_saturated = False
    for col in feature_cols:
        vals = df[col].values
        pct_max = (vals == vals.max()).mean() * 100
        pct_min = (vals == vals.min()).mean() * 100
        n_unique = len(np.unique(vals))

        if pct_max > 50 or pct_min > 50 or n_unique <= 5:
            any_saturated = True
            print(f"   ⚠️  {col}: {n_unique} unique values | "
                  f"{pct_max:.0f}% at max | {pct_min:.0f}% at min")

    if not any_saturated:
        print("   ✅ No saturated features found.")

    # ── 4. Overall verdict ────────────────────────────────────────────────
    print(f"\n{'=' * 65}")
    print(f"  VERDICT")
    print(f"{'=' * 65}")
    print(f"  Good features  : {len(good_features)} → {good_features}")
    print(f"  Useless features: {len(bad_features)} → {bad_features}")

    if len(good_features) >= 5:
        print("\n  ✅ Dataset looks TRAINABLE. Proceed with train_model.py.")
    elif len(good_features) >= 3:
        print("\n  ⚠️  Dataset is MARGINAL. Expect 65-75% accuracy.")
        print("     Try adding more samples or improving the weak features.")
    else:
        print("\n  ❌ Dataset is NOT READY. Features are not separating classes.")
        print("     Fix the feature extraction in detector.py before training.")

    # ── 5. Sample size recommendation ─────────────────────────────────────
    print(f"\n📏 SAMPLE SIZE GUIDANCE")
    n = min(len(real), len(fake))
    if n < 200:
        print(f"   ❌ {n} samples per class is too few. Aim for 500+ per class.")
    elif n < 500:
        print(f"   ⚠️  {n} samples per class is low. Results will be noisy.")
        print(f"      Aim for 500–1000 per class for reliable accuracy.")
    elif n < 1000:
        print(f"   ✅ {n} samples per class is acceptable for this approach.")
    else:
        print(f"   ✅ {n} samples per class — good dataset size.")

    print()


if __name__ == "__main__":
    analyze_dataset()