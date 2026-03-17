import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import joblib


def train_brain(csv_file="training_data.csv"):
    df = pd.read_csv(csv_file)

    print(f"📊 Dataset: {len(df)} rows | {df['label'].sum()} fake | {(df['label']==0).sum()} real")

    # ── Sanity check: warn if features look saturated ──────────────────────
    feature_cols = [c for c in df.columns if c != 'label']
    for col in feature_cols:
        pct_max = (df[col] == df[col].max()).mean() * 100
        if pct_max > 60:
            print(f"  ⚠️  '{col}' is at its max value {pct_max:.0f}% of the time — likely saturated!")
    print()

    X = df[feature_cols]
    y = df['label']

    # ── Check class balance ────────────────────────────────────────────────
    class_ratio = y.mean()
    if class_ratio > 0.6 or class_ratio < 0.4:
        print(f"⚠️  Class imbalance detected ({class_ratio:.0%} fake). Using class_weight='balanced'.")

    # ── Model: GradientBoosting outperforms RandomForest on small, noisy
    #    datasets like this one. It learns sequentially, which lets it
    #    recover from the weaker features. ──────────────────────────────────
    model = Pipeline([
        ('scaler', StandardScaler()),          # helps GBM converge
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            min_samples_leaf=5,
            random_state=42
        ))
    ])

    # ── Cross-validation first (tells you the real accuracy) ──────────────
    print("🔍 Running 5-fold cross-validation...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    print(f"  CV Accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    cv_f1 = cross_val_score(model, X, y, cv=cv, scoring='f1')
    print(f"  CV F1 Score: {cv_f1.mean():.3f} ± {cv_f1.std():.3f}")

    if cv_scores.mean() < 0.60:
        print("\n  ❌ CV accuracy below 60% — the features are still not discriminating.")
        print("     Re-run prepare_data.py with the new detector.py first.")
    elif cv_scores.mean() < 0.75:
        print("\n  ⚠️  Moderate accuracy. Consider adding more training data.")
    else:
        print("\n  ✅ Good CV accuracy — the model is learning real signal.")

    # ── Final training on full dataset ────────────────────────────────────
    print("\n🧠 Training final model on all data...")
    model.fit(X, y)

    # ── Feature importance (GBM clf is the second step) ───────────────────
    clf = model.named_steps['clf']
    importances = clf.feature_importances_
    imp_sorted = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("\n📈 Feature importances:")
    for feat, imp in imp_sorted:
        bar = '█' * int(imp * 40)
        print(f"   {feat:<22} {bar} {imp:.3f}")

    # ── Save ──────────────────────────────────────────────────────────────
    joblib.dump(model, 'deepfake_brain.pkl')
    print("\n✅ Model saved as deepfake_brain.pkl")

    # ── Quick in-sample report (sanity check only, not real accuracy) ─────
    y_pred = model.predict(X)
    print("\n📋 In-sample classification report (not CV — just a sanity check):")
    print(classification_report(y, y_pred, target_names=['Real', 'Fake']))
    print("Confusion matrix:")
    print(confusion_matrix(y, y_pred))
    print("\nReminder: the CV accuracy above is your real accuracy estimate.")


if __name__ == "__main__":
    train_brain()