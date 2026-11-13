"""Telecom customer churn prediction — training & evaluation.

Dataset: IBM Telco Customer Churn (WA_Fn-UseC_-Telco-Customer-Churn.csv)
7,043 rows, 21 columns. Churn: Yes 1,869 (26.5%), No 5,174 (73.5%).
This is an IBM demo / educational dataset.

Run:  python src/train.py
Outputs: metrics.json, plots/*.svg
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUT = ROOT / "plots"
OUT.mkdir(exist_ok=True)

RANDOM_STATE = 42


def load_and_clean():
    df = pd.read_csv(DATA)
    assert df.shape == (7043, 21), f"unexpected shape {df.shape}"
    # TotalCharges is stored as text; 11 rows are blank (tenure == 0).
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    n_missing = int(df["TotalCharges"].isna().sum())
    df = df.dropna(subset=["TotalCharges"]).reset_index(drop=True)
    df = df.drop(columns=["customerID"])
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])
    return X, y, n_missing


def main():
    X, y, n_missing = load_and_clean()
    print(f"rows after cleaning: {len(X)} (dropped {n_missing} blank TotalCharges)")
    print(f"churn rate: {y.mean():.4f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"train={len(X_train)} test={len(X_test)} "
          f"train churn={y_train.mean():.3f} test churn={y_test.mean():.3f}")

    num_cols = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ])

    # Imbalance: ~26.5% churn. Logistic uses class_weight='balanced';
    # XGBoost uses scale_pos_weight = neg/pos on the training set.
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / pos
    print(f"scale_pos_weight (train): {spw:.3f}")

    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier

    models = {
        "Logistic Regression": Pipeline([
            ("pre", pre),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                       random_state=RANDOM_STATE)),
        ]),
        "XGBoost": Pipeline([
            ("pre", pre),
            ("clf", XGBClassifier(n_estimators=300, max_depth=4,
                                  learning_rate=0.05, subsample=0.9,
                                  colsample_bytree=0.9, scale_pos_weight=spw,
                                  eval_metric="logloss",
                                  random_state=RANDOM_STATE, n_jobs=2)),
        ]),
        "LightGBM": Pipeline([
            ("pre", pre),
            ("clf", LGBMClassifier(n_estimators=300, num_leaves=31,
                                   learning_rate=0.05, class_weight="balanced",
                                   random_state=RANDOM_STATE, n_jobs=2,
                                   verbose=-1)),
        ]),
    }

    results = {}
    roc_data = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        pred = (proba >= 0.5).astype(int)
        results[name] = {
            "roc_auc": round(float(roc_auc_score(y_test, proba)), 4),
            "accuracy": round(float(accuracy_score(y_test, pred)), 4),
            "precision_churn": round(float(precision_score(y_test, pred, zero_division=0)), 4),
            "recall_churn": round(float(recall_score(y_test, pred, zero_division=0)), 4),
            "f1_churn": round(float(f1_score(y_test, pred, zero_division=0)), 4),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc_data[name] = (fpr, tpr)
        print(name, results[name])

    # --- ROC plot (SVG: text, so it can be versioned cleanly) ---
    plt.figure(figsize=(7, 5))
    for name, (fpr, tpr) in roc_data.items():
        plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curves — Telecom Churn (test set)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT / "roc_curves.svg")
    plt.close()

    # --- Confusion matrix for the best model by ROC-AUC ---
    best_name = max(results, key=lambda k: results[k]["roc_auc"])
    cm = np.array(results[best_name]["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, cmap="Blues")
    plt.title(f"Confusion matrix — {best_name} (test set)")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.xticks([0, 1], ["No churn", "Churn"])
    plt.yticks([0, 1], ["No churn", "Churn"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14)
    plt.tight_layout()
    plt.savefig(OUT / "confusion_matrix.svg")
    plt.close()
    print(f"best model by ROC-AUC: {best_name}")

    # --- SHAP: top drivers (LightGBM, TreeExplainer, 300 test rows) ---
    import shap
    lgbm_pipe = models["LightGBM"]
    pre_fitted = lgbm_pipe.named_steps["pre"]
    clf = lgbm_pipe.named_steps["clf"]
    X_test_enc = pre_fitted.transform(X_test)
    feature_names = pre_fitted.get_feature_names_out()
    rng = np.random.RandomState(RANDOM_STATE)
    idx = rng.choice(len(X_test), size=min(300, len(X_test)), replace=False)
    explainer = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_test_enc[idx])
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    mean_abs = np.abs(shap_values).mean(axis=0)
    order = np.argsort(mean_abs)[::-1]
    top = [
        {"feature": str(feature_names[i]), "mean_abs_shap": round(float(mean_abs[i]), 4)}
        for i in order[:10]
    ]
    print("SHAP top 10:", top)

    plt.figure(figsize=(8, 5))
    names = [t["feature"].replace("cat__", "").replace("num__", "") for t in top[:10]][::-1]
    vals = [t["mean_abs_shap"] for t in top[:10]][::-1]
    plt.barh(names, vals)
    plt.xlabel("Mean |SHAP value|")
    plt.title("Top 10 churn drivers — LightGBM + SHAP (test sample)")
    plt.tight_layout()
    plt.savefig(OUT / "shap_top_drivers.svg")
    plt.close()

    payload = {
        "dataset": {"rows": 7043, "columns": 21, "churn_yes": 1869,
                    "churn_no": 5174, "rows_after_cleaning": len(X),
                    "blank_total_charges_dropped": n_missing},
        "split": {"test_size": 0.20, "random_state": RANDOM_STATE, "stratified": True},
        "best_model_by_roc_auc": best_name,
        "results": results,
        "shap_top_10_lightgbm": top,
        "shap_top_5": top[:5],
    }
    (ROOT / "metrics.json").write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
