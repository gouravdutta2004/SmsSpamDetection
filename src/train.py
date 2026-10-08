import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import csv
import json
import os

import joblib
import pandas as pd

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold, cross_validate
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from src.preprocessing import clean_text


RANDOM_STATE = 42
N_SPLITS = 5


def build_pipelines():
    """Create independent TF-IDF + classifier pipelines."""
    def tfidf():
        return TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=5000,
        )

    return {
        "Multinomial Naive Bayes": Pipeline([
            ("tfidf", tfidf()),
            ("clf", MultinomialNB()),
        ]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", tfidf()),
            ("clf", CalibratedClassifierCV(
                LinearSVC(random_state=RANDOM_STATE, dual="auto")
            )),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", tfidf()),
            ("clf", RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def main():
    print("Starting SMS Spam Detection Model Training Pipeline...")

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, "data", "SMSSpamCollection")
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(
        data_path,
        sep="\t",
        names=["label", "message"],
        quoting=csv.QUOTE_NONE,
    )
    print(f"Total samples: {len(df)}")

    df["clean_message"] = df["message"].apply(clean_text)
    df["target"] = df["label"].map({"ham": 0, "spam": 1})

    # Use the original message as the grouping key.
    # This prevents exact duplicate SMS messages from crossing
    # either the final train/test boundary or CV folds.
    groups = df["message"]

    # Create a deterministic, approximately 80/20, group-aware holdout.
    # One fold of a 5-fold StratifiedGroupKFold is used as the final test set.
    holdout_cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )
    train_idx, test_idx = next(
        holdout_cv.split(
            df["clean_message"],
            df["target"],
            groups=groups,
        )
    )

    X_train = df.iloc[train_idx]["clean_message"]
    X_test = df.iloc[test_idx]["clean_message"]
    y_train = df.iloc[train_idx]["target"]
    y_test = df.iloc[test_idx]["target"]
    groups_train = groups.iloc[train_idx]

    # Safety check: no exact message group may occur in both splits.
    train_groups = set(groups.iloc[train_idx])
    test_groups = set(groups.iloc[test_idx])
    overlap = train_groups.intersection(test_groups)
    if overlap:
        raise RuntimeError(
            f"Group leakage detected: {len(overlap)} message groups overlap "
            "between train and test."
        )

    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print("Verified: no exact message groups overlap between train and test.")

    candidate_models = build_pipelines()

    print("\nBenchmarking candidate models with 5-fold grouped CV:")
    cv = StratifiedGroupKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    results = {}
    best_model_name = None
    best_cv_f1 = -1.0

    for name, pipeline in candidate_models.items():
        cv_res = cross_validate(
            pipeline,
            X_train,
            y_train,
            groups=groups_train,
            cv=cv,
            scoring={
                "accuracy": "accuracy",
                "precision": "precision",
                "recall": "recall",
                "f1": "f1",
            },
            n_jobs=-1,
        )

        mean_cv_f1 = cv_res["test_f1"].mean()
        mean_cv_acc = cv_res["test_accuracy"].mean()
        mean_cv_prec = cv_res["test_precision"].mean()
        mean_cv_rec = cv_res["test_recall"].mean()

        results[name] = {
            "cv_accuracy": round(float(mean_cv_acc), 4),
            "cv_precision": round(float(mean_cv_prec), 4),
            "cv_recall": round(float(mean_cv_rec), 4),
            "cv_f1_score": round(float(mean_cv_f1), 4),
            "cv_f1_std": round(float(cv_res["test_f1"].std()), 4),
        }

        print(f"{name:<25} | Mean CV F1: {mean_cv_f1:.4f}")

        if mean_cv_f1 > best_cv_f1:
            best_cv_f1 = mean_cv_f1
            best_model_name = name

    print(
        f"\nBest model selected: {best_model_name} "
        f"(grouped CV F1: {best_cv_f1:.4f})"
    )

    # Rebuild a fresh pipeline and train it on the complete training partition.
    best_pipeline = build_pipelines()[best_model_name]
    best_pipeline.fit(X_train, y_train)

    print("\nEvaluating selected pipeline on grouped hold-out test set:")
    y_pred = best_pipeline.predict(X_test)
    y_proba = best_pipeline.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, pos_label=1))
    rec = float(recall_score(y_test, y_pred, pos_label=1))
    f1 = float(f1_score(y_test, y_pred, pos_label=1))
    auc = float(roc_auc_score(y_test, y_proba))

    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC AUC:   {auc:.4f}")

    pipeline_save_path = os.path.join(models_dir, "pipeline.joblib")
    metrics_save_path = os.path.join(models_dir, "metrics.json")

    joblib.dump(best_pipeline, pipeline_save_path)

    metrics_payload = {
        "best_model": best_model_name,
        "evaluation": {
            "holdout_method": "StratifiedGroupKFold single fold",
            "cv_method": "5-fold StratifiedGroupKFold",
            "group_key": "raw message",
            "random_state": RANDOM_STATE,
            "train_samples": int(len(X_train)),
            "test_samples": int(len(X_test)),
            "test_group_overlap": 0,
        },
        "models": results,
        "test_metrics": {
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
        },
        "total_samples": int(len(df)),
        "spam_count": int((df["label"] == "spam").sum()),
        "ham_count": int((df["label"] == "ham").sum()),
    }

    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    print(f"\nSaved pipeline to: {pipeline_save_path}")
    print(f"Saved metrics to: {metrics_save_path}")


if __name__ == "__main__":
    main()
