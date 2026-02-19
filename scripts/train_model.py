"""
Train and evaluate models for chronic absenteeism prediction.

Trains a prior-year attendance baseline, logistic regression, random forest,
and XGBoost. Tunes XGBoost via RandomizedSearchCV. Saves the best model,
a comparison table, and a feature importance chart.

Usage:
    python scripts/train_model.py
"""

import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier

# Allow importing sibling modules
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_test_split import load_and_prepare, get_cv_splits


MODELS_DIR = Path(__file__).resolve().parent.parent / 'models'


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def evaluate(name, y_true, y_pred, y_prob):
    """Return a dict of evaluation metrics."""
    return {
        'model': name,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_prob),
    }


def print_confusion(name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(f'\n  {name} confusion matrix:')
    print(f'    TN={cm[0,0]:,}  FP={cm[0,1]:,}')
    print(f'    FN={cm[1,0]:,}  TP={cm[1,1]:,}')


# ---------------------------------------------------------------------------
# 1. Baseline — prior-year attendance
# ---------------------------------------------------------------------------

def baseline_model(X_train, y_train, X_test, y_test, feature_names):
    """Predict chronic if prev_att_rate < 0.90; fall back to train chronic rate."""
    print('\n=== Baseline: prior-year attendance ===')

    train_chronic_rate = y_train.mean()
    prev_idx = feature_names.index('prev_att_rate')
    has_prior_idx = feature_names.index('has_prior_data')

    prev_att = X_test.iloc[:, prev_idx].values
    has_prior = X_test.iloc[:, has_prior_idx].values

    # Students with prior data: chronic if att_rate < 0.90
    # Students without: predict 1 if train chronic rate >= 0.5, else 0
    fallback_pred = int(train_chronic_rate >= 0.5)
    y_pred = np.where(
        has_prior == 1,
        (prev_att < 0.90).astype(int),
        fallback_pred,
    )
    # Probability proxy: 1 - prev_att_rate (clipped), or train chronic rate
    y_prob = np.where(
        has_prior == 1,
        np.clip(1 - prev_att, 0, 1),
        train_chronic_rate,
    )

    result = evaluate('Baseline (prior att)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  F1={result['f1']:.4f}  AUC={result['auc_roc']:.4f}")
    return result


# ---------------------------------------------------------------------------
# 2. Candidate models
# ---------------------------------------------------------------------------

def train_logistic(X_train, y_train, X_test, y_test):
    print('\n=== Logistic Regression ===')
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = evaluate('Logistic Regression', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  F1={result['f1']:.4f}  AUC={result['auc_roc']:.4f}")
    return model, result


def train_random_forest(X_train, y_train, X_test, y_test):
    print('\n=== Random Forest ===')
    model = RandomForestClassifier(
        class_weight='balanced', n_estimators=300, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = evaluate('Random Forest', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  F1={result['f1']:.4f}  AUC={result['auc_roc']:.4f}")
    return model, result


def train_xgboost(X_train, y_train, X_test, y_test):
    print('\n=== XGBoost (default) ===')
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = neg / pos
    model = XGBClassifier(
        scale_pos_weight=ratio, eval_metric='logloss',
        n_estimators=300, random_state=42, n_jobs=-1,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = evaluate('XGBoost (default)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  F1={result['f1']:.4f}  AUC={result['auc_roc']:.4f}")
    return model, result


# ---------------------------------------------------------------------------
# 3. Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgboost(X_train, y_train, X_test, y_test, cv_splits):
    print('\n=== XGBoost hyperparameter tuning (RandomizedSearchCV) ===')
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = neg / pos

    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 800],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
    }

    base = XGBClassifier(
        scale_pos_weight=ratio, eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=30, scoring='f1', cv=cv_splits,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    print(f'  Best CV F1: {search.best_score_:.4f}')
    print(f'  Best params: {search.best_params_}')

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    result = evaluate('XGBoost (tuned)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  F1={result['f1']:.4f}  AUC={result['auc_roc']:.4f}")
    return best, result


# ---------------------------------------------------------------------------
# 5. Feature importance
# ---------------------------------------------------------------------------

def plot_feature_importance(model, feature_names, path):
    importances = model.feature_importances_
    idx = np.argsort(importances)[-20:]
    top_names = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(top_names, top_vals)
    ax.set_xlabel('Feature Importance')
    ax.set_title('Top 20 Features — XGBoost (tuned)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'\nFeature importance chart saved to {path}')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    print('=== Loading data ===\n')
    X_train, y_train, X_test, y_test, feature_names = load_and_prepare()
    cv_splits = get_cv_splits(X_train, y_train)

    results = []

    # 1. Baseline
    results.append(baseline_model(X_train, y_train, X_test, y_test, feature_names))

    # 2. Candidate models
    lr_model, lr_res = train_logistic(X_train, y_train, X_test, y_test)
    results.append(lr_res)

    rf_model, rf_res = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_res)

    xgb_model, xgb_res = train_xgboost(X_train, y_train, X_test, y_test)
    results.append(xgb_res)

    # 3. Tuned XGBoost
    xgb_tuned, xgb_tuned_res = tune_xgboost(
        X_train, y_train, X_test, y_test, cv_splits,
    )
    results.append(xgb_tuned_res)

    # 4. Comparison table
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values('f1', ascending=False).reset_index(drop=True)
    print('\n=== Model comparison ===\n')
    print(comparison.to_string(index=False, float_format='{:.4f}'.format))
    comparison.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
    print(f'\nSaved to {MODELS_DIR / "model_comparison.csv"}')

    # 5. Feature importance (best model = tuned XGBoost)
    plot_feature_importance(xgb_tuned, feature_names, MODELS_DIR / 'feature_importance.png')

    # 6. Save best model
    joblib.dump(xgb_tuned, MODELS_DIR / 'best_model.joblib')
    print(f'Best model saved to {MODELS_DIR / "best_model.joblib"}')


if __name__ == '__main__':
    main()
