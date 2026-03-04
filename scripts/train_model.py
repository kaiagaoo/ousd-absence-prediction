"""
Train and evaluate models for chronic absenteeism prediction.

Trains a prior-year attendance baseline, logistic regression, decision tree,
random forest, and XGBoost. Tunes XGBoost via RandomizedSearchCV.
All models are optimized for recall. Saves the best model,
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, average_precision_score, classification_report,
    confusion_matrix, f1_score, precision_recall_curve, precision_score,
    recall_score, roc_auc_score,
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
        'pr_auc': average_precision_score(y_true, y_prob),
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
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
    return result


# ---------------------------------------------------------------------------
# 2. Candidate models
# ---------------------------------------------------------------------------

def train_logistic(X_train, y_train, X_test, y_test):
    print('\n=== Logistic Regression (L2) ===')
    model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = evaluate('Logistic Regression (L2)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
    return model, result


def train_logistic_elasticnet(X_train, y_train, X_test, y_test):
    print('\n=== Logistic Regression (ElasticNet) ===')
    model = LogisticRegression(
        penalty='elasticnet', solver='saga', l1_ratio=0.5,
        class_weight='balanced', max_iter=2000, random_state=42,
    )
    model.fit(X_train, y_train)

    # Report how many features were zeroed out by L1
    n_zero = (model.coef_[0] == 0).sum()
    n_total = len(model.coef_[0])
    print(f'  Features zeroed by L1: {n_zero}/{n_total}')

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = evaluate('Logistic (ElasticNet)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
    return model, result


def train_decision_tree(X_train, y_train, X_test, y_test):
    print('\n=== Decision Tree ===')
    model = DecisionTreeClassifier(
        class_weight='balanced', random_state=42,
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    result = evaluate('Decision Tree', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
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
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
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
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
    return model, result


# ---------------------------------------------------------------------------
# 3. Hyperparameter tuning
# ---------------------------------------------------------------------------

def tune_xgboost(X_train, y_train, X_test, y_test, cv_splits):
    print('\n=== XGBoost hyperparameter tuning (optimizing recall, with ElasticNet) ===')
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = neg / pos

    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 800],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0],
    }

    base = XGBClassifier(
        scale_pos_weight=ratio, eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=40, scoring='recall', cv=cv_splits,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    print(f'  Best CV Recall: {search.best_score_:.4f}')
    print(f'  Best params: {search.best_params_}')

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    result = evaluate('XGBoost (tuned-recall)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
    return best, result


def tune_xgboost_f1(X_train, y_train, X_test, y_test, cv_splits):
    """Tune XGBoost optimizing for F1 score with ElasticNet regularization."""
    print('\n=== XGBoost hyperparameter tuning (optimizing F1, with ElasticNet) ===')
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    ratio = neg / pos

    param_dist = {
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.05, 0.1],
        'n_estimators': [200, 500, 800],
        'subsample': [0.7, 0.8, 0.9],
        'colsample_bytree': [0.7, 0.8, 0.9],
        'min_child_weight': [1, 3, 5],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [0.1, 1.0, 5.0, 10.0],
    }

    base = XGBClassifier(
        scale_pos_weight=ratio, eval_metric='logloss',
        random_state=42, n_jobs=-1,
    )

    search = RandomizedSearchCV(
        base, param_dist,
        n_iter=40, scoring='f1', cv=cv_splits,
        random_state=42, n_jobs=-1, verbose=1,
    )
    search.fit(X_train, y_train)

    print(f'  Best CV F1: {search.best_score_:.4f}')
    print(f'  Best params: {search.best_params_}')

    best = search.best_estimator_
    y_pred = best.predict(X_test)
    y_prob = best.predict_proba(X_test)[:, 1]
    result = evaluate('XGBoost (tuned-F1)', y_test, y_pred, y_prob)
    print_confusion(result['model'], y_test, y_pred)
    print(f"  F1={result['f1']:.4f}  Recall={result['recall']:.4f}  AUC={result['auc_roc']:.4f}")
    return best, result


# ---------------------------------------------------------------------------
# 4. Threshold tuning
# ---------------------------------------------------------------------------

def tune_threshold(model, X_test, y_test):
    """Find optimal thresholds for different recall-precision tradeoffs."""
    print('\n=== Threshold tuning (XGBoost tuned) ===')
    y_prob = model.predict_proba(X_test)[:, 1]
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_test, y_prob)

    # Evaluate at specific threshold values
    target_thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50,
                         0.55, 0.60, 0.65, 0.70]
    rows = []
    for t in target_thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        rows.append({
            'threshold': t,
            'precision': precision_score(y_test, y_pred_t, zero_division=0),
            'recall': recall_score(y_test, y_pred_t, zero_division=0),
            'f1': f1_score(y_test, y_pred_t, zero_division=0),
            'flagged': y_pred_t.sum(),
            'flagged_pct': y_pred_t.mean() * 100,
        })

    threshold_df = pd.DataFrame(rows)
    print('\n  Threshold sweep:')
    print(threshold_df.to_string(index=False, float_format='{:.4f}'.format))

    # Find best F1 threshold
    best_f1_idx = threshold_df['f1'].idxmax()
    best_row = threshold_df.iloc[best_f1_idx]
    print(f'\n  Best F1 threshold: {best_row["threshold"]:.2f} '
          f'(P={best_row["precision"]:.3f}, R={best_row["recall"]:.3f}, '
          f'F1={best_row["f1"]:.3f}, flagged={int(best_row["flagged"]):,})')

    return threshold_df, precision_arr, recall_arr, thresholds


def plot_precision_recall_tradeoff(precision_arr, recall_arr, threshold_df, path):
    """Plot precision-recall curve and threshold tradeoff chart."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Precision-Recall curve
    axes[0].plot(recall_arr, precision_arr, linewidth=2)
    axes[0].set_xlabel('Recall')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision-Recall Curve — XGBoost (tuned)')
    axes[0].set_xlim([0, 1.02])
    axes[0].set_ylim([0, 1.02])
    axes[0].grid(True, alpha=0.3)

    # Right: Precision, Recall, F1 vs threshold
    axes[1].plot(threshold_df['threshold'], threshold_df['precision'],
                 'b-o', label='Precision', markersize=4)
    axes[1].plot(threshold_df['threshold'], threshold_df['recall'],
                 'r-o', label='Recall', markersize=4)
    axes[1].plot(threshold_df['threshold'], threshold_df['f1'],
                 'g-o', label='F1', markersize=4)
    best_f1_idx = threshold_df['f1'].idxmax()
    best_t = threshold_df.iloc[best_f1_idx]['threshold']
    axes[1].axvline(x=best_t, color='gray', linestyle='--', alpha=0.7,
                    label=f'Best F1 (t={best_t:.2f})')
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Score')
    axes[1].set_title('Metrics vs. Classification Threshold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f'Precision-recall tradeoff chart saved to {path}')


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

    lr_en_model, lr_en_res = train_logistic_elasticnet(X_train, y_train, X_test, y_test)
    results.append(lr_en_res)

    dt_model, dt_res = train_decision_tree(X_train, y_train, X_test, y_test)
    results.append(dt_res)

    rf_model, rf_res = train_random_forest(X_train, y_train, X_test, y_test)
    results.append(rf_res)

    xgb_model, xgb_res = train_xgboost(X_train, y_train, X_test, y_test)
    results.append(xgb_res)

    # 3. Tuned XGBoost (recall)
    xgb_tuned, xgb_tuned_res = tune_xgboost(
        X_train, y_train, X_test, y_test, cv_splits,
    )
    results.append(xgb_tuned_res)

    # 4. Tuned XGBoost (F1)
    xgb_f1, xgb_f1_res = tune_xgboost_f1(
        X_train, y_train, X_test, y_test, cv_splits,
    )
    results.append(xgb_f1_res)

    # 5. Comparison table
    comparison = pd.DataFrame(results)
    comparison = comparison.sort_values('recall', ascending=False).reset_index(drop=True)
    print('\n=== Model comparison ===\n')
    print(comparison.to_string(index=False, float_format='{:.4f}'.format))
    comparison.to_csv(MODELS_DIR / 'model_comparison.csv', index=False)
    print(f'\nSaved to {MODELS_DIR / "model_comparison.csv"}')

    # 6. Threshold tuning on recall-tuned model
    threshold_df, precision_arr, recall_arr, _ = tune_threshold(
        xgb_tuned, X_test, y_test,
    )
    threshold_df.to_csv(MODELS_DIR / 'threshold_analysis.csv', index=False)
    print(f'Saved to {MODELS_DIR / "threshold_analysis.csv"}')

    plot_precision_recall_tradeoff(
        precision_arr, recall_arr, threshold_df,
        MODELS_DIR / 'precision_recall_tradeoff.png',
    )

    # 7. Feature importance (best model = recall-tuned XGBoost)
    plot_feature_importance(xgb_tuned, feature_names, MODELS_DIR / 'feature_importance.png')

    # 8. Save models
    joblib.dump(xgb_tuned, MODELS_DIR / 'best_model.joblib')
    print(f'Recall-tuned model saved to {MODELS_DIR / "best_model.joblib"}')
    joblib.dump(xgb_f1, MODELS_DIR / 'best_model_f1.joblib')
    print(f'F1-tuned model saved to {MODELS_DIR / "best_model_f1.joblib"}')


if __name__ == '__main__':
    main()
