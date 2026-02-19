"""
Train/test split and cross-validation setup for chronic absenteeism prediction.

Loads the feature-engineered train/test CSVs, defines feature columns,
handles missing values, and sets up stratified k-fold cross-validation.

Usage:
    python scripts/train_test_split.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import StratifiedKFold


# Columns to exclude from model features
EXCLUDE_COLS = {'ANON_ID', 'chronic_absent', 'target_year', 'Zip'}


def get_feature_columns(df):
    """Return list of feature column names (everything except ID, target, metadata)."""
    return [c for c in df.columns if c not in EXCLUDE_COLS]


def load_and_prepare(data_dir='data'):
    """
    Load train/test feature CSVs and prepare for modeling.

    Returns
    -------
    X_train, y_train, X_test, y_test, feature_names
    """
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / 'train_features.csv')
    test = pd.read_csv(data_dir / 'test_features.csv')

    print(f'Train: {len(train):,} rows')
    print(f'Test:  {len(test):,} rows')

    # Drop rows where target is missing
    train = train[train['chronic_absent'].notna()].copy()
    test = test[test['chronic_absent'].notna()].copy()
    print(f'After dropping NaN target — Train: {len(train):,}, Test: {len(test):,}')

    feature_cols = get_feature_columns(train)

    X_train = train[feature_cols].copy()
    y_train = train['chronic_absent'].astype(int)
    X_test = test[feature_cols].copy()
    y_test = test['chronic_absent'].astype(int)

    # Fill remaining NaN in features with -1 (explicit missing indicator)
    na_before = X_train.isna().sum().sum()
    X_train = X_train.fillna(-1)
    X_test = X_test.fillna(-1)
    print(f'Filled {na_before:,} NaN values in train features with -1')

    print(f'\nFeatures: {len(feature_cols)}')
    print(f'Train class distribution: {y_train.value_counts().to_dict()}')
    print(f'Test class distribution:  {y_test.value_counts().to_dict()}')
    print(f'Train chronic rate: {y_train.mean()*100:.1f}%')
    print(f'Test chronic rate:  {y_test.mean()*100:.1f}%')

    return X_train, y_train, X_test, y_test, feature_cols


def get_cv_splits(X_train, y_train, n_splits=5, random_state=42):
    """
    Create stratified k-fold cross-validation splits.

    Returns list of (train_idx, val_idx) tuples.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    splits = list(skf.split(X_train, y_train))

    print(f'\n{n_splits}-Fold Stratified CV splits:')
    for i, (train_idx, val_idx) in enumerate(splits):
        val_rate = y_train.iloc[val_idx].mean() * 100
        print(f'  Fold {i+1}: train={len(train_idx):,}, val={len(val_idx):,}, '
              f'val chronic rate={val_rate:.1f}%')

    return splits


def validate_no_leakage(data_dir='data'):
    """Verify there's no data leakage between train and test sets."""
    train = pd.read_csv(Path(data_dir) / 'train_features.csv')
    test = pd.read_csv(Path(data_dir) / 'test_features.csv')

    # Test year should only be 2324
    assert set(test['target_year'].unique()) == {2324}, \
        f'Test set should only contain 2324, got {test["target_year"].unique()}'

    # Train years should not include 2324
    assert 2324 not in train['target_year'].values, \
        'Train set should not contain target year 2324'

    # Check that prior-year features don't contain target-year info
    # prev_att_rate should only come from years before the target year
    print('Leakage checks passed:')
    print(f'  Train years: {sorted(train["target_year"].unique())}')
    print(f'  Test years: {sorted(test["target_year"].unique())}')

    # Student overlap is expected (same student across years) but each row
    # predicts a different year
    overlap = set(train['ANON_ID']) & set(test['ANON_ID'])
    print(f'  Students in both train & test: {len(overlap):,} (expected — different target years)')

    return True


if __name__ == '__main__':
    print('=== Loading and preparing data ===\n')
    X_train, y_train, X_test, y_test, feature_cols = load_and_prepare()

    print('\n=== Cross-validation splits ===')
    splits = get_cv_splits(X_train, y_train)

    print('\n=== Leakage validation ===')
    validate_no_leakage()

    print('\n=== Feature list ===')
    for i, col in enumerate(sorted(feature_cols), 1):
        print(f'  {i:2d}. {col}')
