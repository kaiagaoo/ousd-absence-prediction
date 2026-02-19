"""
Feature engineering for chronic absenteeism prediction.

Builds per-student feature vectors for a target year using only data from
prior years (no data leakage). Features include:
- Prior-year attendance (AttRate, DaysAbs, chronic_absent)
- Multi-year attendance trends (mean, slope over available years)
- Suspensions and GPA history
- Demographics (static)
- School-level aggregates (computed from prior-year data)
- Neighborhood crime/socioeconomic features (from target year zip)

Usage:
    python scripts/feature_engineering.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def compute_slope(values):
    """OLS slope for a series of values indexed 0..n-1. Returns 0 if <2 values."""
    vals = values.dropna()
    if len(vals) < 2:
        return 0.0
    x = np.arange(len(vals))
    return np.polyfit(x, vals.values, 1)[0]


def build_prior_year_features(df, target_year):
    """
    For each student enrolled in target_year, build features from prior years.

    Parameters
    ----------
    df : DataFrame
        Full longitudinal dataset (all years).
    target_year : int
        The year to predict (e.g. 2324).

    Returns
    -------
    DataFrame with one row per student enrolled in target_year, columns are features + target.
    """
    all_years = sorted(df['year'].unique())
    prior_years = [y for y in all_years if y < target_year]

    # Students enrolled in target year (have valid AttRate)
    target_rows = df[(df['year'] == target_year) & df['AttRate'].notna()].copy()
    target_ids = set(target_rows['ANON_ID'])

    # Prior-year data for these students
    prior = df[(df['ANON_ID'].isin(target_ids)) &
               (df['year'].isin(prior_years)) &
               df['AttRate'].notna()].copy()

    # --- Prior-year attendance features ---
    # Most recent prior year
    if len(prior_years) >= 1:
        last_prior_year = prior_years[-1]
        last_year = prior[prior['year'] == last_prior_year][['ANON_ID', 'AttRate', 'DaysAbs',
                                                              'DaysEnr', 'Susp', 'chronic_absent',
                                                              'CurrWeightedTotGPA', 'Grade']].copy()
        last_year.columns = ['ANON_ID', 'prev_att_rate', 'prev_days_abs', 'prev_days_enr',
                             'prev_susp', 'prev_chronic', 'prev_gpa', 'prev_grade']
    else:
        last_year = pd.DataFrame({'ANON_ID': list(target_ids)})

    # 2nd most recent prior year
    if len(prior_years) >= 2:
        second_prior_year = prior_years[-2]
        second_year = prior[prior['year'] == second_prior_year][['ANON_ID', 'AttRate', 'chronic_absent']].copy()
        second_year.columns = ['ANON_ID', 'prev2_att_rate', 'prev2_chronic']
        last_year = last_year.merge(second_year, on='ANON_ID', how='left')

    # Multi-year aggregates
    multi = prior.groupby('ANON_ID').agg(
        prior_mean_att_rate=('AttRate', 'mean'),
        prior_min_att_rate=('AttRate', 'min'),
        prior_max_susp=('Susp', 'max'),
        prior_total_susp=('Susp', 'sum'),
        prior_chronic_count=('chronic_absent', 'sum'),
        prior_years_enrolled=('year', 'count'),
        prior_mean_gpa=('CurrWeightedTotGPA', 'mean'),
    ).reset_index()

    # Attendance slope (trend over all prior years)
    att_slope = prior.sort_values('year').groupby('ANON_ID')['AttRate'].apply(compute_slope).reset_index()
    att_slope.columns = ['ANON_ID', 'att_rate_slope']

    # Year-over-year change in AttRate (most recent transition)
    if len(prior_years) >= 2:
        last2 = prior[prior['year'].isin(prior_years[-2:])].sort_values('year')
        yoy = last2.groupby('ANON_ID')['AttRate'].apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 2 else np.nan
        ).reset_index()
        yoy.columns = ['ANON_ID', 'att_rate_yoy_change']
    else:
        yoy = pd.DataFrame({'ANON_ID': list(target_ids), 'att_rate_yoy_change': np.nan})

    # --- School-level features (from prior year data to avoid leakage) ---
    if len(prior_years) >= 1:
        prior_last = df[(df['year'] == last_prior_year) & df['AttRate'].notna()]
        school_agg = prior_last.groupby('SiteName').agg(
            school_chronic_rate=('chronic_absent', 'mean'),
            school_mean_att=('AttRate', 'mean'),
            school_size=('ANON_ID', 'count'),
            school_mean_susp=('Susp', 'mean'),
        ).reset_index()

        # Map to target-year students via their target-year school
        target_school = target_rows[['ANON_ID', 'SiteName']].copy()
        target_school = target_school.merge(school_agg, on='SiteName', how='left')
        target_school = target_school.drop(columns=['SiteName'])
    else:
        target_school = pd.DataFrame({'ANON_ID': list(target_ids)})

    # --- Demographics from target year ---
    demo_cols = ['ANON_ID', 'Gen', 'Eth', 'Fluency', 'SpEd', 'SED', 'Grade', 'Zip']
    target_demo = target_rows[demo_cols].copy()

    # Age at start of school year (approximate: target year maps to calendar year)
    target_birthdate = target_rows[['ANON_ID', 'Birthdate']].copy()
    target_birthdate['Birthdate'] = pd.to_datetime(target_birthdate['Birthdate'], errors='coerce')
    # Map year code to approximate school start (e.g. 2324 -> Sept 2023)
    cal_year = 2000 + target_year // 100
    target_birthdate['age'] = cal_year - target_birthdate['Birthdate'].dt.year
    target_birthdate = target_birthdate[['ANON_ID', 'age']]

    # --- Neighborhood features from target year ---
    neigh_cols = ['ANON_ID', 'total_crimes', 'violent_crimes', 'property_crimes',
                  'drug_crimes', 'other_crimes', 'poverty_rate_pct',
                  'median_household_income', 'unemployment_rate_pct',
                  'high_school_plus_rate_pct', 'college_degree_rate_pct',
                  'median_gross_rent', 'median_home_value', 'uninsured_rate_pct']
    target_neigh = target_rows[neigh_cols].copy()

    # --- Target variable ---
    target_var = target_rows[['ANON_ID', 'chronic_absent']].copy()

    # --- Merge everything ---
    features = target_demo.copy()
    for right_df in [target_birthdate, last_year, multi, att_slope, yoy,
                     target_school, target_neigh, target_var]:
        features = features.merge(right_df, on='ANON_ID', how='left')

    # Encode categoricals
    features = encode_categoricals(features)

    # Add indicator for whether student has prior-year data
    features['has_prior_data'] = features['prev_att_rate'].notna().astype(int)
    features['has_prior2_data'] = features.get('prev2_att_rate', pd.Series(dtype=float)).notna().astype(int)

    # Add target_year column for bookkeeping
    features['target_year'] = target_year

    return features


def encode_categoricals(df):
    """One-hot encode categorical demographic columns."""
    cat_cols = ['Gen', 'Eth', 'Fluency', 'SpEd', 'SED']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')
    df = pd.get_dummies(df, columns=cat_cols, drop_first=False, dtype=int)
    return df


def build_all_features(input_path, output_dir):
    """
    Build feature sets for all viable target years and save to CSV.

    Training years: 1920, 2021, 2122, 2223 (each using prior years as features)
    Test year: 2324 (using 1718-2223 as features)
    """
    print(f'Loading {input_path}...')
    df = pd.read_csv(input_path)
    print(f'  {df.shape[0]:,} rows, {df["ANON_ID"].nunique():,} students')

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Build features for each target year
    # 1718 and 1819 skipped: 1718 has no prior data, 1819 has only 1 prior year
    target_years = [1920, 2021, 2122, 2223, 2324]
    all_frames = []

    for ty in target_years:
        print(f'\nBuilding features for target year {ty}...')
        feat = build_prior_year_features(df, ty)
        print(f'  {len(feat):,} students, {feat.shape[1]} columns')
        print(f'  Has prior data: {feat["has_prior_data"].mean()*100:.1f}%')
        print(f'  Chronic rate: {feat["chronic_absent"].mean()*100:.1f}%')
        all_frames.append(feat)

    # Align columns across all years (some dummies may differ)
    all_cols = set()
    for f in all_frames:
        all_cols.update(f.columns)
    for i, f in enumerate(all_frames):
        for col in all_cols - set(f.columns):
            all_frames[i][col] = 0

    # Split into train and test
    train_frames = [f for f in all_frames if f['target_year'].iloc[0] != 2324]
    test_frame = [f for f in all_frames if f['target_year'].iloc[0] == 2324][0]

    train = pd.concat(train_frames, ignore_index=True)

    # Ensure column order matches
    cols = sorted(train.columns.tolist())
    train = train[cols]
    test_frame = test_frame[cols]

    train_path = output_dir / 'train_features.csv'
    test_path = output_dir / 'test_features.csv'
    train.to_csv(train_path, index=False)
    test_frame.to_csv(test_path, index=False)

    print(f'\n--- Summary ---')
    print(f'Train: {len(train):,} rows ({train["target_year"].value_counts().to_dict()})')
    print(f'Test:  {len(test_frame):,} rows (2324)')
    print(f'Features: {train.shape[1]} columns')
    print(f'Train chronic rate: {train["chronic_absent"].mean()*100:.1f}%')
    print(f'Test chronic rate:  {test_frame["chronic_absent"].mean()*100:.1f}%')
    print(f'\nSaved: {train_path}')
    print(f'Saved: {test_path}')

    return train, test_frame


if __name__ == '__main__':
    build_all_features('data/evaldata_cleaned.csv', 'data')
