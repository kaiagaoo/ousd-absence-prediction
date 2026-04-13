"""
Feature engineering on data/evaldata_cleaned_final.csv.

Extends scripts/feature_engineering.py with:
- Distance (prev, prior-mean, current-year)
- School homeless rate (prev, prior-mean, current-year)
- Dual neighborhood features (both school_* and home_* crime + socio)

Outputs:
    data/train_features_final.csv
    data/test_features_final.csv

Also exposes `make_model_matrices(train, test)` — returns tree-ready (NaNs kept)
and linear-ready (median-imputed + scaled) matrices. Trees handle missing
values natively; linear / distance-based models do not.
"""

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"

YEAR_TAGS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]
TAG_TO_INT = {t: int(t) for t in YEAR_TAGS}  # '2324' -> 2324

CAT_COLS = ["Gen", "Eth", "Fluency", "SpEd", "SED"]

NEIGH_COLS = [  # per-year neighborhood measures (exist with school_/home_ prefix)
    "total_crimes", "violent_crimes", "property_crimes", "drug_crimes", "other_crimes",
    "total_population", "poverty_rate_pct", "median_household_income",
    "unemployment_rate_pct", "high_school_plus_rate_pct", "college_degree_rate_pct",
    "median_gross_rent", "median_home_value", "uninsured_rate_pct",
]


def compute_slope(values):
    vals = values.dropna()
    if len(vals) < 2:
        return 0.0
    return np.polyfit(np.arange(len(vals)), vals.values, 1)[0]


def build_prior_year_features(df, target_year):
    """Build one-row-per-student feature frame predicting `target_year`."""
    all_years = sorted(df["year"].unique())
    prior_years = [y for y in all_years if y < target_year]

    target_rows = df[(df["year"] == target_year) & df["AttRate"].notna()].copy()
    target_ids = set(target_rows["ANON_ID"])
    prior = df[(df["ANON_ID"].isin(target_ids)) &
               (df["year"].isin(prior_years)) &
               df["AttRate"].notna()].copy()

    # --- Most recent prior year (attendance, distance, homeless) ---
    if prior_years:
        last_y = prior_years[-1]
        prev = prior[prior["year"] == last_y][[
            "ANON_ID", "AttRate", "DaysAbs", "DaysEnr", "Susp", "chronic_absent",
            "CurrWeightedTotGPA", "Grade", "dist_km", "School_Homeless_Rate",
        ]].copy()
        prev.columns = [
            "ANON_ID", "prev_att_rate", "prev_days_abs", "prev_days_enr",
            "prev_susp", "prev_chronic", "prev_gpa", "prev_grade",
            "prev_dist_km", "prev_school_homeless_rate",
        ]
    else:
        prev = pd.DataFrame({"ANON_ID": list(target_ids)})

    if len(prior_years) >= 2:
        y2 = prior_years[-2]
        prev2 = prior[prior["year"] == y2][["ANON_ID", "AttRate", "chronic_absent"]].copy()
        prev2.columns = ["ANON_ID", "prev2_att_rate", "prev2_chronic"]
        prev = prev.merge(prev2, on="ANON_ID", how="left")

    # --- Multi-year aggregates ---
    multi = prior.groupby("ANON_ID").agg(
        prior_mean_att_rate=("AttRate", "mean"),
        prior_min_att_rate=("AttRate", "min"),
        prior_max_susp=("Susp", "max"),
        prior_total_susp=("Susp", "sum"),
        prior_chronic_count=("chronic_absent", "sum"),
        prior_years_enrolled=("year", "count"),
        prior_mean_gpa=("CurrWeightedTotGPA", "mean"),
        prior_mean_dist_km=("dist_km", "mean"),
        prior_mean_school_homeless_rate=("School_Homeless_Rate", "mean"),
    ).reset_index()

    slope = (prior.sort_values("year").groupby("ANON_ID")["AttRate"]
             .apply(compute_slope).reset_index())
    slope.columns = ["ANON_ID", "att_rate_slope"]

    if len(prior_years) >= 2:
        last2 = prior[prior["year"].isin(prior_years[-2:])].sort_values("year")
        yoy = last2.groupby("ANON_ID")["AttRate"].apply(
            lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 2 else np.nan
        ).reset_index()
        yoy.columns = ["ANON_ID", "att_rate_yoy_change"]
    else:
        yoy = pd.DataFrame({"ANON_ID": list(target_ids), "att_rate_yoy_change": np.nan})

    # --- School-level aggregates from last prior year ---
    if prior_years:
        last_prior = df[(df["year"] == last_y) & df["AttRate"].notna()]
        school_agg = last_prior.groupby("SiteName").agg(
            school_chronic_rate=("chronic_absent", "mean"),
            school_mean_att=("AttRate", "mean"),
            school_size=("ANON_ID", "count"),
            school_mean_susp=("Susp", "mean"),
        ).reset_index()
        target_school = target_rows[["ANON_ID", "SiteName"]].merge(
            school_agg, on="SiteName", how="left"
        ).drop(columns=["SiteName"])
    else:
        target_school = pd.DataFrame({"ANON_ID": list(target_ids)})

    # --- Demographics (target year) ---
    demo_cols = ["ANON_ID"] + CAT_COLS + ["Grade"]
    target_demo = target_rows[demo_cols].copy()

    # Age (approx)
    cal_year = 2000 + target_year // 100
    bd = target_rows[["ANON_ID", "Birthdate"]].copy()
    bd["Birthdate"] = pd.to_datetime(bd["Birthdate"], errors="coerce")
    bd["age"] = cal_year - bd["Birthdate"].dt.year
    bd = bd[["ANON_ID", "age"]]

    # --- Neighborhood (target year, both school_ and home_ prefixes) ---
    neigh_full = ["ANON_ID"]
    for prefix in ("school", "home"):
        for c in NEIGH_COLS:
            col = f"{prefix}_{c}"
            if col in target_rows.columns:
                neigh_full.append(col)
    target_neigh = target_rows[neigh_full].copy()

    # --- Target-year distance + homeless ---
    cur = target_rows[["ANON_ID", "dist_km", "School_Homeless_Rate"]].copy()
    cur.columns = ["ANON_ID", "dist_km", "school_homeless_rate"]

    # --- Targets (binary, 3-class, continuous) ---
    tgt = target_rows[["ANON_ID", "chronic_absent", "absence_tier", "AttRate"]].copy()
    tgt = tgt.rename(columns={"AttRate": "att_rate_target"})

    features = target_demo
    for right in [bd, prev, multi, slope, yoy, target_school, target_neigh, cur, tgt]:
        features = features.merge(right, on="ANON_ID", how="left")

    # One-hot encode
    for col in CAT_COLS:
        if col in features.columns:
            features[col] = features[col].fillna("Unknown")
    features = pd.get_dummies(features, columns=CAT_COLS, drop_first=False, dtype=int)

    features["has_prior_data"] = features["prev_att_rate"].notna().astype(int)
    features["has_prior2_data"] = features.get(
        "prev2_att_rate", pd.Series(dtype=float)
    ).notna().astype(int)
    features["target_year"] = target_year
    return features


def build_all_features(input_path=None, output_dir=None):
    input_path = Path(input_path or DATA / "evaldata_cleaned_final.csv")
    output_dir = Path(output_dir or DATA)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {input_path}")
    df = pd.read_csv(input_path, low_memory=False)
    # year may be string like '1920' — convert to int for ordering
    df["year"] = df["year"].astype(int)
    print(f"  rows: {len(df):,}  students: {df['ANON_ID'].nunique():,}")

    target_years = [1920, 2021, 2122, 2223, 2324]
    frames = []
    for ty in target_years:
        print(f"  target {ty}...")
        f = build_prior_year_features(df, ty)
        print(f"    {len(f):,} students, {f.shape[1]} cols, "
              f"chronic={f['chronic_absent'].mean()*100:.1f}%")
        frames.append(f)

    # Column alignment
    all_cols = set().union(*[set(f.columns) for f in frames])
    for f in frames:
        for c in all_cols - set(f.columns):
            f[c] = 0

    train = pd.concat([f for f in frames if f["target_year"].iloc[0] != 2324],
                      ignore_index=True)
    test = [f for f in frames if f["target_year"].iloc[0] == 2324][0]

    cols = sorted(train.columns.tolist())
    train, test = train[cols], test[cols]

    train.to_csv(output_dir / "train_features_final.csv", index=False)
    test.to_csv(output_dir / "test_features_final.csv", index=False)
    print(f"\nSaved train ({train.shape}) and test ({test.shape})")
    return train, test


# ---------------------------------------------------------------------------
# Model-matrix helper
# ---------------------------------------------------------------------------

TARGET_COLS = {"chronic_absent", "absence_tier", "att_rate_target"}
EXCLUDE = {"ANON_ID", "target_year", "Zip"} | TARGET_COLS


def make_model_matrices(train, test, target_col="chronic_absent",
                        save_preprocessor=True):
    """
    Produce tree-ready and linear-ready feature matrices.

    target_col:
        'chronic_absent'     -> binary classification (int 0/1)
        'absence_tier'       -> 3-class classification (int 0/1/2)
        'att_rate_target'    -> regression on AttRate (float)

    Returns:
        X_train_tree, X_test_tree,      # NaNs preserved, no scaling
        X_train_linear, X_test_linear,  # median-imputed, scaled numerics
        y_train, y_test,
        feature_names,
        preprocessor                    # fitted ColumnTransformer
    """
    if target_col not in TARGET_COLS:
        raise ValueError(f"target_col must be one of {TARGET_COLS}")

    train = train[train[target_col].notna()].copy()
    test = test[test[target_col].notna()].copy()

    feature_cols = [c for c in train.columns if c not in EXCLUDE]
    X_train = train[feature_cols].copy()
    X_test = test[feature_cols].copy()
    cast = float if target_col == "att_rate_target" else int
    y_train = train[target_col].astype(cast).reset_index(drop=True)
    y_test = test[target_col].astype(cast).reset_index(drop=True)

    # --- Tree-ready: keep NaN, keep dummies as-is ---
    X_train_tree = X_train.reset_index(drop=True)
    X_test_tree = X_test.reset_index(drop=True)

    # --- Linear-ready: scale numerics, passthrough dummies ---
    # Dummy columns come from one-hot encoding and are named like 'Eth_Asian'
    dummy_cols = [c for c in feature_cols if any(
        c.startswith(p + "_") for p in CAT_COLS
    )]
    numeric_cols = [c for c in feature_cols if c not in dummy_cols]

    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_cols),
        ("dum", "passthrough", dummy_cols),
    ])
    X_train_linear = pd.DataFrame(
        pre.fit_transform(X_train), columns=numeric_cols + dummy_cols
    ).reset_index(drop=True)
    X_test_linear = pd.DataFrame(
        pre.transform(X_test), columns=numeric_cols + dummy_cols
    ).reset_index(drop=True)

    if save_preprocessor:
        MODELS.mkdir(parents=True, exist_ok=True)
        joblib.dump(pre, MODELS / "linear_preprocessor.joblib")

    return (X_train_tree, X_test_tree, X_train_linear, X_test_linear,
            y_train, y_test, feature_cols, pre)


if __name__ == "__main__":
    build_all_features()
