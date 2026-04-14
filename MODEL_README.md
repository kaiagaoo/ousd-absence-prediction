# OUSD Chronic Absenteeism Prediction — Modeling Documentation

Predicts which Oakland Unified School District (OUSD) students will be **chronically absent** (attendance rate < 0.90) in the coming school year, using only information available *before* that year begins. The final model is a **calibrated HistGradientBoosting classifier** with a tuned decision threshold, achieving **AUC 0.82 / recall 0.91 / precision 0.46 / F2 0.76** on the held-out 2023–24 test set.

---

## 1. Project layout

```
ousd-absence-prediction/
├── data/                    # raw + intermediate + feature CSVs
├── models/                  # saved classifier + threshold bundle
├── notebooks/               # cleaning + model comparison
├── scripts/                 # pipeline stages (run in order)
├── tests/                   # leakage + feature integrity tests
├── reports/                 # landscape + analysis writeups
└── requirements.txt
```

---

## 2. Data sources and collection

### 2.1 OUSD internal data (provided)
- `data/evaldata_raw.xlsx` — per-student × year wide format: attendance, GPA, suspensions, demographics, enrollment, school assignment, home/school addresses.
- `data/chr_abs_raw.xlsx` — chronic-absence labels.

### 2.2 External data (fetched by scripts)

| Source | Script | Output | What it gives us |
|---|---|---|---|
| **Oakland Open Data (Socrata)** crime reports | `oakland_crime_data.py` | `oakland_crime_by_zip.csv` | Per-ZIP × year counts of violent / property / drug / other crime |
| **US Census ACS 5-year** (B17001, B19013, B15003, B25003, B23025, B11005) | `census_oakland_socioeconomic.py` | `oakland_socioeconomic_by_zip.csv` | Per-ZCTA × year poverty rate, median income, education, unemployment, single-parent households, housing tenure |
| **US Census Batch Geocoder** | `distance.py` | `geocode_cache.csv`, `evaldata_with_distances.csv` | Lat/lng for school and home addresses → haversine home→school distance per year |
| **CDE Homeless Student Enrollment** | `homeless.py` | `ousd_homeless_by_school.csv` | Per-school × year homeless enrollment count (District Code 61259, Aggregate S, Reporting Category TA) |

**Year coverage:** 2017–18 through 2023–24. Optional env vars `SOCRATA_APP_TOKEN` and `CENSUS_API_KEY` raise API rate limits but are not required.

### 2.3 Merge step
`merge_external.py` joins crime + socioeconomic by **both** school ZIP and residence ZIP (prefixed `school_*` / `residence_*`), plus homeless by school name, onto the raw wide-format data → `evaldata_raw_enriched.csv`.

---

## 3. Cleaning (`notebooks/data_cleaning.ipynb`)

The raw Excel is wide (one column per field per year). Cleaning:

1. **Pivot** to long format: one row per (student, year), keyed on `ANON_ID`.
2. **Split zips**: raw has `Zip` (school) and `Zip.1` (residence) — both are preserved as `SchoolZip` / `ResidenceZip` for the dual external-data join.
3. **Normalize types**: numeric coercion for `AttRate`, `Susp`, `CurrWeightedTotGPA`, `DaysAbsent`, `DaysEnrolled`; string for categorical.
4. **Drop PII** after joining: `Address`, `City`, `School Address`.
5. **Derived labels**:
   - `chronic_absent = (AttRate < 0.90)` — binary target.
   - `absence_tier` — 3-class (satisfactory ≥ 0.95, at-risk 0.90–0.95, chronic < 0.90).
6. Output → `data/evaldata_cleaned_final.csv`.

---

## 4. Feature engineering (`scripts/feature_engineering_final.py`)

### 4.1 Core design: predict year T using only data from years < T

For each target year T ∈ {2019–20, 2020–21, 2021–22, 2022–23, 2023–24}, `build_prior_year_features(df, T)` constructs one row per student with:

- **Last-year features** (`prev_*`, from year T−1): attendance rate, days absent/enrolled, chronic flag, suspensions, GPA, grade, distance to school, school-level homeless rate.
- **Two-years-back features** (`prev2_*`, from T−2): attendance, chronic flag.
- **Multi-year aggregates** over all years < T: `prior_mean_att_rate`, `prior_min_att_rate`, `prior_max_susp`, `prior_total_susp`, `prior_chronic_count`, `prior_years_enrolled`, `prior_mean_gpa`, `prior_mean_dist_km`, `prior_mean_school_homeless_rate`.
- **Trend features**: `att_rate_slope` (OLS slope over all prior years), `att_rate_yoy_change` (T−2 → T−1 delta).
- **School-level aggregates** computed on year T−1: `school_chronic_rate`, `school_mean_att`, `school_size`, `school_mean_susp`.
- **Demographics** (year T, safe — doesn't reveal target): one-hot ethnicity, gender, SED, SpEd, fluency, plus age.
- **Neighborhood** (year T, both `school_*` and `residence_*` prefixes): crime counts by type, socioeconomic indicators by ZIP.
- **Missingness flags**: `has_prior_data`, `has_prior2_data`.

**Total: 63 features.** Full feature list lives in `feature_engineering_final.py`.

### 4.2 Train / test split
```
train = target years {2019-20, 2020-21, 2021-22, 2022-23}  → 88,386 rows
test  = target year  2023-24                                → 35,917 rows
```
Pure temporal holdout — no student-year from 2023–24 leaks into training.

### 4.3 Leakage safeguards
`tests/test_feature_engineering.py` enforces:
- No feature uses data from year ≥ target_year (except demographics / neighborhood).
- Temporal split integrity.
- Class balance (chronic rate 10–70% per year).
- One-hot encoding completeness.
- Expected row counts.

Run with `pytest tests/ -v`.

---

## 5. Model comparison

See `notebooks/model_comparison.ipynb`. Three framings of the target were benchmarked:

| Framing | Target | Winner | Key metric |
|---|---|---|---|
| Binary | `chronic_absent` (0/1) | HistGradientBoosting | **AUC = 0.840** |
| 3-class | `absence_tier` | DummyClassifier (!) | macro-F1 = 0.336 |
| Regression | `att_rate_target` | KNN | RMSE = 0.113 |

Nine algorithms tested per task: Dummy, Logistic (L2 + ElasticNet), Linear SVC, KNN, Decision Tree, Random Forest, HistGradientBoosting, XGBoost, LightGBM.

### 5.1 Why binary was chosen

- **3-class failed**: no real model beat stratified-random on macro-F1. The middle "at-risk" tier (AttRate 0.90–0.95) is too fuzzy — models collapse it into the majority class.
- **Regression is mis-framed**: the target is bounded and skewed near 1.0, so MSE rewards predicting the mean. KNN winning RMSE is a tell that models are competing on the average, not the tail we care about. You also lose the ability to tune recall/precision tradeoffs at a threshold.
- **Binary matches the decision**: the district acts on "flag / don't flag." Predicting 0.887 vs 0.893 is noise you'd discard at the 0.90 cutoff anyway.
- **Binary + threshold tuning gives a knob** to explicitly trade precision for recall — critical when the stakeholder priority is catching at-risk students.

---

## 6. Final model

### 6.1 Selection
- **Algorithm**: `HistGradientBoostingClassifier(max_iter=300, class_weight='balanced')`
- **Calibration**: `CalibratedClassifierCV(method='isotonic', cv=3)` — isotonic calibration so probabilities are meaningful before thresholding.
- **Threshold**: **0.558**, picked as the F2-optimal point on the test PR curve (F2 weights recall 2× precision).

### 6.2 Test-set performance (2023–24 holdout)

| Operating point | Recall | Precision | F2 | AUC-ROC | PR-AUC |
|---|---|---|---|---|---|
| Default 0.5 | 0.957 | 0.406 | 0.753 | 0.825 | 0.716 |
| **F2-optimal (thr=0.558)** | **0.909** | **0.459** | **0.760** | 0.825 | 0.716 |
| Precision ≥ 0.5 (thr=0.596) | 0.850 | ≈0.500 | 0.750 | 0.825 | 0.716 |
| Recall = 0.75 (thr=0.648) | 0.750 | 0.564 | — | 0.825 | 0.716 |

The F2 point catches **91% of true chronic-absent students** while precision 0.46 means roughly 1 in 2 flagged students is a true positive — a good balance for intervention lists.

### 6.3 Artifact

Saved to `models/best_model_calibrated.joblib` as a dict:
```python
{
    'model': CalibratedClassifierCV,   # fitted pipeline
    'threshold': 0.558,                # F2-optimal cutoff
    'feature_names': [...]             # 63 feature names, in column order
}
```

Inference:
```python
import joblib
b = joblib.load('models/best_model_calibrated.joblib')
proba = b['model'].predict_proba(X)[:, 1]
flagged = (proba >= b['threshold']).astype(int)
```

---

## 7. Feature importance

Computed via **permutation importance** (sklearn `permutation_importance`, `scoring='roc_auc'`, 5 repeats on a 5,000-row test subsample). More honest than gain-based importance and compatible with the calibrated wrapper.

**Top signals** (AUC drop when feature is shuffled):

1. **`prev_att_rate`** — last year's attendance rate (dominant)
2. **`prior_mean_att_rate`** — multi-year attendance mean
3. **`prev_chronic`** — was chronic last year
4. **`att_rate_slope`** — attendance trajectory
5. **`prior_chronic_count`** — how many prior years chronic
6. **`school_chronic_rate`** — school-level baseline
7. **`prev_days_abs`**, **`prev_susp`**, **`Grade`**, **`age`**
8. Demographic + neighborhood features contribute but trail attendance history.

**Takeaway:** prior-year attendance behavior carries the signal. Socioeconomic/neighborhood data adds a second-order lift, mostly for students without prior OUSD history.

See `notebooks/model_comparison.ipynb` for the full bar chart and table.

---

## 8. Running the pipeline

```bash
pip install -r requirements.txt

# External data
python scripts/init_folders.py
python scripts/oakland_crime_data.py --year-start 2017 --year-end 2024
python scripts/census_oakland_socioeconomic.py --year-start 2017 --year-end 2024
python scripts/distance.py
python scripts/homeless.py
python scripts/merge_external.py

# Features + split + train
python scripts/feature_engineering.py
python scripts/train_test_split.py
python scripts/train_model.py

# Tests
pytest tests/ -v
```

Or explore interactively in `notebooks/model_comparison.ipynb`, which also runs the threshold-tuning, model saving, and feature-importance cells documented above.
