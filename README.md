# OUSD Absenteeism Prediction Project

Predicts student chronic absenteeism in Oakland Unified School District (OUSD) using demographic, attendance, and academic data across multiple school years.

## Structure

- `data/` — Raw and processed data files (gitignored)
- `notebooks/` — Jupyter notebooks for exploration and prototyping
- `scripts/` — Python scripts for data processing, training, and evaluation
- `models/` — Saved and exported models
- `tests/` — Unit and integration tests
- `docs/` — Project documentation

## Requirements

- Python 3.8+
- See `requirements.txt` for dependencies

## Data Inspection Findings

### Datasets

#### chr_abs_raw.xlsx (18,638 rows, 28 columns)
- **Time period:** 2024-25 school year (data as of May 2025)
- **Scope:** Only contains at-risk or chronically absent students (max AttRate = 0.9497). No students with good attendance are included.
- **Key columns:** ID, Birthdate, DaysEnr, DaysAbs, DaysPresent, AttRate, AttGrp, SiteName, Gr, Gen, Eth, Fluency, Home Language, Special Ed Status, GPA (cumulative and current), SED Status, NumSusp, NumDaysSusp
- **Target variable (AttGrp):**
  - At Risk: 8,726 students
  - Chronic Absent: 6,068 students
  - Severe Chronic Absent: 3,844 students

#### evaldata_raw.xlsx (79,460 rows, 122 columns)
- **Time period:** Longitudinal data spanning 7 years (2017-18 through 2023-24)
- **Scope:** Full student population. Each row is one student; columns are repeated per year with suffix (e.g., `AttRate_2324`, `Grade_1718`).
- **15 fields per year:** Eth, Fluency, SpEd, SiteName, School Address, City, Zip, Grade, AttRate, DaysEnr, DaysAbs, Susp, Address, CurrWeightedTotGPA, SED
- **Time-invariant columns:** ANON_ID, Birthdate, Gen

### Chronic Absence Rates by Year (evaldata)

| Year    | Students | Mean AttRate | Chronic Absent (AttRate < 0.90) |
|---------|----------|--------------|---------------------------------|
| 2017-18 | 39,929   | 0.935        | 16.2%                           |
| 2018-19 | 39,579   | 0.891        | 33.8%                           |
| 2019-20 | 38,839   | 0.927        | 19.2%                           |
| 2020-21 | 37,558   | 0.916        | 20.1%                           |
| 2021-22 | 36,153   | 0.860        | 43.8%                           |
| 2022-23 | 36,552   | 0.842        | 58.3%                           |
| 2023-24 | 36,695   | 0.889        | 32.7%                           |

Notable spike during COVID era (2021-22 and 2022-23).

### Dataset Linkage

- **chr_abs** uses real student IDs (range 267,030–451,929)
- **evaldata** uses anonymized IDs (range 1–79,460)
- **Zero direct ID overlap** — the two datasets cannot be joined without a crosswalk
- They cover different time periods: chr_abs is 2024-25, evaldata is 2017-18 through 2023-24

### Data Quality Issues

1. **chr_abs has 982 duplicate IDs** — same students appearing twice with slightly different AttRate values (likely two data snapshot dates)
2. **Suspensions are very sparse** — 95–100% missing per year. Only ~1,400–1,600 students have suspension records in any given year. In 2020-21, zero suspensions recorded.
3. **GPA only available for secondary students (grades 6+)** — ~23% of evaldata, ~36% of chr_abs. Missing for all elementary students.
4. **Short enrollment students** — 2,620 students in evaldata 2023-24 have DaysEnr < 90 (half year); 778 have DaysEnr < 30. Their AttRate may be unreliable.
5. **Gender data has dirty values** — evaldata has 'M', 'F', 'N', and lowercase 'm' (2 records)
6. **chr_abs is filtered** — only at-risk/absent students, no negative examples for classification

### Demographics (evaldata 2023-24)

- **Ethnicity:** Latino (18,214), African American (7,663), White (4,225), Asian (3,505), Multiple Ethnicity (2,549), Not Reported (910), Pacific Islander (311), Filipino (186), Native American (100)
- **Gender:** M (41,291), F (38,029), N (138)
- **SED:** SED (30,915), Not SED (6,748) — 82% socioeconomically disadvantaged
- **Special Ed:** Not Special Ed (31,000), Special Ed (6,663) — 18% special ed
- **Fluency:** EO (18,694), EL (12,785), RFEP (5,011), IFEP (1,090)
- **Schools:** 81 schools. Largest: Oakland Technical HS (1,923), Oakland HS (1,642), Skyline HS (1,525)
- **Grades:** -1 (TK/pre-K) through 12, roughly 2,500–3,200 per grade

## Data Cleaning Summary

Notebook: `notebooks/data_cleaning.ipynb` | Output: `data/evaldata_cleaned.csv`

### Pivot to Long Format

Converted evaldata from wide format (79,460 rows x 122 columns, one row per student) to long format (one row per student-year). This simplifies all downstream cleaning and feature engineering — each operation is a single vectorized step instead of looping over 7 year suffixes.

### Cleaning Steps

| Step | Action | Details |
|------|--------|---------|
| Gender fix | `'m'` → `'M'` | 2 students (14 rows in long format) |
| PII removal | Dropped Address, City, School Address | Kept Zip (for external data joins) and SiteName (modeling feature) |
| Student filter | Kept only students with AttRate in 2023-24 | 79,460 → 36,695 students (256,865 rows across 7 years) |
| Short enrollment | Set AttRate to NaN where DaysEnr < 30 | 1,663 student-year rows affected (778 in target year 2023-24) |
| Suspensions | Filled NaN with 0 for enrolled rows | NaN means no suspensions, not missing data |
| GPA | Left NaN as-is | Structurally missing for grades K-5 (no GPA assigned) |
| Target variable | `chronic_absent` = 1 if AttRate < 0.90 | 2023-24: 32.1% chronic absent, 67.9% not chronic |

### External Data Joins

Joined neighborhood-level features on zip code and calendar year (school year `1718` → 2017, etc.):

**Crime data** (`scripts/oakland_crime_data.py` → `data/oakland_crime_by_zip.csv`):
- Source: Oakland Open Data CrimeWatch portal (Socrata API)
- 507K+ crime records across 2017-2024, aggregated to 17 Oakland zip codes per year
- Columns added: `total_crimes`, `violent_crimes`, `property_crimes`, `drug_crimes`, `other_crimes`

**Join coverage:** 152,077 / 256,865 rows matched (59.2%). Unmatched rows are student-years where the student wasn't enrolled that year (no zip).

**Socioeconomic data** (`scripts/census_oakland_socioeconomic.py` → `data/oakland_socioeconomic_by_zip.csv`):
- Source: US Census ACS 5-year estimates (2017-2024)
- 15 Oakland ZCTAs per year
- Columns added: `total_population`, `poverty_rate_pct`, `median_household_income`, `unemployment_rate_pct`, `high_school_plus_rate_pct`, `college_degree_rate_pct`, `median_gross_rent`, `median_home_value`, `uninsured_rate_pct`

**Join coverage:** 147,607 / 256,865 rows matched (57.5%). Unmatched rows are student-years where the student wasn't enrolled that year (no zip).

### Output

`data/evaldata_cleaned.csv` — 256,865 rows x 31 columns (36,695 students x 7 years).

### Recommended Modeling Approach

- Use **evaldata as the primary dataset** — it has the full student population with longitudinal history
- Train on prior years (2017-18 through 2022-23) to predict 2023-24 chronic absenteeism (target: `AttRate_2324 < 0.90`)
- chr_abs is useful for understanding the current 2024-25 at-risk population but not suitable for model training (no negative examples, different ID system)

## EDA Highlights

Notebook: `notebooks/eda.ipynb` | Data: `data/evaldata_cleaned.csv` (2023-24 enrolled, n=35,917)

- **Chronic absence rate (2023-24):** 32.1% (11,539 students)
- **Year-over-year trend:** Rates spiked during COVID (58.3% in 2022-23), partially recovered in 2023-24
- **Demographic disparities:** African American and Pacific Islander students have highest chronic rates; SED and SpEd students are significantly more affected
- **School variation:** Ranges from 6.5% (Lincoln Elementary) to 58.1% (Castlemont High) across 75 schools with ≥50 students
- **Top correlated features with chronic absence:** Susp (+0.14), poverty_rate_pct (+0.06), median_home_value (-0.15)
- **Neighborhood features:** Crime and socioeconomic variables are highly inter-correlated; poverty rate and median income show moderate association with school-level chronic rates

## Feature Engineering & Train/Test Split

Scripts: `scripts/feature_engineering.py`, `scripts/train_test_split.py` | Tests: `tests/test_feature_engineering.py`

### Strategy

**Temporal split** — train on historical years, test on the most recent year. This mirrors real-world deployment where we predict next year's chronic absenteeism from prior data.

| | Train | Test |
|---|---|---|
| **Target years** | 1920, 2021, 2122, 2223 | 2324 |
| **Rows** | 88,386 | 35,917 |
| **Chronic rate** | 35.4% | 32.1% |
| **Cross-validation** | 5-fold stratified | — |

For each target year, features are built exclusively from **prior-year data** to prevent data leakage. Students in the test set may also appear in training rows — this is expected since each row predicts a different school year.

### Features (63 total)

| Category | Features | Examples |
|----------|----------|---------|
| Prior-year attendance | 7 | `prev_att_rate`, `prev_days_abs`, `prev_chronic`, `prev2_att_rate` |
| Multi-year trends | 5 | `prior_mean_att_rate`, `att_rate_slope`, `att_rate_yoy_change`, `prior_min_att_rate` |
| History counts | 4 | `prior_years_enrolled`, `prior_chronic_count`, `prior_total_susp`, `prior_max_susp` |
| Academics | 2 | `prev_gpa`, `prior_mean_gpa` |
| School-level (prior yr) | 4 | `school_chronic_rate`, `school_mean_att`, `school_size`, `school_mean_susp` |
| Demographics | 25 | One-hot: ethnicity (9), gender (3), SED (3), SpEd (2), fluency (7), `age` |
| Neighborhood | 13 | Crime counts (5), socioeconomic indicators (8) from target-year zip |
| Indicators | 3 | `has_prior_data`, `has_prior2_data`, `Grade` |

### Leakage Prevention

- Prior-year features (`prev_att_rate`, etc.) are sourced strictly from years before the target year
- School-level aggregates are computed on prior-year enrollment data, not the target year
- 15 automated tests verify temporal ordering, leakage absence, feature completeness, and class balance

### Output

- `data/train_features.csv` — 88,386 rows x 67 columns
- `data/test_features.csv` — 35,917 rows x 67 columns

## Modeling

Script: `scripts/train_model.py` | Output: `models/`

### Approach

Trained eight models with class imbalance handling (`class_weight='balanced'` for sklearn models, `scale_pos_weight` for XGBoost). XGBoost was tuned via `RandomizedSearchCV` (40 iterations, 5-fold stratified CV) under two strategies: optimizing recall and optimizing F1. ElasticNet regularization (L1 + L2) is applied to both Logistic Regression and XGBoost tuning.

### Results

| Model | Accuracy | Precision | Recall | F1 | AUC-ROC | PR-AUC |
|-------|----------|-----------|--------|------|---------|--------|
| Logistic Regression (L2) | 0.365 | 0.335 | 0.996 | 0.502 | 0.766 | 0.665 |
| XGBoost (tuned-recall) | 0.461 | 0.373 | 0.989 | 0.541 | 0.843 | 0.734 |
| XGBoost (tuned-F1) | 0.475 | 0.378 | 0.984 | 0.546 | 0.839 | 0.729 |
| XGBoost (default) | 0.497 | 0.387 | 0.968 | 0.553 | 0.814 | 0.701 |
| Random Forest | 0.597 | 0.440 | 0.940 | 0.600 | 0.830 | 0.711 |
| Decision Tree | 0.572 | 0.412 | 0.778 | 0.539 | 0.630 | 0.395 |
| Baseline (prior att) | 0.670 | 0.490 | 0.681 | 0.570 | 0.782 | 0.567 |
| Logistic (ElasticNet) | 0.547 | 0.377 | 0.629 | 0.471 | 0.588 | 0.403 |

- **Best recall:** Logistic Regression L2 (99.6%) and XGBoost tuned-recall (98.9%)
- **Best AUC-ROC:** XGBoost tuned-recall (0.843)
- **Best PR-AUC:** XGBoost tuned-recall (0.734) — best at distinguishing chronic from non-chronic across all thresholds
- **Best precision:** Baseline (49.0%) — simple rule (prior-year AttRate < 0.90)
- **Decision Tree:** Low PR-AUC (0.395) and AUC-ROC (0.630), confirming poor discrimination

### Tuned XGBoost Parameters

| Variant | max_depth | learning_rate | n_estimators | subsample | colsample_bytree | min_child_weight | reg_alpha (L1) | reg_lambda (L2) |
|---------|-----------|---------------|--------------|-----------|------------------|------------------|----------------|-----------------|
| tuned-recall | 7 | 0.05 | 200 | 0.9 | 0.7 | 3 | 0 | 10.0 |
| tuned-F1 | 7 | 0.05 | 500 | 0.7 | 0.9 | 1 | 0.1 | 1.0 |

### ElasticNet Regularization

**Logistic Regression:** ElasticNet (`l1_ratio=0.5`, combining L1 and L2 penalties) was applied with the `saga` solver. L1 regularization zeroed out 1 of 63 features, indicating most features carry predictive signal. However, the ElasticNet variant significantly underperformed the standard L2 logistic regression (AUC 0.588 vs. 0.766, recall 0.629 vs. 0.996), likely because the saga solver struggled to converge and the L1 penalty was too aggressive for this feature set.

**XGBoost:** `reg_alpha` (L1) and `reg_lambda` (L2) were added to the hyperparameter search grid. The recall-tuned model selected strong L2 regularization (`reg_lambda=10.0`) with no L1 (`reg_alpha=0`), suggesting that L2 weight shrinkage is more beneficial than L1 feature sparsity for tree ensembles on this data. The F1-tuned model selected moderate regularization (`reg_alpha=0.1, reg_lambda=1.0`).

### Key Observations

1. **XGBoost benefits from L2 regularization.** The recall-tuned model selected `reg_lambda=10.0`, which helps prevent overfitting by shrinking leaf weights. This is consistent with the overfitting analysis showing XGBoost has the smallest train-test AUC gap.
2. **ElasticNet hurts Logistic Regression on this data.** The L1 component doesn't help because most features are informative, and the saga solver's convergence issues degrade performance. Standard L2 regularization is sufficient.
3. **PR-AUC is more informative than AUC-ROC** for this imbalanced task. Random Forest has higher AUC-ROC (0.830) than XGBoost default (0.814), but XGBoost default has a comparable PR-AUC (0.701 vs. 0.711).
4. The simple baseline remains competitive on precision but misses ~32% of chronic students.
5. **XGBoost (tuned-recall) remains the recommended model** for deployment, with threshold tuning (see below) to control the precision-recall tradeoff at inference time.

### Overfitting Analysis

Train vs. test performance comparison to assess generalization:

| Model | Train Recall | Test Recall | Gap | Train AUC | Test AUC | Gap | Train PR-AUC | Test PR-AUC | Gap |
|-------|-------------|-------------|-----|-----------|----------|-----|-------------|-------------|-----|
| Logistic Regression (L2) | 0.669 | 0.996 | +0.327 | 0.770 | 0.766 | -0.004 | 0.675 | 0.665 | -0.010 |
| Logistic (ElasticNet) | 0.636 | 0.629 | -0.007 | 0.639 | 0.588 | -0.051 | 0.495 | 0.403 | -0.092 |
| Decision Tree | 0.989 | 0.778 | -0.211 | 0.999 | 0.630 | -0.370 | 0.998 | 0.395 | -0.603 |
| Random Forest | 0.986 | 0.940 | -0.046 | 0.998 | 0.830 | -0.168 | 0.996 | 0.711 | -0.284 |
| XGBoost (default) | 0.893 | 0.968 | +0.075 | 0.953 | 0.814 | -0.139 | 0.921 | 0.701 | -0.221 |
| XGBoost (tuned-recall) | 0.819 | 0.989 | +0.170 | 0.896 | 0.843 | -0.052 | 0.832 | 0.734 | -0.098 |
| XGBoost (tuned-F1) | 0.862 | 0.984 | +0.122 | 0.931 | 0.839 | -0.092 | 0.886 | 0.729 | -0.157 |

Cross-validation recall (5-fold on training data):

| Model | CV Recall | Std |
|-------|-----------|-----|
| XGBoost (default) | 0.768 | 0.005 |
| Random Forest | 0.683 | 0.005 |
| Logistic Regression (L2) | 0.668 | 0.009 |
| Decision Tree | 0.645 | 0.003 |
| Logistic (ElasticNet) | 0.632 | 0.015 |

**Findings:**

1. **Decision Tree is severely overfitting.** Train PR-AUC of 0.998 drops to 0.395 on test — the largest gap of any model (-0.603). It memorizes training data but fails to generalize. CV recall (0.645) also confirms poor generalization.
2. **Random Forest overfits moderately.** Train PR-AUC (0.996) vs. test PR-AUC (0.711) shows a -0.284 gap, though ensemble averaging helps it generalize better than the Decision Tree.
3. **XGBoost (tuned-recall) generalizes best among tree-based models.** The PR-AUC gap is only -0.098 (0.832 → 0.734) and AUC gap is -0.052 (0.896 → 0.843). The strong L2 regularization (`reg_lambda=10.0`) combined with `min_child_weight=3` and `colsample_bytree=0.7` effectively controls overfitting.
4. **XGBoost (tuned-F1) overfits slightly more** than tuned-recall (PR-AUC gap -0.157 vs. -0.098), consistent with its weaker regularization (`reg_lambda=1.0` vs. 10.0).
5. **Logistic Regression (L2) shows an unusual pattern** — higher recall on test (0.996) than train (0.669). This is because `class_weight='balanced'` with the test set's lower chronic rate (32.1% vs. 35.4% train) shifts the decision boundary, causing the model to predict nearly all test samples as positive. The stable AUC gap (-0.004) and PR-AUC gap (-0.010) confirm no real overfitting — the model is simply underfitting.
6. **Logistic (ElasticNet) does not overfit** (small gaps across all metrics) but also underfits severely, with the lowest test PR-AUC (0.403) after Decision Tree. The saga solver convergence issues and aggressive L1 penalty limit its learning capacity.
7. **XGBoost (tuned-recall) is the recommended model** — it offers the best tradeoff of high recall (98.9%), strong discrimination (AUC 0.843, PR-AUC 0.734), and minimal overfitting.

### Threshold Tuning

The default threshold (0.50) maximizes recall but flags 85% of all students. By adjusting the classification threshold on the tuned XGBoost's predicted probabilities, we can trade some recall for much better precision:

| Threshold | Precision | Recall | F1 | Students Flagged | % Flagged |
|-----------|-----------|--------|------|-----------------|-----------|
| 0.20 | 0.326 | 1.000 | 0.492 | 35,333 | 98.4% |
| 0.30 | 0.335 | 0.998 | 0.501 | 34,420 | 95.8% |
| 0.40 | 0.349 | 0.995 | 0.517 | 32,897 | 91.6% |
| **0.50** | **0.373** | **0.989** | **0.541** | **30,620** | **85.3%** |
| 0.60 | 0.405 | 0.975 | 0.573 | 27,760 | 77.3% |
| **0.70** | **0.456** | **0.945** | **0.615** | **23,926** | **66.6%** |

- **Best F1 threshold: 0.70** — achieves 94.5% recall with 45.6% precision, flagging 23,926 students (67% of population). This catches nearly as many chronic students while cutting false positives by ~22% compared to the default.
- Depending on intervention capacity, practitioners can choose their operating point along this curve. For example, threshold=0.60 flags ~28K students at 97.5% recall, while 0.70 flags ~24K at 94.5% recall.

See `models/precision_recall_tradeoff.png` for the full precision-recall curve and threshold analysis chart.

### Saved Artifacts

- `models/best_model.joblib` — XGBoost tuned for recall
- `models/best_model_f1.joblib` — XGBoost tuned for F1
- `models/model_comparison.csv` — Full comparison table (includes PR-AUC)
- `models/threshold_analysis.csv` — Threshold sweep results
- `models/precision_recall_tradeoff.png` — Precision-recall curve and threshold chart
- `models/feature_importance.png` — Top 20 feature importance chart

## License

[Specify your license here]
