# Feature Library — Final Model

Reference for all **84 features** used by the final calibrated HistGradientBoosting model (`models/best_model_calibrated.joblib`). Features are built by `scripts/feature_engineering_final.py :: build_prior_year_features(df, target_year)` and saved to `data/train_features_final.csv` / `data/test_features_final.csv`.

**Core rule:** predicting year `T`, every feature below uses data strictly from years `< T`, *except* demographics and neighborhood (which are measured at year `T` and do not leak the attendance label).

Notation: `T` = target school year. `T−1` = most recent prior year. `T−2` = two years back.

---

## 1. Prior-year attendance & behavior (10)

Source: OUSD `evaldata_cleaned_final.csv`, student row from year `T−1`.

| Feature | Type | Definition |
|---|---|---|
| `prev_att_rate` | float | Attendance rate in year `T−1` (`DaysEnr − DaysAbs) / DaysEnr`). |
| `prev_days_abs` | int | Days absent in `T−1`. |
| `prev_days_enr` | int | Days enrolled in `T−1`. |
| `prev_susp` | int | Suspensions in `T−1`. |
| `prev_chronic` | 0/1 | `AttRate < 0.90` in `T−1`. |
| `prev_gpa` | float | `CurrWeightedTotGPA` in `T−1` (NaN for non-HS grades). |
| `prev_grade` | int | Grade level in `T−1`. |
| `prev_dist_km` | float | Home→school haversine distance in `T−1` (km). |
| `prev_school_homeless_rate` | float | CDE homeless rate at student's school in `T−1`. |
| `prev2_att_rate` | float | Attendance rate in year `T−2`. |
| `prev2_chronic` | 0/1 | Chronic flag in `T−2`. |

## 2. Multi-year aggregates (9)

Aggregated across **all** prior years with attendance data (`y < T`).

| Feature | Type | Definition |
|---|---|---|
| `prior_mean_att_rate` | float | Mean `AttRate` over all prior years. |
| `prior_min_att_rate` | float | Min `AttRate` over all prior years. |
| `prior_max_susp` | int | Max suspensions in any prior year. |
| `prior_total_susp` | int | Sum of suspensions over all prior years. |
| `prior_chronic_count` | int | # prior years with `chronic_absent=1`. |
| `prior_years_enrolled` | int | # prior years the student appears with valid attendance. |
| `prior_mean_gpa` | float | Mean `CurrWeightedTotGPA` across prior years. |
| `prior_mean_dist_km` | float | Mean home→school distance across prior years. |
| `prior_mean_school_homeless_rate` | float | Mean school homeless rate across prior years. |

## 3. Trend features (2)

| Feature | Type | Definition |
|---|---|---|
| `att_rate_slope` | float | OLS slope of `AttRate` over prior years (0 if < 2 years). |
| `att_rate_yoy_change` | float | `AttRate(T−1) − AttRate(T−2)`. |

## 4. School-level aggregates (4)

Computed on `T−1` cohort of the student's year-`T` school.

| Feature | Type | Definition |
|---|---|---|
| `school_chronic_rate` | float | Fraction of students chronic at that school in `T−1`. |
| `school_mean_att` | float | Mean `AttRate` at the school in `T−1`. |
| `school_size` | int | # students enrolled at the school in `T−1`. |
| `school_mean_susp` | float | Mean suspensions at the school in `T−1`. |

## 5. Demographics (25, one-hot)

Year-`T` student attributes. Safe: they don't reveal the `T` attendance label.

| Feature(s) | Type |
|---|---|
| `Grade` | int (target-year grade) |
| `age` | int (calendar year − birth year) |
| `Gen_F`, `Gen_M`, `Gen_N` | 0/1 |
| `Eth_African American`, `Eth_Asian`, `Eth_Filipino`, `Eth_Latino`, `Eth_Multiple Ethnicity`, `Eth_Native American`, `Eth_Not Reported`, `Eth_Pacific Islander`, `Eth_White` | 0/1 |
| `Fluency_ADEL`, `Fluency_EL`, `Fluency_EO`, `Fluency_IFEP`, `Fluency_RFEP`, `Fluency_TBD`, `Fluency_Unknown` | 0/1 |
| `SED_SED`, `SED_Not SED`, `SED_Unknown` | 0/1 (socio-economically disadvantaged) |
| `SpEd_Special Ed`, `SpEd_Not Special Ed` | 0/1 |

## 6. Current-year distance & homeless (2)

| Feature | Type | Definition |
|---|---|---|
| `dist_km` | float | Haversine home→school distance in year `T`. |
| `school_homeless_rate` | float | CDE homeless rate at assigned school in `T`. |

## 7. Neighborhood — school ZIP (14)

Per-year indicators for the ZIP of the student's school, year `T`. Source: Oakland Open Data (crime) + Census ACS 5-year (socioeconomic), joined in `merge_external.py`.

| Prefix `school_` suffix | Source | Definition |
|---|---|---|
| `total_crimes` | Socrata | All crime reports in ZIP × year. |
| `violent_crimes` | Socrata | Violent-offense count. |
| `property_crimes` | Socrata | Property-offense count. |
| `drug_crimes` | Socrata | Drug-offense count. |
| `other_crimes` | Socrata | Residual category. |
| `total_population` | ACS B01003 | ZIP population. |
| `poverty_rate_pct` | ACS B17001 | % below poverty line. |
| `median_household_income` | ACS B19013 | Median HH income ($). |
| `unemployment_rate_pct` | ACS B23025 | Unemployment rate %. |
| `high_school_plus_rate_pct` | ACS B15003 | % adults with ≥ HS diploma. |
| `college_degree_rate_pct` | ACS B15003 | % adults with ≥ bachelor's. |
| `median_gross_rent` | ACS B25064 | Median gross rent ($). |
| `median_home_value` | ACS B25077 | Median home value ($). |
| `uninsured_rate_pct` | ACS B27001 | % without health insurance. |

## 8. Neighborhood — home ZIP (14)

Same 14 indicators as §7, prefixed `home_` instead of `school_`, joined on the student's **residence** ZIP.

## 9. Missingness flags (2)

| Feature | Type | Definition |
|---|---|---|
| `has_prior_data` | 0/1 | 1 iff `prev_att_rate` is non-null (student appears in `T−1`). |
| `has_prior2_data` | 0/1 | 1 iff `prev2_att_rate` is non-null. |

---

## Excluded columns

These exist in the feature CSVs but are **not** passed to the model (`EXCLUDE` set in `feature_engineering_final.py`):

`ANON_ID`, `target_year`, `Zip`, `chronic_absent`, `absence_tier`, `att_rate_target`.

## Missing-value policy

- **Tree models** (final HistGB, XGB, LightGBM, RF): NaNs passed through; splits learn the missing direction.
- **Linear / distance models** (Logistic, SVC, KNN): numeric columns median-imputed + `StandardScaler`; dummies passed through. Fitted `ColumnTransformer` saved to `models/linear_preprocessor.joblib`.

## Top signals (permutation importance, AUC drop)

1. `prev_att_rate`
2. `prior_mean_att_rate`
3. `prev_chronic`
4. `att_rate_slope`
5. `prior_chronic_count`
6. `school_chronic_rate`
7. `prev_days_abs`, `prev_susp`, `Grade`, `age`

Neighborhood + demographic features give a second-order lift, largest for students without prior OUSD history.
