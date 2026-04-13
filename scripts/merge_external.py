"""
Merge external variables (distance, homeless, crime, socioeconomic) onto
evaldata_raw.xlsx in wide format (one row per student, per-year suffixed columns).

Inputs:
  data/evaldata_raw.xlsx
  data/evaldata_with_distances.csv       (from scripts/distance.py)
  data/ousd_homeless_by_school.csv       (from scripts/homeless.py)
  data/oakland_crime_by_zip.csv
  data/oakland_socioeconomic_by_zip.csv

Output:
  data/evaldata_raw_enriched.csv
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

YEARS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]
# School year (int used in crime/socio tables) = starting calendar year
YEAR_TAG_TO_INT = {
    "1718": 2017, "1819": 2018, "1920": 2019, "2021": 2020,
    "2122": 2021, "2223": 2022, "2324": 2023,
}


def clean_zip(z):
    if z is None or pd.isna(z):
        return ""
    s = re.sub(r"[^0-9]", "", str(z).split(".")[0])
    return s[:5]


def load_raw():
    dist_csv = DATA / "evaldata_with_distances.csv"
    if dist_csv.exists():
        print(f"Loading {dist_csv.name} (raw + distances)")
        return pd.read_csv(dist_csv, low_memory=False)
    print("Loading evaldata_raw.xlsx (no distances file yet)")
    return pd.read_excel(DATA / "evaldata_raw.xlsx")


def merge_zip_panel(df, panel, panel_cols, zip_col, year_tag, prefix):
    """Left-join a (year, zip_code, ...) panel onto df using df[zip_col] as zip
    for the given year_tag. Adds `{prefix}_{col}_{year_tag}` columns."""
    year_int = YEAR_TAG_TO_INT[year_tag]
    sub = panel[panel["year"] == year_int][["zip_code"] + panel_cols].copy()
    sub["zip_code"] = sub["zip_code"].astype(str).str.zfill(5)

    key = df[zip_col].apply(clean_zip) if zip_col in df.columns else pd.Series([""] * len(df))
    tmp = pd.DataFrame({"zip_code": key})
    merged = tmp.merge(sub, on="zip_code", how="left")
    for c in panel_cols:
        df[f"{prefix}_{c}_{year_tag}"] = merged[c].values
    return df


def merge_homeless(df, homeless):
    """Join by school name per year (raw data has SiteName_YY; homeless table has school_name)."""
    # Build lookup: (year_tag, name_norm) -> (homeless_n, rate, enrollment)
    h = homeless.copy()
    h["name_norm"] = h["school_name"].astype(str).str.lower().str.strip()
    h["year_tag"] = h["year_tag"].astype(str)

    for y in YEARS:
        site_col = f"SiteName_{y}"
        if site_col not in df.columns:
            continue
        sub = h[h["year_tag"] == y][["name_norm", "homeless_n", "homeless_rate", "enrollment"]]
        names = df[site_col].astype(str).str.lower().str.strip()
        tmp = pd.DataFrame({"name_norm": names}).merge(sub, on="name_norm", how="left")
        df[f"School_Homeless_{y}"] = tmp["homeless_n"].values
        df[f"School_Homeless_Rate_{y}"] = tmp["homeless_rate"].values
        df[f"School_Enrollment_{y}"] = tmp["enrollment"].values
    return df


def main():
    df = load_raw()
    print(f"  rows: {len(df):,}  cols: {len(df.columns)}")

    crime = pd.read_csv(DATA / "oakland_crime_by_zip.csv")
    socio = pd.read_csv(DATA / "oakland_socioeconomic_by_zip.csv")
    crime_cols = ["total_crimes", "violent_crimes", "property_crimes", "drug_crimes", "other_crimes"]
    socio_cols = [
        "total_population", "poverty_rate_pct", "median_household_income",
        "unemployment_rate_pct", "high_school_plus_rate_pct",
        "college_degree_rate_pct", "median_gross_rent", "median_home_value",
        "uninsured_rate_pct",
    ]

    print("Merging crime + socioeconomic by school zip and residence zip per year...")
    for y in YEARS:
        school_zip = f"Zip_{y}"
        home_zip = f"Zip_{y}.1"
        if school_zip in df.columns:
            df = merge_zip_panel(df, crime, crime_cols, school_zip, y, "school")
            df = merge_zip_panel(df, socio, socio_cols, school_zip, y, "school")
        if home_zip in df.columns:
            df = merge_zip_panel(df, crime, crime_cols, home_zip, y, "home")
            df = merge_zip_panel(df, socio, socio_cols, home_zip, y, "home")

    homeless_csv = DATA / "ousd_homeless_by_school.csv"
    if homeless_csv.exists():
        print("Merging homeless...")
        df = merge_homeless(df, pd.read_csv(homeless_csv))
    else:
        print(f"  skipping homeless (run scripts/homeless.py first to create {homeless_csv.name})")

    out = DATA / "evaldata_raw_enriched.csv"
    df.to_csv(out, index=False)
    print(f"\nSaved {out}  shape={df.shape}")


if __name__ == "__main__":
    main()
