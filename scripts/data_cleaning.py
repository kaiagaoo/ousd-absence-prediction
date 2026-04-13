"""
Clean data/evaldata_raw_enriched.csv (wide, with distance / homeless / dual-zip
crime + socio columns already merged) into a long-format file suitable for
feature engineering.

Output: data/evaldata_cleaned_final.csv  (one row per student-year)

Missing values are intentionally preserved. Imputation strategy depends on the
downstream model (tree-based models handle NaN; linear / distance-based models
need imputation + scaling). That split happens in
`feature_engineering_final.make_model_matrices`, not here.
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
IN_CSV = ROOT / "data" / "evaldata_raw_enriched.csv"
OUT_CSV = ROOT / "data" / "evaldata_cleaned_final.csv"

YEARS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]
YEAR_TO_CAL = {"1718": 2017, "1819": 2018, "1920": 2019, "2021": 2020,
               "2122": 2021, "2223": 2022, "2324": 2023}
ID_COLS = ["ANON_ID", "Birthdate", "Gen"]
PII_PREFIXES = ("Address", "City", "School Address")


def find_base_fields(columns):
    """Collect {base_name: [years_present]} from columns like 'AttRate_1920'."""
    pat = re.compile(r"^(.*)_(" + "|".join(YEARS) + r")$")
    bases = {}
    for c in columns:
        m = pat.match(c)
        if not m:
            continue
        base, yr = m.group(1), m.group(2)
        bases.setdefault(base, []).append(yr)
    return bases


def pivot_long(df_wide):
    bases = find_base_fields(df_wide.columns)
    # Drop PII bases entirely — never carry into long format
    bases = {b: yrs for b, yrs in bases.items() if not b.startswith(PII_PREFIXES)}

    frames = []
    for yr in YEARS:
        cols = {f"{b}_{yr}": b for b, yrs in bases.items() if yr in yrs}
        sub = df_wide[ID_COLS + list(cols.keys())].copy()
        sub.rename(columns=cols, inplace=True)
        sub["year"] = yr
        frames.append(sub)
    return pd.concat(frames, ignore_index=True)


def main():
    print(f"Loading {IN_CSV}")
    df_wide = pd.read_csv(IN_CSV, low_memory=False)
    print(f"  wide shape: {df_wide.shape}")

    df = pivot_long(df_wide)
    print(f"  long shape: {df.shape}")
    print(f"  columns: {len(df.columns)}")

    # --- Gender cleanup ---
    df["Gen"] = df["Gen"].replace({"m": "M"})

    # --- Filter to students with valid AttRate in 2324 ---
    target_ids = df.loc[(df["year"] == "2324") & df["AttRate"].notna(), "ANON_ID"].unique()
    df = df[df["ANON_ID"].isin(target_ids)].copy()
    print(f"  students with 2324 AttRate: {len(target_ids):,}")

    # --- Short enrollment -> AttRate NaN ---
    short = df["DaysEnr"].notna() & (df["DaysEnr"] < 30)
    print(f"  short-enrollment rows (DaysEnr<30): {short.sum():,}")
    df.loc[short, "AttRate"] = np.nan

    # --- Suspensions: 0 where enrolled, else NaN ---
    enrolled = df["DaysEnr"].notna()
    df.loc[enrolled, "Susp"] = df.loc[enrolled, "Susp"].fillna(0)

    # --- Targets: three flavors so downstream can pick ---
    # 1) binary  chronic_absent : 1 if AttRate < 0.90
    # 2) 3-class absence_tier   : 0=satisfactory (>=0.95), 1=at-risk (0.90-0.95),
    #                             2=chronic (<0.90)  — California / OUSD convention
    # 3) continuous AttRate (already present)
    has = df["AttRate"].notna()
    df["chronic_absent"] = np.where(has, (df["AttRate"] < 0.90).astype(int), np.nan)
    tier = np.where(
        df["AttRate"] < 0.90, 2,
        np.where(df["AttRate"] < 0.95, 1, 0),
    )
    df["absence_tier"] = np.where(has, tier, np.nan)

    # --- Calendar year (bookkeeping) ---
    df["calendar_year"] = df["year"].map(YEAR_TO_CAL)

    # --- QC ---
    print("\n  AttRate available per year:")
    print(df.groupby("year")["AttRate"].apply(lambda x: x.notna().sum()).to_string())
    t = df.loc[df["year"] == "2324", "chronic_absent"].dropna()
    print(f"\n  2324 chronic rate: {t.mean()*100:.1f}%  (n={len(t):,})")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}  shape={df.shape}")


if __name__ == "__main__":
    main()
