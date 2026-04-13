"""
Download CDE Homeless Student Enrollment files, filter to OUSD (District Code 61259),
and aggregate per-school counts + rates by year.

Outputs:
  data/homeless/hseYYYY.txt                 (raw downloads)
  data/ousd_homeless_by_school.csv          (School Code x year wide-ish table)
"""
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "homeless"
RAW_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = ROOT / "data" / "ousd_homeless_by_school.csv"

OUSD_DISTRICT_CODE = "61259"

FILES = {
    "2024-25": "https://www3.cde.ca.gov/demo-downloads/homeless/hse2425.txt",
    "2023-24": "https://www3.cde.ca.gov/demo-downloads/homeless/hse2324.txt",
    "2022-23": "https://www3.cde.ca.gov/demo-downloads/homeless/hse2223.txt",
    "2021-22": "https://www3.cde.ca.gov/demo-downloads/homeless/hse2122.txt",
    "2020-21": "https://www3.cde.ca.gov/demo-downloads/homeless/hse2021.txt",
    "2019-20": "https://www3.cde.ca.gov/demo-downloads/homeless/hse1920.txt",
}

YEAR_TO_TAG = {
    "2024-25": "2425", "2023-24": "2324", "2022-23": "2223",
    "2021-22": "2122", "2020-21": "2021", "2019-20": "1920",
}


def download(year, url):
    path = RAW_DIR / (url.rsplit("/", 1)[-1])
    if path.exists() and path.stat().st_size > 0:
        print(f"  {year}: cached ({path.name})")
        return path
    print(f"  {year}: downloading...")
    r = requests.get(url, timeout=120)
    if r.status_code != 200:
        print(f"    HTTP {r.status_code}")
        return None
    path.write_bytes(r.content)
    return path


def load_ousd(path, year):
    df = pd.read_csv(path, sep="\t", dtype=str, encoding="latin-1", low_memory=False)
    df.columns = [c.strip() for c in df.columns]
    # Filter to OUSD at the school level with Reporting Category == "TA" (Total / All)
    df = df[df["District Code"].astype(str).str.strip() == OUSD_DISTRICT_CODE]
    df = df[df["Aggregate Level"].astype(str).str.strip() == "S"]
    df = df[df["Reporting Category"].astype(str).str.strip() == "TA"]
    for col in ("Cumulative Enrollment", "Homeless Student Enrollment"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["year"] = year
    df["year_tag"] = YEAR_TO_TAG[year]
    df["homeless_rate"] = df["Homeless Student Enrollment"] / df["Cumulative Enrollment"]
    return df[[
        "year", "year_tag", "School Code", "School Name",
        "Cumulative Enrollment", "Homeless Student Enrollment", "homeless_rate",
    ]].rename(columns={
        "School Code": "school_code",
        "School Name": "school_name",
        "Cumulative Enrollment": "enrollment",
        "Homeless Student Enrollment": "homeless_n",
    })


def main():
    print("=" * 70)
    print("CDE Homeless Enrollment — OUSD")
    print("=" * 70)

    frames = []
    for year, url in FILES.items():
        p = download(year, url)
        if p is None:
            continue
        try:
            frames.append(load_ousd(p, year))
        except Exception as e:
            print(f"  {year}: parse failed: {e}")

    if not frames:
        print("No data loaded.")
        sys.exit(1)

    out = pd.concat(frames, ignore_index=True)
    print(f"\nRows (OUSD school x year): {len(out)}")
    out.to_csv(OUT_CSV, index=False)
    print(f"Saved {OUT_CSV}")


if __name__ == "__main__":
    main()
