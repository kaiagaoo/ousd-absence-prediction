#!/usr/bin/env python3
"""Fetch Oakland socioeconomic data by ZIP (ZCTA) from the US Census API.

Data sources: ACS 5-year Detailed Tables
- B17001: Poverty Status
- B19013: Median Household Income
- B23025: Employment Status
- B15003: Educational Attainment
- B25064: Median Gross Rent
- B25077: Median Home Value
- B27001: Health Insurance Coverage
- B01003: Total Population

DEFAULT_YEAR_START = 2017
DEFAULT_YEAR_END = 2024

Usage:
  python census_oakland_socioeconomic.py
  python census_oakland_socioeconomic.py --year 2021 --out data/oakland_socioeconomic.csv

Set your Census API key in the environment variable CENSUS_API_KEY.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Dict, List, Sequence

DEFAULT_YEAR_START = 2017
DEFAULT_YEAR_END = 2024
DEFAULT_OUT = "data/oakland_socioeconomic_by_zip.csv"

# Common Oakland ZIP Code Tabulation Areas (ZCTAs).
OAKLAND_ZCTAS = [
    "94601", "94602", "94603", "94605", "94606", "94607", "94608",
    "94609", "94610", "94611", "94612", "94613", "94615", "94618",
    "94619", "94621", "94623",
]

# Census variables to fetch with friendly names
CENSUS_VARIABLES = {
    # Population
    "B01003_001E": "total_population",
    # Poverty
    "B17001_001E": "poverty_universe",
    "B17001_002E": "below_poverty",
    # Income
    "B19013_001E": "median_household_income",
    # Employment (population 16+)
    "B23025_001E": "employment_universe",
    "B23025_002E": "in_labor_force",
    "B23025_005E": "unemployed",
    # Educational attainment (population 25+)
    "B15003_001E": "education_universe",
    "B15003_017E": "high_school_diploma",
    "B15003_018E": "ged",
    "B15003_021E": "associates_degree",
    "B15003_022E": "bachelors_degree",
    "B15003_023E": "masters_degree",
    "B15003_024E": "professional_degree",
    "B15003_025E": "doctorate_degree",
    # Rent
    "B25064_001E": "median_gross_rent",
    # Home value
    "B25077_001E": "median_home_value",
    # Health insurance (civilian noninstitutionalized)
    "B27001_001E": "health_insurance_universe",
    "B27001_004E": "male_under6_no_insurance",
    "B27001_007E": "male_6to18_no_insurance",
    "B27001_010E": "male_19to25_no_insurance",
    "B27001_013E": "male_26to34_no_insurance",
    "B27001_016E": "male_35to44_no_insurance",
    "B27001_019E": "male_45to54_no_insurance",
    "B27001_022E": "male_55to64_no_insurance",
    "B27001_025E": "male_65to74_no_insurance",
    "B27001_028E": "male_75plus_no_insurance",
    "B27001_032E": "female_under6_no_insurance",
    "B27001_035E": "female_6to18_no_insurance",
    "B27001_038E": "female_19to25_no_insurance",
    "B27001_041E": "female_26to34_no_insurance",
    "B27001_044E": "female_35to44_no_insurance",
    "B27001_047E": "female_45to54_no_insurance",
    "B27001_050E": "female_55to64_no_insurance",
    "B27001_053E": "female_65to74_no_insurance",
    "B27001_056E": "female_75plus_no_insurance",
}


def fetch_json(url: str) -> List[List[str]]:
    """Fetch JSON data from URL."""
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def build_url(year: int, api_key: str | None, variables: list[str], zctas: list[str] | None = None) -> str:
    """Build Census API URL."""
    base = f"https://api.census.gov/data/{year}/acs/acs5"
    var_list = ",".join(["NAME"] + variables)
    
    # For years before 2020, use wildcard and filter locally
    if year < 2020 or zctas is None:
        params = {
            "get": var_list,
            "for": "zip code tabulation area:*",
            "in": "state:06",  # California
        }
    else:
        zcta_list = ",".join(zctas)
        params = {
            "get": var_list,
            "for": f"zip code tabulation area:{zcta_list}",
        }
    if api_key:
        params["key"] = api_key
    return f"{base}?{urllib.parse.urlencode(params)}"


def fetch_all_data(year: int, api_key: str | None, zctas: list[str]) -> Dict[str, Dict[str, str]]:
    """Fetch all socioeconomic data, handling API variable limits by batching."""
    all_vars = list(CENSUS_VARIABLES.keys())
    zcta_set = set(zctas)
    
    # Census API limits to ~50 variables per request, batch them
    batch_size = 45
    batches = [all_vars[i:i + batch_size] for i in range(0, len(all_vars), batch_size)]
    
    # Store results keyed by ZCTA
    results: Dict[str, Dict[str, str]] = {}
    
    # For years before 2020, we need to fetch all CA ZCTAs and filter
    use_wildcard = year < 2020
    
    for batch in batches:
        url = build_url(year, api_key, batch, None if use_wildcard else zctas)
        rows = fetch_json(url)
        header = rows[0]
        
        for row in rows[1:]:
            zcta = row[header.index("zip code tabulation area")]
            # Filter to Oakland ZCTAs
            if zcta not in zcta_set:
                continue
            if zcta not in results:
                results[zcta] = {"zcta": zcta, "name": row[0]}
            
            for var in batch:
                idx = header.index(var)
                results[zcta][var] = row[idx]
    
    return results


def compute_derived_fields(data: Dict[str, Dict[str, str]]) -> None:
    """Compute derived socioeconomic indicators."""
    for zcta, row in data.items():
        # Poverty rate
        try:
            poverty_universe = float(row.get("B17001_001E", 0) or 0)
            below_poverty = float(row.get("B17001_002E", 0) or 0)
            row["poverty_rate"] = f"{(below_poverty / poverty_universe * 100):.1f}" if poverty_universe > 0 else ""
        except (ValueError, ZeroDivisionError):
            row["poverty_rate"] = ""
        
        # Unemployment rate
        try:
            labor_force = float(row.get("B23025_002E", 0) or 0)
            unemployed = float(row.get("B23025_005E", 0) or 0)
            row["unemployment_rate"] = f"{(unemployed / labor_force * 100):.1f}" if labor_force > 0 else ""
        except (ValueError, ZeroDivisionError):
            row["unemployment_rate"] = ""
        
        # College degree rate (bachelor's or higher)
        try:
            edu_universe = float(row.get("B15003_001E", 0) or 0)
            bachelors = float(row.get("B15003_022E", 0) or 0)
            masters = float(row.get("B15003_023E", 0) or 0)
            professional = float(row.get("B15003_024E", 0) or 0)
            doctorate = float(row.get("B15003_025E", 0) or 0)
            college_plus = bachelors + masters + professional + doctorate
            row["college_degree_rate"] = f"{(college_plus / edu_universe * 100):.1f}" if edu_universe > 0 else ""
        except (ValueError, ZeroDivisionError):
            row["college_degree_rate"] = ""
        
        # High school or higher rate
        try:
            edu_universe = float(row.get("B15003_001E", 0) or 0)
            hs_diploma = float(row.get("B15003_017E", 0) or 0)
            ged = float(row.get("B15003_018E", 0) or 0)
            associates = float(row.get("B15003_021E", 0) or 0)
            bachelors = float(row.get("B15003_022E", 0) or 0)
            masters = float(row.get("B15003_023E", 0) or 0)
            professional = float(row.get("B15003_024E", 0) or 0)
            doctorate = float(row.get("B15003_025E", 0) or 0)
            hs_plus = hs_diploma + ged + associates + bachelors + masters + professional + doctorate
            row["high_school_plus_rate"] = f"{(hs_plus / edu_universe * 100):.1f}" if edu_universe > 0 else ""
        except (ValueError, ZeroDivisionError):
            row["high_school_plus_rate"] = ""
        
        # Uninsured rate
        try:
            health_universe = float(row.get("B27001_001E", 0) or 0)
            uninsured_vars = [
                "B27001_004E", "B27001_007E", "B27001_010E", "B27001_013E",
                "B27001_016E", "B27001_019E", "B27001_022E", "B27001_025E", "B27001_028E",
                "B27001_032E", "B27001_035E", "B27001_038E", "B27001_041E",
                "B27001_044E", "B27001_047E", "B27001_050E", "B27001_053E", "B27001_056E",
            ]
            total_uninsured = sum(float(row.get(v, 0) or 0) for v in uninsured_vars)
            row["uninsured_rate"] = f"{(total_uninsured / health_universe * 100):.1f}" if health_universe > 0 else ""
        except (ValueError, ZeroDivisionError):
            row["uninsured_rate"] = ""


def write_csv(all_data: List[Dict[str, Dict[str, str]]], out_path: str) -> None:
    """Write data to CSV with friendly column names."""
    # Define output columns with friendly names
    output_columns = [
        ("year", "year"),
        ("zcta", "zip_code"),
        ("name", "area_name"),
        ("B01003_001E", "total_population"),
        ("B17001_001E", "poverty_universe"),
        ("B17001_002E", "below_poverty"),
        ("poverty_rate", "poverty_rate_pct"),
        ("B19013_001E", "median_household_income"),
        ("B23025_002E", "labor_force"),
        ("B23025_005E", "unemployed"),
        ("unemployment_rate", "unemployment_rate_pct"),
        ("B15003_001E", "education_universe_25plus"),
        ("high_school_plus_rate", "high_school_plus_rate_pct"),
        ("college_degree_rate", "college_degree_rate_pct"),
        ("B25064_001E", "median_gross_rent"),
        ("B25077_001E", "median_home_value"),
        ("B27001_001E", "health_insurance_universe"),
        ("uninsured_rate", "uninsured_rate_pct"),
    ]
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow([col[1] for col in output_columns])
        # Write data rows sorted by year and ZCTA
        for data in all_data:
            for zcta in sorted(data.keys()):
                row = data[zcta]
                writer.writerow([row.get(col[0], "") for col in output_columns])


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Oakland socioeconomic data by ZCTA from the US Census API."
    )
    parser.add_argument(
        "--year-start", type=int, default=DEFAULT_YEAR_START,
        help=f"Start year (default: {DEFAULT_YEAR_START})"
    )
    parser.add_argument(
        "--year-end", type=int, default=DEFAULT_YEAR_END,
        help=f"End year (default: {DEFAULT_YEAR_END})"
    )
    parser.add_argument(
        "--out", type=str, default=DEFAULT_OUT,
        help=f"Output CSV path (default: {DEFAULT_OUT})"
    )
    parser.add_argument(
        "--zctas", type=str, default=",".join(OAKLAND_ZCTAS),
        help="Comma-separated ZCTAs to include (default: Oakland ZCTAs)"
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    api_key = os.getenv("CENSUS_API_KEY")
    if not api_key:
        print("Warning: CENSUS_API_KEY not set; requests may be rate-limited.", file=sys.stderr)
    
    zctas = [z.strip() for z in args.zctas.split(",") if z.strip()]
    years = list(range(args.year_start, args.year_end + 1))
    
    all_data: List[Dict[str, Dict[str, str]]] = []
    total_rows = 0
    
    for year in years:
        print(f"Fetching socioeconomic data for {len(zctas)} ZCTAs (year {year})...")
        try:
            data = fetch_all_data(year, api_key, zctas)
            # Add year to each row
            for zcta in data:
                data[zcta]["year"] = str(year)
            compute_derived_fields(data)
            all_data.append(data)
            total_rows += len(data)
            print(f"  Retrieved {len(data)} rows for {year}")
        except Exception as e:
            print(f"  Warning: Could not fetch data for {year}: {e}", file=sys.stderr)
    
    write_csv(all_data, args.out)
    print(f"Wrote {total_rows} total rows to {args.out}")
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
