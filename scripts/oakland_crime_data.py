#!/usr/bin/env python3
"""Fetch Oakland crime data by ZIP code from Oakland Open Data portal.

Data source: Oakland Open Data - CrimeWatch (data.oaklandca.gov)
API: Socrata Open Data API (SODA)

Usage:
  python oakland_crime_data.py
  python oakland_crime_data.py --year-start 2017 --year-end 2024

No API key required for public datasets (but rate-limited without one).
Optional: Set SOCRATA_APP_TOKEN environment variable for higher rate limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import urllib.parse
import urllib.request
from collections import defaultdict
from typing import Dict, List, Sequence

DEFAULT_YEAR_START = 2017
DEFAULT_YEAR_END = 2024
DEFAULT_OUT = "data/oakland_crime_by_zip.csv"

# Oakland ZIP codes
OAKLAND_ZIPS = [
    "94601", "94602", "94603", "94605", "94606", "94607", "94608",
    "94609", "94610", "94611", "94612", "94613", "94615", "94618",
    "94619", "94621", "94623", "96409"
]

# Oakland Open Data CrimeWatch dataset ID
# This dataset contains crime incidents with location data
DATASET_ID = "ppgh-7dqv"  # CrimeWatch Data - has historical data


def fetch_json(url: str, app_token: str | None = None) -> List[Dict]:
    """Fetch JSON data from Socrata API."""
    req = urllib.request.Request(url)
    if app_token:
        req.add_header("X-App-Token", app_token)
    req.add_header("Accept", "application/json")
    
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = resp.read()
    return json.loads(data.decode("utf-8"))


def build_crime_url(year: int, limit: int = 50000, offset: int = 0) -> str:
    """Build Socrata API URL for crime data."""
    base = f"https://data.oaklandca.gov/resource/{DATASET_ID}.json"
    
    # Filter by year using datetime range
    start_date = f"{year}-01-01T00:00:00"
    end_date = f"{year}-12-31T23:59:59"
    where_clause = f"datetime >= '{start_date}' AND datetime <= '{end_date}'"
    
    params = {
        "$where": where_clause,
        "$limit": str(limit),
        "$offset": str(offset),
        "$select": "crimetype,datetime,address,city,state,location",
    }
    return f"{base}?{urllib.parse.urlencode(params)}"
    return f"{base}?{urllib.parse.urlencode(params)}"


def extract_zip_from_location(record: Dict) -> str | None:
    """Extract ZIP code from crime record."""
    # Try to get ZIP from address field
    address = record.get("address", "")
    
    # Look for 5-digit ZIP pattern at end of address
    import re
    zip_match = re.search(r'\b(946\d{2})\b', address)
    if zip_match:
        return zip_match.group(1)
    
    return None


def geocode_to_zip(lat: float, lon: float, zip_boundaries: Dict) -> str | None:
    """Simple point-in-polygon check for Oakland ZIP codes.
    
    Uses approximate bounding boxes for Oakland ZIPs.
    """
    # Approximate ZIP code centroids and boundaries for Oakland
    # Format: {zip: (min_lat, max_lat, min_lon, max_lon)}
    zip_boxes = {
        "94601": (37.773, 37.795, -122.225, -122.195),
        "94602": (37.795, 37.830, -122.225, -122.190),
        "94603": (37.735, 37.773, -122.195, -122.155),
        "94605": (37.760, 37.800, -122.175, -122.130),
        "94606": (37.785, 37.810, -122.255, -122.225),
        "94607": (37.795, 37.815, -122.295, -122.255),
        "94608": (37.830, 37.855, -122.295, -122.260),
        "94609": (37.825, 37.845, -122.270, -122.245),
        "94610": (37.805, 37.830, -122.255, -122.225),
        "94611": (37.820, 37.865, -122.235, -122.190),
        "94612": (37.800, 37.815, -122.280, -122.260),
        "94613": (37.785, 37.800, -122.190, -122.170),
        "94618": (37.840, 37.870, -122.260, -122.230),
        "94619": (37.785, 37.820, -122.200, -122.165),
        "94621": (37.755, 37.785, -122.225, -122.185),
    }
    
    for zip_code, (min_lat, max_lat, min_lon, max_lon) in zip_boxes.items():
        if min_lat <= lat <= max_lat and min_lon <= lon <= max_lon:
            return zip_code
    
    return None


def fetch_crime_data_for_year(year: int, app_token: str | None) -> List[Dict]:
    """Fetch all crime data for a given year with pagination."""
    all_records = []
    offset = 0
    limit = 50000
    
    while True:
        url = build_crime_url(year, limit, offset)
        try:
            records = fetch_json(url, app_token)
            if not records:
                break
            all_records.extend(records)
            if len(records) < limit:
                break
            offset += limit
        except Exception as e:
            print(f"    Warning: Error fetching page at offset {offset}: {e}", file=sys.stderr)
            break
    
    return all_records


def aggregate_by_zip(records: List[Dict], oakland_zips: set) -> Dict[str, Dict[str, int]]:
    """Aggregate crime counts by ZIP code and crime type."""
    # Initialize counters
    zip_crimes: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    
    for record in records:
        # Try to determine ZIP code
        zip_code = extract_zip_from_location(record)
        
        # If no ZIP from address, try geocoding from location
        if not zip_code and record.get("location"):
            loc = record["location"]
            try:
                coords = loc.get("coordinates", [])
                if len(coords) >= 2:
                    lon, lat = coords[0], coords[1]
                    if lat and lon:
                        zip_code = geocode_to_zip(lat, lon, {})
            except (ValueError, TypeError):
                pass
        
        # Skip if not an Oakland ZIP
        if not zip_code or zip_code not in oakland_zips:
            continue
        
        crime_type = record.get("crimetype", "UNKNOWN")
        zip_crimes[zip_code]["total"] += 1
        zip_crimes[zip_code][crime_type] += 1
    
    return zip_crimes


def categorize_crimes(zip_crimes: Dict[str, Dict[str, int]]) -> Dict[str, Dict[str, int]]:
    """Categorize crimes into broader categories."""
    categories = {
        "violent": ["ASSAULT", "ROBBERY", "HOMICIDE", "KIDNAPPING", "WEAPONS", "BATTERY"],
        "property": ["BURGLARY", "LARCENY", "THEFT", "VANDALISM", "VEHICLE", "ARSON"],
        "drug": ["DRUG", "NARCOTIC"],
        "other": [],
    }
    
    result: Dict[str, Dict[str, int]] = {}
    
    for zip_code, crimes in zip_crimes.items():
        result[zip_code] = {
            "total_crimes": crimes.get("total", 0),
            "violent_crimes": 0,
            "property_crimes": 0,
            "drug_crimes": 0,
            "other_crimes": 0,
        }
        
        for crime_type, count in crimes.items():
            if crime_type == "total":
                continue
            
            crime_upper = crime_type.upper()
            categorized = False
            
            for category, keywords in categories.items():
                if any(kw in crime_upper for kw in keywords):
                    result[zip_code][f"{category}_crimes"] += count
                    categorized = True
                    break
            
            if not categorized:
                result[zip_code]["other_crimes"] += count
    
    return result


def write_csv(all_data: List[Dict[str, Dict[str, int]]], years: List[int], out_path: str) -> None:
    """Write aggregated crime data to CSV."""
    columns = ["year", "zip_code", "total_crimes", "violent_crimes", "property_crimes", "drug_crimes", "other_crimes"]
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(columns)
        
        for year, data in zip(years, all_data):
            for zip_code in sorted(data.keys()):
                row = data[zip_code]
                writer.writerow([
                    year,
                    zip_code,
                    row.get("total_crimes", 0),
                    row.get("violent_crimes", 0),
                    row.get("property_crimes", 0),
                    row.get("drug_crimes", 0),
                    row.get("other_crimes", 0),
                ])


def parse_args(argv: Sequence[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fetch Oakland crime data by ZIP code from Oakland Open Data."
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
    return parser.parse_args(argv)


def main(argv: Sequence[str]) -> int:
    args = parse_args(argv)
    app_token = os.getenv("SOCRATA_APP_TOKEN")
    if not app_token:
        print("Note: SOCRATA_APP_TOKEN not set; requests may be rate-limited.", file=sys.stderr)
    
    oakland_zips = set(OAKLAND_ZIPS)
    years = list(range(args.year_start, args.year_end + 1))
    
    all_data: List[Dict[str, Dict[str, int]]] = []
    years_with_data: List[int] = []
    
    for year in years:
        print(f"Fetching crime data for {year}...")
        try:
            records = fetch_crime_data_for_year(year, app_token)
            print(f"  Retrieved {len(records)} crime records")
            
            if records:
                zip_crimes = aggregate_by_zip(records, oakland_zips)
                categorized = categorize_crimes(zip_crimes)
                
                # Ensure all Oakland ZIPs are represented
                for zip_code in oakland_zips:
                    if zip_code not in categorized:
                        categorized[zip_code] = {
                            "total_crimes": 0,
                            "violent_crimes": 0,
                            "property_crimes": 0,
                            "drug_crimes": 0,
                            "other_crimes": 0,
                        }
                
                all_data.append(categorized)
                years_with_data.append(year)
                print(f"  Aggregated to {len(categorized)} ZIP codes")
        except Exception as e:
            print(f"  Warning: Could not fetch data for {year}: {e}", file=sys.stderr)
    
    if all_data:
        write_csv(all_data, years_with_data, args.out)
        total_rows = sum(len(d) for d in all_data)
        print(f"Wrote {total_rows} rows to {args.out}")
    else:
        print("No data retrieved.", file=sys.stderr)
        return 1
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
