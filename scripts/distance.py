"""
Geocode school and home addresses in evaldata_raw.xlsx using the US Census
batch geocoder, then compute per-year home->school haversine distance (km).

Output: data/evaldata_with_distances.csv  (raw columns + *_lat_YY, *_lon_YY, dist_km_YY)
"""
import re
from math import radians, sin, cos, sqrt, atan2
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
RAW_XLSX = ROOT / "data" / "evaldata_raw.xlsx"
OUT_CSV = ROOT / "data" / "evaldata_with_distances.csv"
CACHE_CSV = ROOT / "data" / "geocode_cache.csv"

YEARS = ["1718", "1819", "1920", "2021", "2122", "2223", "2324"]
BATCH_SIZE = 9000  # Census geocoder limit is 10,000


def clean_zip(z):
    if z is None or pd.isna(z):
        return ""
    s = re.sub(r"[^0-9\-]", "", str(z).split(".")[0].strip())
    return s[:5] if len(s) >= 5 else s


def load_cache(path):
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    return {r["key"]: (r["lat"], r["lon"]) for _, r in df.iterrows()}


def save_cache(cache, path):
    pd.DataFrame(
        [{"key": k, "lat": v[0], "lon": v[1]} for k, v in cache.items()]
    ).to_csv(path, index=False)


def collect_addresses(df):
    out = {}
    for y in YEARS:
        for addr_col, city_col, zip_col in [
            (f"School Address_{y}", f"City_{y}", f"Zip_{y}"),
            (f"Address_{y}", f"City_{y}.1", f"Zip_{y}.1"),
        ]:
            if addr_col not in df.columns:
                continue
            for _, row in df[[addr_col, city_col, zip_col]].drop_duplicates().iterrows():
                addr = "" if pd.isna(row[addr_col]) else str(row[addr_col]).strip()
                city = "" if pd.isna(row[city_col]) else str(row[city_col]).strip()
                zipc = clean_zip(row[zip_col])
                if addr and addr.lower() != "nan":
                    out[f"{addr}|{city}|{zipc}"] = (addr, city, zipc)
    return out


def geocode_batch(batch_keys, addresses):
    rows = []
    for i, k in enumerate(batch_keys):
        street, city, zipc = addresses[k]
        # Sanitize commas/quotes in CSV payload
        street_s = street.replace('"', "").replace(",", " ")
        city_s = city.replace('"', "").replace(",", " ")
        rows.append(f'{i},"{street_s}","{city_s}","CA","{zipc}"')
    payload = "\n".join(rows)

    resp = requests.post(
        "https://geocoding.geo.census.gov/geocoder/locations/addressbatch",
        files={"addressFile": ("addr.csv", payload, "text/csv")},
        data={"benchmark": "Public_AR_Current"},
        timeout=600,
    )
    resp.raise_for_status()

    out = {}
    for line in resp.text.strip().split("\n"):
        parts = re.findall(r'"([^"]*)"', line)
        if not parts:
            continue
        try:
            idx = int(parts[0])
        except ValueError:
            continue
        if idx >= len(batch_keys):
            continue
        status = parts[2] if len(parts) > 2 else ""
        if status in ("Match", "Tie") and len(parts) >= 6:
            try:
                lon_str, lat_str = parts[5].split(",")
                out[batch_keys[idx]] = (float(lat_str), float(lon_str))
                continue
            except ValueError:
                pass
        out[batch_keys[idx]] = (np.nan, np.nan)
    return out


def geocode_all(addresses):
    cache = load_cache(CACHE_CSV)
    todo = [k for k in addresses if k not in cache]
    print(f"Addresses: {len(addresses)} total | {len(cache)} cached | {len(todo)} to geocode")

    for start in range(0, len(todo), BATCH_SIZE):
        batch = todo[start:start + BATCH_SIZE]
        n = start // BATCH_SIZE + 1
        print(f"  Batch {n}: {len(batch)} addresses...")
        try:
            cache.update(geocode_batch(batch, addresses))
        except Exception as e:
            print(f"    failed: {e}")
            continue
        save_cache(cache, CACHE_CSV)

    matched = sum(1 for v in cache.values() if not pd.isna(v[0]))
    print(f"Geocoded: {len(cache)} | matched: {matched} ({matched/max(len(cache),1):.1%})")
    return cache


def haversine_km(lat1, lon1, lat2, lon2):
    if any(pd.isna(v) for v in (lat1, lon1, lat2, lon2)):
        return np.nan
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, (float(lat1), float(lon1), float(lat2), float(lon2)))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return R * 2 * atan2(sqrt(a), sqrt(1 - a))


def attach_and_compute(df, cache):
    def lookup(addr, city, zipc):
        if pd.isna(addr) or not str(addr).strip() or str(addr).lower() == "nan":
            return (np.nan, np.nan)
        key = f"{str(addr).strip()}|{'' if pd.isna(city) else str(city).strip()}|{clean_zip(zipc)}"
        return cache.get(key, (np.nan, np.nan))

    for y in YEARS:
        for prefix, addr_col, city_col, zip_col in [
            ("school", f"School Address_{y}", f"City_{y}", f"Zip_{y}"),
            ("home", f"Address_{y}", f"City_{y}.1", f"Zip_{y}.1"),
        ]:
            if addr_col not in df.columns:
                df[f"{prefix}_lat_{y}"] = np.nan
                df[f"{prefix}_lon_{y}"] = np.nan
                continue
            latlon = df.apply(
                lambda r: lookup(r[addr_col], r[city_col], r[zip_col]), axis=1
            )
            df[f"{prefix}_lat_{y}"] = [v[0] for v in latlon]
            df[f"{prefix}_lon_{y}"] = [v[1] for v in latlon]

        df[f"dist_km_{y}"] = df.apply(
            lambda r: haversine_km(
                r[f"home_lat_{y}"], r[f"home_lon_{y}"],
                r[f"school_lat_{y}"], r[f"school_lon_{y}"],
            ),
            axis=1,
        )
    return df


def main():
    print(f"Loading {RAW_XLSX}")
    df = pd.read_excel(RAW_XLSX)
    print(f"  shape: {df.shape}")

    addresses = collect_addresses(df)
    print(f"Unique addresses: {len(addresses)}")

    cache = geocode_all(addresses)
    df = attach_and_compute(df, cache)

    print("\nDistance QC:")
    for y in YEARS:
        s = df[f"dist_km_{y}"]
        print(f"  {y}  missing={s.isna().mean():.1%}  median={s.median():.2f}km  max={s.max():.1f}km")

    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved {OUT_CSV}")


if __name__ == "__main__":
    main()
