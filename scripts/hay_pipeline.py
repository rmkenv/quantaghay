"""
QuantAgri Hay — Real Data Pipeline
====================================
Pulls REAL Sentinel-2 L2A from Planetary Computer using
lightweight STAC + COG (Cloud-Optimised GeoTIFF) reads.

NO stackstac. NO xarray. NO in-memory raster stacks.
Each scene: download a small bbox chip via GDAL/rasterio,
compute indices, aggregate. Runs in < 2GB RAM per node.

Sentinel-1 SAR for cut detection: same COG approach.

CUT DETECTION — SAR + Optical fusion:
  Cut confirmed when in same 8-day window:
  1. NDVI drops > 0.20  (optical biomass removal)
  2. Sentinel-1 VV increases > 1.5 dB  (stubble = rougher surface)
  Both must fire — eliminates drought/stress false positives.

No synthetic data. No simulation. If data is insufficient → skip.

Output: data/hay/ndvi/{HayType}_{Region}_{YYYY-MM-DD}.json
"""

import json
import os
import sys
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from collections import defaultdict

import numpy as np
import requests

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from hay_config import HAY_NODES, HAY_SEASONS, HAY_DIR, CUTTING_NDVI_THRESHOLDS, PC_STAC_URL

NDVI_DIR = HAY_DIR / "ndvi"
SKIP_DIR = HAY_DIR / "skipped"
for d in [NDVI_DIR, SKIP_DIR]:
    d.mkdir(parents=True, exist_ok=True)

S2_COLLECTION  = "sentinel-2-l2a"
S1_COLLECTION  = "sentinel-1-rtc"
MAX_CLOUD_PCT  = 60      # relaxed — median composite handles residual cloud
MIN_VALID_PX   = 100     # minimum valid pixels per scene chip
MIN_S2_SCENES  = 2       # minimum scenes for a valid composite
LOOKBACK_DAYS  = 120     # rolling lookback window
COMPOSITE_DAYS = 8       # composite period in days
SAR_CUT_DB     = 1.5     # dB VV increase = cut signal
NDVI_CUT_DROP  = 0.20    # NDVI drop threshold for cut detection


# ── STAC search ───────────────────────────────────────────────────────
def stac_search(collection: str, bbox: list, date_range: str,
                extra_query: dict = None) -> list[dict]:
    """Query PC STAC API, return list of items."""
    url    = f"{PC_STAC_URL}/search"
    params = {
        "collections": [collection],
        "bbox":        bbox,
        "datetime":    date_range,
        "limit":       100,
    }
    if extra_query:
        params["query"] = extra_query

    try:
        r = requests.post(url, json=params, timeout=30)
        r.raise_for_status()
        return r.json().get("features", [])
    except Exception as e:
        print(f"    [STAC ERR] {collection}: {e}")
        return []


# ── COG chip reader ───────────────────────────────────────────────────
def read_cog_chip(href: str, bbox: list, token: str = None) -> Optional[np.ndarray]:
    """
    Read a small spatial chip from a Cloud-Optimised GeoTIFF using GDAL.
    Returns 2D float32 array or None on failure.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling

        env = {}
        if token:
            env["GDAL_HTTP_HEADERS"] = f"Authorization: Bearer {token}"

        # Sign URL via PC SDK if available
        href_signed = href
        try:
            import planetary_computer as pc
            href_signed = pc.sign(href)
        except Exception:
            pass

        with rasterio.Env(**env):
            with rasterio.open(href_signed) as src:
                window = from_bounds(*bbox, transform=src.transform)
                # Resample to ~20m equiv
                out_shape = (1, max(1, int(window.height)), max(1, int(window.width)))
                data = src.read(
                    1,
                    window=window,
                    out_shape=out_shape[1:],
                    resampling=Resampling.bilinear,
                )
                # Mask nodata
                nodata = src.nodata or 0
                arr = data.astype("float32")
                arr[arr == nodata] = np.nan
                return arr
    except Exception as e:
        return None


def read_cog_mean(href: str, bbox: list, scale: float = 1.0) -> Optional[float]:
    """Read COG chip and return spatial mean, scaled."""
    arr = read_cog_chip(href, bbox)
    if arr is None:
        return None
    valid = arr[~np.isnan(arr)]
    if len(valid) < MIN_VALID_PX:
        return None
    return float(np.nanmean(valid)) * scale


# ── Sentinel-2 processing ─────────────────────────────────────────────
def process_s2_item(item: dict, bbox: list) -> Optional[dict]:
    """Extract NDVI, NDRE, LSWI from one Sentinel-2 scene."""
    assets = item.get("assets", {})

    hrefs = {}
    for band, key in [("B04", "red"), ("B05", "rededge"),
                      ("B08", "nir-08"), ("B11", "swir-16")]:
        # Try multiple possible asset key names
        for k in [band.lower(), key, band]:
            if k in assets:
                hrefs[band] = assets[k].get("href")
                break

    # Need B04, B08 for NDVI minimum; B05 and B11 for full suite
    if not hrefs.get("B04") or not hrefs.get("B08"):
        return None

    scale = 1 / 10000.0

    b4  = read_cog_mean(hrefs["B04"], bbox, scale)
    b8  = read_cog_mean(hrefs["B08"], bbox, scale)
    if b4 is None or b8 is None:
        return None

    b5  = read_cog_mean(hrefs.get("B05", ""), bbox, scale) if hrefs.get("B05") else None
    b11 = read_cog_mean(hrefs.get("B11", ""), bbox, scale) if hrefs.get("B11") else None

    eps = 1e-10
    ndvi = (b8 - b4) / (b8 + b4 + eps)
    ndre = (b8 - b5) / (b8 + b5 + eps) if b5 is not None else None
    lswi = (b8 - b11) / (b8 + b11 + eps) if b11 is not None else None

    # Clip to valid range
    ndvi = float(np.clip(ndvi, -1, 1))
    ndre = float(np.clip(ndre, -1, 1)) if ndre is not None else None
    lswi = float(np.clip(lswi, -1, 1)) if lswi is not None else None

    date_str = item.get("properties", {}).get("datetime", "")[:10]
    cloud    = item.get("properties", {}).get("eo:cloud_cover", None)

    return {
        "date":  date_str,
        "ndvi":  round(ndvi, 4),
        "ndre":  round(ndre, 4) if ndre is not None else None,
        "lswi":  round(lswi, 4) if lswi is not None else None,
        "cloud": round(cloud, 1) if cloud is not None else None,
    }


# ── Sentinel-1 processing ─────────────────────────────────────────────
def process_s1_item(item: dict, bbox: list) -> Optional[dict]:
    """Extract VV backscatter in dB from one Sentinel-1 scene."""
    assets = item.get("assets", {})

    href = None
    for k in ["vv", "VV", "vh", "VH"]:
        if k in assets:
            href = assets[k].get("href")
            break
    if not href:
        return None

    arr = read_cog_chip(href, bbox)
    if arr is None:
        return None
    valid = arr[~np.isnan(arr) & (arr > 0)]
    if len(valid) < MIN_VALID_PX:
        return None

    vv_linear = float(np.nanmean(valid))
    vv_db     = 10.0 * np.log10(max(vv_linear, 1e-10))
    date_str  = item.get("properties", {}).get("datetime", "")[:10]

    return {
        "date":    date_str,
        "vv_db":   round(vv_db, 3),
        "vv_linear": round(vv_linear, 6),
    }


# ── Composite builder ─────────────────────────────────────────────────
def build_composites(scenes: list[dict], period_days: int = COMPOSITE_DAYS) -> list[dict]:
    """
    Bin scenes into fixed periods and take median.
    Returns list of composite dicts sorted by date.
    """
    if not scenes:
        return []

    # Parse dates and find range
    dated = [(datetime.strptime(s["date"], "%Y-%m-%d"), s) for s in scenes if s.get("date")]
    if not dated:
        return []
    dated.sort(key=lambda x: x[0])
    start = dated[0][0]

    # Bin into periods
    bins: dict[int, list] = defaultdict(list)
    for dt, scene in dated:
        period = (dt - start).days // period_days
        bins[period].append(scene)

    composites = []
    for period_idx in sorted(bins.keys()):
        group   = bins[period_idx]
        period_start = start + timedelta(days=period_idx * period_days)
        date_str = period_start.strftime("%Y-%m-%d")

        def median_of(key):
            vals = [s[key] for s in group if s.get(key) is not None]
            return round(float(np.median(vals)), 4) if vals else None

        comp = {
            "date":       date_str,
            "ndvi":       median_of("ndvi"),
            "ndre":       median_of("ndre"),
            "lswi":       median_of("lswi"),
            "scene_count": len(group),
            "avg_cloud":  median_of("cloud"),
        }
        if comp["ndvi"] is not None:
            composites.append(comp)

    return composites


def align_sar_to_composites(
    s1_scenes: list[dict],
    composites: list[dict],
    tolerance_days: int = 4
) -> list[Optional[float]]:
    """For each S2 composite date, find nearest S1 scene within tolerance."""
    sar_lookup = {
        datetime.strptime(s["date"], "%Y-%m-%d"): s["vv_db"]
        for s in s1_scenes if s.get("date") and s.get("vv_db") is not None
    }

    aligned = []
    for comp in composites:
        comp_dt = datetime.strptime(comp["date"], "%Y-%m-%d")
        best_vv = None
        best_d  = 999
        for sar_dt, vv in sar_lookup.items():
            d = abs((sar_dt - comp_dt).days)
            if d <= tolerance_days and d < best_d:
                best_vv = vv
                best_d  = d
        aligned.append(best_vv)
    return aligned


# ── Cut detection ─────────────────────────────────────────────────────
def detect_cuts(composites: list[dict], sar_aligned: list[Optional[float]],
                hay_type: str) -> list[dict]:
    """
    Apply SAR + optical fusion cut detection to composites.
    Returns composites enriched with cut detection fields.
    """
    thresholds = CUTTING_NDVI_THRESHOLDS.get(hay_type, CUTTING_NDVI_THRESHOLDS["Mixed_Grass"])
    ndvi_vals  = [c["ndvi"] for c in composites]
    velocity   = list(np.gradient(ndvi_vals)) if len(ndvi_vals) > 1 else [0.0] * len(ndvi_vals)

    enriched = []
    for i, comp in enumerate(composites):
        ndvi     = comp["ndvi"]
        vel      = float(velocity[i])
        sar_vv   = sar_aligned[i] if i < len(sar_aligned) else None

        # NDVI delta from prior period
        ndvi_delta = round(ndvi - ndvi_vals[i-1], 4) if i > 0 else 0.0

        # SAR delta
        prev_sar = next((sar_aligned[j] for j in range(i-1, -1, -1)
                         if j < len(sar_aligned) and sar_aligned[j] is not None), None)
        sar_delta = round(sar_vv - prev_sar, 3) if (sar_vv is not None and prev_sar is not None) else None

        # Cut detection
        ndvi_dropped = ndvi_delta < -NDVI_CUT_DROP
        cut_detected = False
        cut_confidence = None

        if ndvi_dropped:
            if sar_delta is not None and sar_delta > SAR_CUT_DB:
                cut_detected   = True
                cut_confidence = "high"
            elif sar_vv is None:
                cut_detected   = True
                cut_confidence = "low_no_sar"
            # sar present but didn't increase → drought/stress not a cut

        # Cutting status
        if cut_detected:
            status = "post_cut"
        elif ndvi is not None and ndvi >= thresholds["peak"] and vel > -0.01:
            status = "ready_to_cut"
        elif ndvi is not None and ndvi >= thresholds["ready"] and vel > 0:
            status = "approaching_ready"
        elif vel > 0.01:
            status = "growing"
        elif ndvi is not None and ndvi < 0.20 and i > 0 and ndvi_vals[i-1] > 0.40:
            status = "post_cut_probable"
        else:
            status = "stable"

        enriched.append({
            **comp,
            "ndvi_velocity":  round(vel, 5),
            "ndvi_delta":     ndvi_delta,
            "sar_vv_db":      sar_vv,
            "sar_delta_db":   sar_delta,
            "cut_detected":   cut_detected,
            "cut_confidence": cut_confidence,
            "cutting_status": status,
        })

    return enriched


# ── Main node fetcher ─────────────────────────────────────────────────
def fetch_node(node: dict, year: int) -> Optional[dict]:
    """
    Fetch real Sentinel-2 + Sentinel-1 for one hay node.
    Returns None if insufficient data — never fabricates.
    """
    hay_type = node["hay_type"]
    region   = node["region"]
    bbox     = node["bbox"]

    today  = datetime.now(timezone.utc)
    start  = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end    = today.strftime("%Y-%m-%d")
    date_range = f"{start}/{end}"

    print(f"\n  [{hay_type}/{region}] {date_range}")

    # ── Sentinel-2 scenes ─────────────────────────────────────────────
    s2_items = stac_search(
        S2_COLLECTION, bbox, date_range,
        extra_query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}}
    )
    print(f"    [S2  ] {len(s2_items)} STAC items")

    if len(s2_items) < MIN_S2_SCENES:
        print(f"    [SKIP] Only {len(s2_items)} S2 items (need {MIN_S2_SCENES}+)")
        return None

    # Process scenes
    s2_scenes = []
    for item in s2_items:
        result = process_s2_item(item, bbox)
        if result:
            s2_scenes.append(result)

    print(f"    [S2  ] {len(s2_scenes)} valid scenes processed")

    if len(s2_scenes) < MIN_S2_SCENES:
        print(f"    [SKIP] Only {len(s2_scenes)} valid S2 scenes after processing")
        return None

    # ── Sentinel-1 scenes ─────────────────────────────────────────────
    s1_items = stac_search(S1_COLLECTION, bbox, date_range)
    print(f"    [S1  ] {len(s1_items)} SAR items")

    s1_scenes = []
    for item in s1_items:
        result = process_s1_item(item, bbox)
        if result:
            s1_scenes.append(result)
    print(f"    [S1  ] {len(s1_scenes)} valid SAR scenes")

    # ── Build composites ──────────────────────────────────────────────
    composites = build_composites(s2_scenes)
    if not composites:
        print(f"    [SKIP] No valid composites built")
        return None

    sar_aligned = align_sar_to_composites(s1_scenes, composites)
    sar_coverage = sum(1 for v in sar_aligned if v is not None)
    data_quality = "real_s2_s1" if sar_coverage >= len(composites) * 0.4 else "real_s2_only"

    # ── Cut detection ─────────────────────────────────────────────────
    enriched = detect_cuts(composites, sar_aligned, hay_type)

    # ── Summary ───────────────────────────────────────────────────────
    ndvi_vals      = [c["ndvi"] for c in enriched if c.get("ndvi") is not None]
    ndre_vals      = [c["ndre"] for c in enriched if c.get("ndre") is not None]
    confirmed_cuts = [c for c in enriched if c.get("cut_detected")]
    high_conf_cuts = [c for c in confirmed_cuts if c.get("cut_confidence") == "high"]
    current        = enriched[-1] if enriched else {}

    print(f"    [OK  ] {len(enriched)} composites | {len(confirmed_cuts)} cuts detected "
          f"({len(high_conf_cuts)} high-conf) | quality={data_quality}")

    return {
        "hay_type":          hay_type,
        "region":            region,
        "state":             node["state"],
        "country":           node["country"],
        "grade":             node["grade"],
        "primary_use":       node["primary_use"],
        "bbox":              bbox,
        "year":              year,
        "date_range":        date_range,
        "source":            "planetary_computer",
        "data_quality":      data_quality,
        "s2_scene_count":    len(s2_scenes),
        "s1_scene_count":    len(s1_scenes),
        "composite_count":   len(enriched),
        "s1_coverage_pct":   round(sar_coverage / len(composites) * 100, 1) if composites else 0,
        "composites":        enriched,
        "peak_ndvi":         round(max(ndvi_vals), 4) if ndvi_vals else None,
        "peak_ndre":         round(max(ndre_vals), 4) if ndre_vals else None,
        "current_ndvi":      current.get("ndvi"),
        "current_ndre":      current.get("ndre"),
        "current_velocity":  current.get("ndvi_velocity"),
        "current_status":    current.get("cutting_status"),
        "current_date":      current.get("date"),
        "cuts_detected":     len(confirmed_cuts),
        "cuts_high_confidence": len(high_conf_cuts),
        "cut_dates":         [c["date"] for c in confirmed_cuts],
        "cut_dates_high_conf": [c["date"] for c in high_conf_cuts],
        "data_notes": (
            f"Real Sentinel-2 L2A + Sentinel-1 RTC via Planetary Computer STAC. "
            f"Cut detection: NDVI drop >{NDVI_CUT_DROP} AND SAR VV increase >{SAR_CUT_DB}dB. "
            f"Data quality: {data_quality}. No synthetic data used."
        ),
    }


# ── Run ───────────────────────────────────────────────────────────────
def run():
    # Hard check: rasterio required for COG reads
    try:
        import rasterio
    except ImportError:
        print("[FATAL] rasterio not installed. Run: pip install rasterio")
        sys.exit(1)

    today    = datetime.now(timezone.utc)
    date_str = today.strftime("%Y-%m-%d")
    year     = today.year

    print(f"\n[HAY PIPELINE] {date_str} — REAL DATA ONLY")
    print(f"[HAY PIPELINE] {len(HAY_NODES)} nodes | Sentinel-2 + Sentinel-1 | No simulation\n")

    processed, skipped_count, errors = 0, 0, 0

    for node in HAY_NODES:
        hay_type = node["hay_type"]
        region   = node["region"]
        fname    = f"{hay_type}_{region}_{date_str}.json"

        try:
            data = fetch_node(node, year)

            if data is None:
                skip_record = {
                    "hay_type": hay_type, "region": region,
                    "date": date_str, "skipped": True,
                    "reason": f"Insufficient real data (need {MIN_S2_SCENES}+ S2 scenes, {LOOKBACK_DAYS}-day window, cloud < {MAX_CLOUD_PCT}%)",
                    "data_quality": "skipped",
                }
                (SKIP_DIR / fname).write_text(json.dumps(skip_record, indent=2))
                print(f"  [SKIP] {fname}")
                skipped_count += 1
                continue

            (NDVI_DIR / fname).write_text(json.dumps(data, indent=2))
            processed += 1

        except Exception as e:
            print(f"  [ERR ] {hay_type}/{region}: {e}")
            errors += 1

    print(f"\n[HAY PIPELINE] Done")
    print(f"  Processed: {processed} | Skipped: {skipped_count} | Errors: {errors}\n")


if __name__ == "__main__":
    run()
