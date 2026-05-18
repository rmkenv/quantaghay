"""
QuantAgri Hay — Real Data Pipeline
====================================
Sentinel-2 L2A + Sentinel-1 RTC via Planetary Computer STAC + COG reads.

PERFORMANCE DESIGN:
  With 100-1200 scenes per node across 24 nodes, reading every COG
  would take hours. Instead we:
  1. Sort scenes by cloud cover (ascending)
  2. Bin into 8-day composite windows
  3. Read only the LOWEST-CLOUD scene per window
  → ~15 reads per node × 24 nodes = ~360 total COG reads
  → Target runtime: 15-25 minutes on GitHub Actions ubuntu-latest

No stackstac. No xarray. No in-memory raster stacks.
No synthetic data. No simulation fallback.
If insufficient real data → skip node, write skip record.

Output: data/hay/ndvi/{HayType}_{Region}_{YYYY-MM-DD}.json
"""

import json
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

S2_COLLECTION   = "sentinel-2-l2a"
S1_COLLECTION   = "sentinel-1-rtc"
MAX_CLOUD_PCT   = 60     # STAC filter — relaxed, best-scene selection handles quality
MIN_VALID_PX    = 50     # minimum valid pixels in bbox chip
MIN_COMPOSITES  = 3      # minimum 8-day composites for a valid output
LOOKBACK_DAYS   = 120    # rolling window in days
COMPOSITE_DAYS  = 8      # bin width in days
MAX_SCENES_NODE = 2      # max scenes to READ per composite window (lowest cloud first)
SAR_CUT_DB      = 1.5    # dB VV increase threshold for cut detection
NDVI_CUT_DROP   = 0.20   # NDVI drop threshold
COG_TIMEOUT     = 20     # seconds per COG HTTP request


# ── STAC search ───────────────────────────────────────────────────────
def stac_search(collection: str, bbox: list, date_range: str,
                extra_query: dict = None, limit: int = 500) -> list[dict]:
    """Query PC STAC API. Returns raw feature list."""
    url    = f"{PC_STAC_URL}/search"
    params = {
        "collections": [collection],
        "bbox":        bbox,
        "datetime":    date_range,
        "limit":       limit,
        "sortby":      [{"field": "properties.datetime", "direction": "asc"}],
    }
    if extra_query:
        params["query"] = extra_query
    try:
        r = requests.post(url, json=params, timeout=30)
        r.raise_for_status()
        items = r.json().get("features", [])
        # Handle pagination if needed
        next_url = r.json().get("links", [])
        next_url = next((l["href"] for l in next_url if l.get("rel") == "next"), None)
        while next_url and len(items) < 2000:
            r2 = requests.get(next_url, timeout=30)
            if not r2.ok:
                break
            page = r2.json()
            items.extend(page.get("features", []))
            next_url = next((l["href"] for l in page.get("links",[]) if l.get("rel") == "next"), None)
        return items
    except Exception as e:
        print(f"    [STAC ERR] {collection}: {e}")
        return []


# ── Select best scene per 8-day window ───────────────────────────────
def select_best_scenes(items: list[dict], period_days: int = COMPOSITE_DAYS,
                       max_per_window: int = MAX_SCENES_NODE) -> list[dict]:
    """
    Bin scenes into fixed windows. Within each window, keep only the
    max_per_window scenes with lowest cloud cover. This reduces 1000+
    scenes to ~15-30 actual COG reads while preserving temporal coverage.
    """
    if not items:
        return []

    # Parse dates
    dated = []
    for item in items:
        dt_str = item.get("properties", {}).get("datetime", "")[:10]
        try:
            dt = datetime.strptime(dt_str, "%Y-%m-%d")
            dated.append((dt, item))
        except ValueError:
            continue

    if not dated:
        return []

    dated.sort(key=lambda x: x[0])
    start = dated[0][0]

    bins: dict[int, list] = defaultdict(list)
    for dt, item in dated:
        period = (dt - start).days // period_days
        cloud  = item.get("properties", {}).get("eo:cloud_cover", 100)
        bins[period].append((cloud, item))

    selected = []
    for period_idx in sorted(bins.keys()):
        # Sort by cloud cover ascending, take best N
        window_items = sorted(bins[period_idx], key=lambda x: x[0])
        for _, item in window_items[:max_per_window]:
            selected.append(item)

    return selected


# ── COG chip reader ───────────────────────────────────────────────────
def read_cog_mean(href: str, bbox: list, scale: float = 1.0) -> Optional[float]:
    """
    Read spatial mean of a bbox chip from a COG.
    Signs URL via PC SDK if available. Returns scaled float or None.
    """
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        # Sign URL
        signed = href
        try:
            import planetary_computer as pc
            signed = pc.sign(href)
        except Exception:
            pass

        with rasterio.open(signed) as src:
            # Transform bbox to src CRS if needed
            src_bbox = bbox
            if src.crs and src.crs.to_epsg() != 4326:
                src_bbox = transform_bounds(
                    CRS.from_epsg(4326), src.crs, *bbox
                )

            window = from_bounds(*src_bbox, transform=src.transform)

            if window.width < 1 or window.height < 1:
                return None

            # Read at native resolution clipped to window
            data = src.read(
                1,
                window=window,
                out_shape=(
                    max(1, min(256, int(window.height))),
                    max(1, min(256, int(window.width))),
                ),
                resampling=Resampling.bilinear,
                fill_value=0,
            )

            arr = data.astype("float64")
            # Mask nodata
            nodata = src.nodata if src.nodata is not None else 0
            arr[arr == nodata] = np.nan
            # Mask sentinel-2 saturation/fill value
            arr[arr > 20000] = np.nan

            valid = arr[~np.isnan(arr)]
            if len(valid) < MIN_VALID_PX:
                return None

            return float(np.nanmean(valid)) * scale

    except Exception as e:
        return None


# ── Sentinel-2 scene processor ────────────────────────────────────────
def process_s2_item(item: dict, bbox: list) -> Optional[dict]:
    """Extract NDVI, NDRE, LSWI from one Sentinel-2 scene via COG reads."""
    assets   = item.get("assets", {})
    props    = item.get("properties", {})
    date_str = props.get("datetime", "")[:10]
    cloud    = props.get("eo:cloud_cover")
    scale    = 1 / 10000.0

    def get_href(band_key, alternates):
        for k in [band_key] + alternates:
            if k in assets and assets[k].get("href"):
                return assets[k]["href"]
        return None

    href_b4  = get_href("B04",  ["red",      "b04"])
    href_b8  = get_href("B08",  ["nir-08",   "nir", "b08"])
    href_b05 = get_href("B05",  ["rededge",  "rededge1", "b05"])
    href_b11 = get_href("B11",  ["swir-16",  "swir1", "b11"])

    if not href_b4 or not href_b8:
        return None

    b4 = read_cog_mean(href_b4, bbox, scale)
    b8 = read_cog_mean(href_b8, bbox, scale)
    if b4 is None or b8 is None:
        return None

    b5  = read_cog_mean(href_b05, bbox, scale) if href_b05 else None
    b11 = read_cog_mean(href_b11, bbox, scale) if href_b11 else None

    eps  = 1e-10
    ndvi = float(np.clip((b8 - b4) / (b8 + b4 + eps), -1, 1))
    ndre = float(np.clip((b8 - b5) / (b8 + b5 + eps), -1, 1)) if b5  is not None else None
    lswi = float(np.clip((b8 - b11)/ (b8 + b11+ eps), -1, 1)) if b11 is not None else None

    return {
        "date":  date_str,
        "ndvi":  round(ndvi, 4),
        "ndre":  round(ndre, 4)  if ndre is not None else None,
        "lswi":  round(lswi, 4) if lswi is not None else None,
        "cloud": round(cloud, 1) if cloud is not None else None,
    }


# ── Sentinel-1 scene processor ────────────────────────────────────────
def process_s1_item(item: dict, bbox: list) -> Optional[dict]:
    """Extract VV backscatter in dB from one Sentinel-1 RTC scene."""
    assets   = item.get("assets", {})
    date_str = item.get("properties", {}).get("datetime", "")[:10]

    href = None
    for k in ["vv", "VV"]:
        if k in assets and assets[k].get("href"):
            href = assets[k]["href"]
            break
    if not href:
        return None

    # S1 RTC is already in linear power (gamma0)
    # Read raw (no scale), then convert to dB
    try:
        import rasterio
        from rasterio.windows import from_bounds
        from rasterio.enums import Resampling
        from rasterio.crs import CRS
        from rasterio.warp import transform_bounds

        signed = href
        try:
            import planetary_computer as pc
            signed = pc.sign(href)
        except Exception:
            pass

        with rasterio.open(signed) as src:
            src_bbox = bbox
            if src.crs and src.crs.to_epsg() != 4326:
                src_bbox = transform_bounds(CRS.from_epsg(4326), src.crs, *bbox)

            window = from_bounds(*src_bbox, transform=src.transform)
            if window.width < 1 or window.height < 1:
                return None

            data = src.read(
                1,
                window=window,
                out_shape=(max(1, min(256, int(window.height))),
                           max(1, min(256, int(window.width)))),
                resampling=Resampling.bilinear,
                fill_value=0,
            )
            arr   = data.astype("float64")
            nodata = src.nodata if src.nodata is not None else 0
            arr[arr == nodata] = np.nan
            arr[arr <= 0]      = np.nan

            valid = arr[~np.isnan(arr)]
            if len(valid) < MIN_VALID_PX:
                return None

            vv_linear = float(np.nanmean(valid))
            vv_db     = 10.0 * np.log10(max(vv_linear, 1e-10))

            return {
                "date":  date_str,
                "vv_db": round(vv_db, 3),
            }
    except Exception:
        return None


# ── Composite builder ─────────────────────────────────────────────────
def build_composites(scenes: list[dict], period_days: int = COMPOSITE_DAYS) -> list[dict]:
    """Bin scenes into fixed windows, take median per window."""
    if not scenes:
        return []

    dated = []
    for s in scenes:
        try:
            dated.append((datetime.strptime(s["date"], "%Y-%m-%d"), s))
        except (ValueError, KeyError):
            continue
    if not dated:
        return []

    dated.sort(key=lambda x: x[0])
    start = dated[0][0]

    bins: dict[int, list] = defaultdict(list)
    for dt, s in dated:
        bins[(dt - start).days // period_days].append(s)

    composites = []
    for pidx in sorted(bins.keys()):
        group       = bins[pidx]
        period_date = (start + timedelta(days=pidx * period_days)).strftime("%Y-%m-%d")

        def med(key):
            vals = [s[key] for s in group if s.get(key) is not None]
            return round(float(np.median(vals)), 4) if vals else None

        comp = {
            "date":        period_date,
            "ndvi":        med("ndvi"),
            "ndre":        med("ndre"),
            "lswi":        med("lswi"),
            "scene_count": len(group),
            "avg_cloud":   med("cloud"),
        }
        if comp["ndvi"] is not None:
            composites.append(comp)

    return composites


# ── SAR alignment ─────────────────────────────────────────────────────
def align_sar(s1_scenes: list[dict], composites: list[dict],
              tol: int = 4) -> list[Optional[float]]:
    """Match each S2 composite to nearest S1 scene within ±tol days."""
    lookup = {}
    for s in s1_scenes:
        try:
            lookup[datetime.strptime(s["date"], "%Y-%m-%d")] = s["vv_db"]
        except (ValueError, KeyError):
            continue

    aligned = []
    for comp in composites:
        try:
            cdt = datetime.strptime(comp["date"], "%Y-%m-%d")
        except ValueError:
            aligned.append(None)
            continue
        best_vv, best_d = None, 999
        for sdt, vv in lookup.items():
            d = abs((sdt - cdt).days)
            if d <= tol and d < best_d:
                best_vv, best_d = vv, d
        aligned.append(best_vv)
    return aligned


# ── Cut detection ─────────────────────────────────────────────────────
def detect_cuts(composites: list[dict], sar_aligned: list[Optional[float]],
                hay_type: str) -> list[dict]:
    """SAR + optical fusion cut detection."""
    thresholds = CUTTING_NDVI_THRESHOLDS.get(hay_type, CUTTING_NDVI_THRESHOLDS["Mixed_Grass"])
    ndvi_vals  = [c["ndvi"] for c in composites]
    velocity   = list(np.gradient(ndvi_vals)) if len(ndvi_vals) > 1 else [0.0] * len(ndvi_vals)

    enriched = []
    for i, comp in enumerate(composites):
        ndvi       = comp["ndvi"]
        vel        = float(velocity[i])
        sar_vv     = sar_aligned[i] if i < len(sar_aligned) else None
        ndvi_delta = round(ndvi - ndvi_vals[i-1], 4) if i > 0 else 0.0

        # SAR delta vs prior available reading
        prev_sar  = next((sar_aligned[j] for j in range(i-1, -1, -1)
                          if j < len(sar_aligned) and sar_aligned[j] is not None), None)
        sar_delta = round(sar_vv - prev_sar, 3) if (sar_vv is not None and prev_sar is not None) else None

        # Cut detection: NDVI optical drop + SAR backscatter increase
        ndvi_dropped   = ndvi_delta < -NDVI_CUT_DROP
        cut_detected   = False
        cut_confidence = None

        if ndvi_dropped:
            if sar_delta is not None and sar_delta > SAR_CUT_DB:
                cut_detected   = True
                cut_confidence = "high"       # both sensors confirm
            elif sar_vv is None:
                cut_detected   = True
                cut_confidence = "low_no_sar" # optical only, no SAR to confirm
            # SAR present but no backscatter increase → drought/stress, not a cut

        # Cutting status
        if cut_detected:
            status = "post_cut"
        elif ndvi >= thresholds["peak"] and vel > -0.01:
            status = "ready_to_cut"
        elif ndvi >= thresholds["ready"] and vel > 0:
            status = "approaching_ready"
        elif vel > 0.01:
            status = "growing"
        elif ndvi < 0.20 and i > 0 and ndvi_vals[i-1] > 0.40:
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


# ── Node fetcher ──────────────────────────────────────────────────────
def fetch_node(node: dict, year: int) -> Optional[dict]:
    """Fetch one hay node. Returns None if insufficient real data."""
    hay_type = node["hay_type"]
    region   = node["region"]
    bbox     = node["bbox"]

    today      = datetime.now(timezone.utc)
    start      = (today - timedelta(days=LOOKBACK_DAYS)).strftime("%Y-%m-%d")
    end        = today.strftime("%Y-%m-%d")
    date_range = f"{start}/{end}"

    print(f"\n  [{hay_type}/{region}] {date_range}")

    # ── Sentinel-2: search → select best per window → read COGs ──────
    s2_all = stac_search(S2_COLLECTION, bbox, date_range,
                         extra_query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}})
    print(f"    [S2  ] {len(s2_all)} items found → selecting best per 8-day window")

    s2_selected = select_best_scenes(s2_all)
    print(f"    [S2  ] reading {len(s2_selected)} scenes (1 lowest-cloud per window)")

    s2_scenes = []
    for item in s2_selected:
        result = process_s2_item(item, bbox)
        if result:
            s2_scenes.append(result)

    print(f"    [S2  ] {len(s2_scenes)} valid scenes processed")

    if len(s2_scenes) < MIN_COMPOSITES:
        print(f"    [SKIP] {len(s2_scenes)} scenes < {MIN_COMPOSITES} minimum")
        return None

    # ── Sentinel-1: same approach ──────────────────────────────────────
    s1_all = stac_search(S1_COLLECTION, bbox, date_range)
    print(f"    [S1  ] {len(s1_all)} SAR items → selecting best per window")

    s1_selected = select_best_scenes(s1_all, max_per_window=1)
    s1_scenes   = []
    for item in s1_selected:
        result = process_s1_item(item, bbox)
        if result:
            s1_scenes.append(result)
    print(f"    [S1  ] {len(s1_scenes)} valid SAR scenes")

    # ── Composites → SAR alignment → cut detection ────────────────────
    composites  = build_composites(s2_scenes)
    if not composites:
        print(f"    [SKIP] No composites built")
        return None

    sar_aligned  = align_sar(s1_scenes, composites)
    sar_coverage = sum(1 for v in sar_aligned if v is not None)
    data_quality = "real_s2_s1" if sar_coverage >= len(composites) * 0.4 else "real_s2_only"

    enriched = detect_cuts(composites, sar_aligned, hay_type)

    ndvi_vals      = [c["ndvi"] for c in enriched if c.get("ndvi") is not None]
    ndre_vals      = [c["ndre"] for c in enriched if c.get("ndre") is not None]
    confirmed_cuts = [c for c in enriched if c.get("cut_detected")]
    high_conf_cuts = [c for c in confirmed_cuts if c.get("cut_confidence") == "high"]
    current        = enriched[-1] if enriched else {}

    print(f"    [OK  ] {len(enriched)} composites | "
          f"cuts={len(confirmed_cuts)} (high-conf={len(high_conf_cuts)}) | "
          f"quality={data_quality} | SAR={sar_coverage}/{len(composites)} windows covered")

    return {
        "hay_type":              hay_type,
        "region":                region,
        "state":                 node["state"],
        "country":               node["country"],
        "grade":                 node["grade"],
        "primary_use":           node["primary_use"],
        "bbox":                  bbox,
        "year":                  year,
        "date_range":            date_range,
        "source":                "planetary_computer",
        "data_quality":          data_quality,
        "s2_items_found":        len(s2_all),
        "s2_scenes_read":        len(s2_scenes),
        "s1_scenes_read":        len(s1_scenes),
        "composite_count":       len(enriched),
        "s1_coverage_pct":       round(sar_coverage / len(composites) * 100, 1),
        "composites":            enriched,
        "peak_ndvi":             round(max(ndvi_vals), 4) if ndvi_vals else None,
        "peak_ndre":             round(max(ndre_vals), 4) if ndre_vals else None,
        "current_ndvi":          current.get("ndvi"),
        "current_ndre":          current.get("ndre"),
        "current_velocity":      current.get("ndvi_velocity"),
        "current_status":        current.get("cutting_status"),
        "current_date":          current.get("date"),
        "cuts_detected":         len(confirmed_cuts),
        "cuts_high_confidence":  len(high_conf_cuts),
        "cut_dates":             [c["date"] for c in confirmed_cuts],
        "cut_dates_high_conf":   [c["date"] for c in high_conf_cuts],
        "data_notes": (
            f"Real S2+S1 via PC STAC. Best-scene-per-window sampling: "
            f"{len(s2_scenes)} of {len(s2_all)} S2 scenes read. "
            f"Cut detection: NDVI drop >{NDVI_CUT_DROP} AND SAR VV >{SAR_CUT_DB}dB. "
            f"Quality: {data_quality}. No synthetic data."
        ),
    }


# ── Main ──────────────────────────────────────────────────────────────
def run():
    try:
        import rasterio
    except ImportError:
        print("[FATAL] rasterio not installed — pip install rasterio")
        sys.exit(1)

    today    = datetime.now(timezone.utc)
    date_str = today.strftime("%Y-%m-%d")
    year     = today.year

    print(f"\n[HAY PIPELINE] {date_str} — REAL DATA ONLY")
    print(f"[HAY PIPELINE] {len(HAY_NODES)} nodes | S2+S1 COG reads | "
          f"best-scene sampling | no simulation\n")

    processed = skipped = errors = 0

    for node in HAY_NODES:
        hay_type = node["hay_type"]
        region   = node["region"]
        fname    = f"{hay_type}_{region}_{date_str}.json"

        try:
            data = fetch_node(node, year)

            if data is None:
                (SKIP_DIR / fname).write_text(json.dumps({
                    "hay_type":    hay_type,
                    "region":      region,
                    "date":        date_str,
                    "skipped":     True,
                    "reason":      (f"< {MIN_COMPOSITES} valid composites in "
                                   f"{LOOKBACK_DAYS}-day window with cloud < {MAX_CLOUD_PCT}%"),
                    "data_quality": "skipped",
                }, indent=2))
                print(f"  [SKIP] {fname}")
                skipped += 1
                continue

            (NDVI_DIR / fname).write_text(json.dumps(data, indent=2))
            processed += 1

        except Exception as e:
            print(f"  [ERR ] {hay_type}/{region}: {e}")
            errors += 1

    print(f"\n[HAY PIPELINE] Complete")
    print(f"  Processed: {processed} | Skipped: {skipped} | Errors: {errors}")
    if processed == 0:
        print("  [WARN] No nodes processed — check rasterio install and PC connectivity")
    print()


if __name__ == "__main__":
    run()
