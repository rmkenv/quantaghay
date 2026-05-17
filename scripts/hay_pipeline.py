"""
QuantAgri Hay — Planetary Computer NDVI Pipeline
=================================================
Pulls Sentinel-2 L2A for all 24 North American hay nodes.
Computes NDVI, NDRE (chlorophyll/protein proxy), NDWI, EVI, LSWI.
Estimates cutting readiness based on NDVI thresholds.

Key difference from commodity crop pipeline:
- Hay is cut multiple times per season (3-8 cuts for alfalfa)
- NDRE (Red Edge) correlates with crude protein content
- Cutting readiness detection: NDVI rise = growing, peak = cut time
- Post-cut regrowth tracking is as important as peak detection

Output: data/hay/ndvi/{hay_type}_{region}_{YYYY-MM-DD}.json
"""

import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent))
from hay_config import (
    HAY_NODES, HAY_SEASONS, HAY_DIR,
    CUTTING_NDVI_THRESHOLDS, PC_STAC_URL, COLLECTION,
    MAX_CLOUD_PCT, RESOLUTION
)

NDVI_DIR = HAY_DIR / "ndvi"

try:
    import planetary_computer as pc
    import pystac_client
    import stackstac
    PC_AVAILABLE = True
except ImportError:
    PC_AVAILABLE = False
    print("[WARN] Planetary Computer not installed — using simulated data")


def cutting_status(ndvi: float, hay_type: str, velocity: float) -> str:
    """
    Estimate cutting readiness from NDVI level and velocity.
    Returns: 'growing' | 'approaching_ready' | 'ready_to_cut' | 'past_peak' | 'post_cut'
    """
    thresholds = CUTTING_NDVI_THRESHOLDS.get(hay_type, CUTTING_NDVI_THRESHOLDS["Mixed_Grass"])
    if ndvi < 0.25:
        return "post_cut"
    elif ndvi < thresholds["past_peak"]:
        return "post_cut" if velocity < 0 else "growing"
    elif ndvi >= thresholds["peak"]:
        return "past_peak" if velocity < -0.01 else "ready_to_cut"
    elif ndvi >= thresholds["ready"]:
        return "approaching_ready" if velocity > 0 else "ready_to_cut"
    else:
        return "growing"


def simulate_node(node: dict, year: int) -> dict:
    """
    Simulate realistic hay NDVI/NDRE time series.
    Hay differs from commodity crops: multiple growth cycles per season.
    """
    hay_type = node["hay_type"]
    region   = node["region"]

    # Hay-specific simulation parameters
    cfg = {
        "Alfalfa":      dict(cycles=5, ndvi_max=0.82, ndre_max=0.45, recovery=21),
        "Timothy":      dict(cycles=2, ndvi_max=0.76, ndre_max=0.38, recovery=35),
        "Orchardgrass": dict(cycles=3, ndvi_max=0.74, ndre_max=0.36, recovery=28),
        "Mixed_Grass":  dict(cycles=3, ndvi_max=0.70, ndre_max=0.33, recovery=30),
        "Bermudagrass": dict(cycles=4, ndvi_max=0.68, ndre_max=0.30, recovery=25),
    }.get(hay_type, dict(cycles=3, ndvi_max=0.72, ndre_max=0.35, recovery=28))

    rng    = np.random.default_rng(seed=abs(hash(region)) % 2**31)
    months = list(range(1, 13))

    # Simulate multi-cut growth pattern
    ndvi_series, ndre_series, lswi_series = [], [], []
    for m in months:
        # Base seasonal curve
        season_f = np.exp(-0.08 * (m - 7) ** 2)
        if m < 3 or m > 10:   # dormant / winter
            ndvi = rng.uniform(0.08, 0.18)
            ndre = rng.uniform(0.05, 0.12)
            lswi = rng.uniform(0.03, 0.10)
        else:
            # Simulate growth-cut-regrowth cycle
            cycle_phase = ((m - 3) * cfg["cycles"] / 7) % 1
            growth_f = np.sin(cycle_phase * np.pi) ** 0.5
            ndvi = float(np.clip(cfg["ndvi_max"] * season_f * growth_f + rng.normal(0, 0.02), 0.10, 0.92))
            ndre = float(np.clip(cfg["ndre_max"] * season_f * growth_f + rng.normal(0, 0.015), 0.05, 0.55))
            lswi = float(np.clip(0.35 * season_f * growth_f + rng.normal(0, 0.018), 0.04, 0.60))

        ndvi_series.append(round(ndvi, 4))
        ndre_series.append(round(ndre, 4))
        lswi_series.append(round(lswi, 4))

    velocity = list(np.gradient(ndvi_series))
    current_ndvi = ndvi_series[-1]
    current_vel  = velocity[-1]

    return dict(
        hay_type       = hay_type,
        region         = region,
        state          = node["state"],
        country        = node["country"],
        grade          = node["grade"],
        primary_use    = node["primary_use"],
        bbox           = node["bbox"],
        year           = year,
        source         = "simulated",
        composites     = [
            dict(
                month    = m,
                ndvi     = ndvi_series[i],
                ndre     = ndre_series[i],
                lswi     = lswi_series[i],
                velocity = round(float(velocity[i]), 5),
                cutting_status = cutting_status(ndvi_series[i], hay_type, float(velocity[i]))
            )
            for i, m in enumerate(months)
        ],
        peak_ndvi          = round(float(max(ndvi_series)), 4),
        peak_ndre          = round(float(max(ndre_series)), 4),
        current_ndvi       = round(current_ndvi, 4),
        current_ndre       = round(ndre_series[-1], 4),
        current_velocity   = round(current_vel, 5),
        current_status     = cutting_status(current_ndvi, hay_type, current_vel),
        estimated_cuts     = cfg["cycles"],
        cloud_cover_pct    = 0.0,
        scene_count        = 0,
    )


def fetch_node(node: dict, year: int) -> dict:
    if not PC_AVAILABLE:
        return simulate_node(node, year)

    hay_type = node["hay_type"]
    region   = node["region"]
    bbox     = node["bbox"]
    m_start, m_end = HAY_SEASONS.get(hay_type, ("04", "10"))
    date_range = f"{year}-{m_start}-01/{year}-{m_end}-30"

    print(f"  [PC ] {hay_type}/{region} · {date_range}")

    try:
        catalog = pystac_client.Client.open(PC_STAC_URL, modifier=pc.sign_inplace)
        items   = catalog.search(
            collections=[COLLECTION],
            bbox=bbox,
            datetime=date_range,
            query={"eo:cloud_cover": {"lt": MAX_CLOUD_PCT}},
        ).item_collection()

        if len(items) == 0:
            return simulate_node(node, year)

        stack = stackstac.stack(
            items,
            assets=["B04", "B05", "B08", "B11"],  # Red, RedEdge, NIR, SWIR
            resolution=RESOLUTION,
            bounds_latlon=bbox,
        )

        eps = 1e-10
        b4  = stack.sel(band="B04").astype("float32") / 10000.0
        b5  = stack.sel(band="B05").astype("float32") / 10000.0  # Red Edge
        b8  = stack.sel(band="B08").astype("float32") / 10000.0
        b11 = stack.sel(band="B11").astype("float32") / 10000.0

        ndvi = ((b8 - b4)  / (b8 + b4  + eps)).clip(-1, 1)
        ndre = ((b8 - b5)  / (b8 + b5  + eps)).clip(-1, 1)
        lswi = ((b8 - b11) / (b8 + b11 + eps)).clip(-1, 1)

        # 8-day composites for hay (tighter than 16-day for multi-cut tracking)
        ndvi_c = ndvi.resample(time="8D").median().mean(dim=["x", "y"])
        ndre_c = ndre.resample(time="8D").median().mean(dim=["x", "y"])
        lswi_c = lswi.resample(time="8D").median().mean(dim=["x", "y"])

        ndvi_vals = [float(v) for v in ndvi_c.values]
        ndre_vals = [float(v) for v in ndre_c.values]
        lswi_vals = [float(v) for v in lswi_c.values]
        times     = [str(t)[:10] for t in ndvi_c.time.values]
        velocity  = list(np.gradient(ndvi_vals))

        avg_cloud = float(
            sum(i.properties.get("eo:cloud_cover", 0) for i in items) / len(items)
        )
        current_ndvi = ndvi_vals[-1] if ndvi_vals else 0
        current_vel  = velocity[-1]  if velocity  else 0

        return dict(
            hay_type     = hay_type,
            region       = region,
            state        = node["state"],
            country      = node["country"],
            grade        = node["grade"],
            primary_use  = node["primary_use"],
            bbox         = bbox,
            year         = year,
            source       = "planetary_computer",
            composites   = [
                dict(
                    date   = t, ndvi = round(n, 4), ndre = round(nr, 4),
                    lswi   = round(l, 4), velocity = round(v, 5),
                    cutting_status = cutting_status(n, hay_type, v)
                )
                for t, n, nr, l, v in zip(times, ndvi_vals, ndre_vals, lswi_vals, velocity)
            ],
            peak_ndvi        = round(float(max(ndvi_vals)), 4),
            peak_ndre        = round(float(max(ndre_vals)), 4),
            current_ndvi     = round(current_ndvi, 4),
            current_ndre     = round(ndre_vals[-1], 4),
            current_velocity = round(current_vel, 5),
            current_status   = cutting_status(current_ndvi, hay_type, current_vel),
            cloud_cover_pct  = round(avg_cloud, 1),
            scene_count      = len(items),
        )

    except Exception as e:
        print(f"  [ERR ] {region}: {e} — simulating")
        return simulate_node(node, year)


def run():
    today    = datetime.now(timezone.utc)
    date_str = today.strftime("%Y-%m-%d")
    year     = today.year

    print(f"\n[HAY NDVI] {date_str} — {len(HAY_NODES)} nodes\n")

    for node in HAY_NODES:
        result = fetch_node(node, year)
        fname  = f"{node['hay_type']}_{node['region']}_{date_str}.json"
        (NDVI_DIR / fname).write_text(json.dumps(result, indent=2))
        status = result.get("current_status", "unknown")
        ndvi   = result.get("current_ndvi", 0)
        print(f"  [OUT] {fname} | NDVI: {ndvi} | Status: {status}")

    print(f"\n[HAY NDVI] Done — {len(HAY_NODES)} nodes\n")


if __name__ == "__main__":
    run()
