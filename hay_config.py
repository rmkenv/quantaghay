"""
QuantAgri Hay — North American Hay Analysis Configuration
All hay production nodes, USDA hay types, price sources, and paths.
"""

from pathlib import Path

ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
HAY_DIR  = DATA_DIR / "hay"

for d in [
    HAY_DIR / "ndvi",
    HAY_DIR / "prices",
    HAY_DIR / "signals",
    HAY_DIR / "newsletter",
    HAY_DIR / "cutting_schedule",
]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hay production nodes ──────────────────────────────────────────────
# bbox = [lon_min, lat_min, lon_max, lat_max]
# Covers all major North American hay-producing regions
HAY_NODES = [

    # ── Alfalfa — Western Irrigated (premium export grade) ──
    dict(hay_type="Alfalfa", region="Imperial_Valley_CA",
         state="CA", country="US", grade="Supreme/Premium",
         primary_use="Dairy/Export",
         bbox=[-115.6, 32.7, -114.8, 33.1]),

    dict(hay_type="Alfalfa", region="Yuma_AZ",
         state="AZ", country="US", grade="Supreme/Premium",
         primary_use="Dairy/Export",
         bbox=[-114.7, 32.4, -114.0, 32.9]),

    dict(hay_type="Alfalfa", region="Columbia_Basin_WA",
         state="WA", country="US", grade="Premium/Good",
         primary_use="Dairy/Export",
         bbox=[-119.5, 46.5, -118.0, 47.5]),

    dict(hay_type="Alfalfa", region="Snake_River_Plain_ID",
         state="ID", country="US", grade="Premium/Good",
         primary_use="Dairy/Export",
         bbox=[-115.5, 42.5, -112.0, 44.0]),

    dict(hay_type="Alfalfa", region="San_Joaquin_Valley_CA",
         state="CA", country="US", grade="Good/Premium",
         primary_use="Dairy",
         bbox=[-121.0, 35.5, -119.0, 37.5]),

    # ── Alfalfa — Great Plains / Interior ──
    dict(hay_type="Alfalfa", region="Central_Kansas",
         state="KS", country="US", grade="Good",
         primary_use="Beef/Dairy",
         bbox=[-100.0, 37.5, -96.0, 39.5]),

    dict(hay_type="Alfalfa", region="Nebraska_Panhandle",
         state="NE", country="US", grade="Good/Premium",
         primary_use="Beef/Dairy",
         bbox=[-104.0, 41.0, -100.5, 42.5]),

    dict(hay_type="Alfalfa", region="Eastern_Colorado",
         state="CO", country="US", grade="Good",
         primary_use="Beef/Dairy",
         bbox=[-104.5, 38.0, -102.0, 40.5]),

    dict(hay_type="Alfalfa", region="Central_Utah",
         state="UT", country="US", grade="Premium",
         primary_use="Dairy/Export",
         bbox=[-112.5, 39.0, -110.5, 41.0]),

    # ── Alfalfa — Canada ──
    dict(hay_type="Alfalfa", region="Southern_Alberta_CA",
         state="AB", country="CA", grade="Good/Premium",
         primary_use="Beef/Export",
         bbox=[-113.5, 49.5, -110.0, 51.5]),

    dict(hay_type="Alfalfa", region="Saskatchewan_Irrigated_CA",
         state="SK", country="CA", grade="Good",
         primary_use="Beef",
         bbox=[-107.0, 50.0, -103.0, 52.0]),

    # ── Timothy — Pacific Northwest (premium Japan/Korea export) ──
    dict(hay_type="Timothy", region="Columbia_Basin_WA",
         state="WA", country="US", grade="Premium",
         primary_use="Export/Equine",
         bbox=[-119.5, 46.5, -118.0, 47.5]),

    dict(hay_type="Timothy", region="Central_Oregon",
         state="OR", country="US", grade="Premium",
         primary_use="Export/Equine",
         bbox=[-121.5, 43.5, -119.0, 45.0]),

    dict(hay_type="Timothy", region="Northern_Idaho",
         state="ID", country="US", grade="Premium",
         primary_use="Export/Equine",
         bbox=[-117.0, 46.5, -115.0, 48.0]),

    # ── Timothy — Canada ──
    dict(hay_type="Timothy", region="Ontario_CA",
         state="ON", country="CA", grade="Good/Premium",
         primary_use="Export/Dairy",
         bbox=[-81.0, 43.5, -76.0, 45.5]),

    dict(hay_type="Timothy", region="Quebec_CA",
         state="QC", country="CA", grade="Good",
         primary_use="Dairy",
         bbox=[-74.0, 45.0, -71.0, 47.0]),

    # ── Orchardgrass ──
    dict(hay_type="Orchardgrass", region="Willamette_Valley_OR",
         state="OR", country="US", grade="Premium",
         primary_use="Equine/Export",
         bbox=[-123.5, 44.0, -122.0, 45.5]),

    dict(hay_type="Orchardgrass", region="Virginia_Shenandoah",
         state="VA", country="US", grade="Good/Premium",
         primary_use="Equine/Beef",
         bbox=[-80.5, 38.0, -78.0, 39.5]),

    # ── Mixed Grass — Midwest ──
    dict(hay_type="Mixed_Grass", region="Iowa",
         state="IA", country="US", grade="Good",
         primary_use="Beef/Dairy",
         bbox=[-96.5, 40.5, -90.0, 43.5]),

    dict(hay_type="Mixed_Grass", region="Missouri",
         state="MO", country="US", grade="Good",
         primary_use="Beef",
         bbox=[-95.8, 36.0, -89.1, 40.6]),

    dict(hay_type="Mixed_Grass", region="South_Dakota",
         state="SD", country="US", grade="Good",
         primary_use="Beef",
         bbox=[-104.0, 43.0, -96.5, 45.9]),

    # ── Bermudagrass — South ──
    dict(hay_type="Bermudagrass", region="Texas_South",
         state="TX", country="US", grade="Good",
         primary_use="Beef",
         bbox=[-100.0, 28.0, -96.0, 31.5]),

    dict(hay_type="Bermudagrass", region="Oklahoma",
         state="OK", country="US", grade="Good",
         primary_use="Beef",
         bbox=[-100.0, 33.7, -94.4, 37.0]),

    dict(hay_type="Bermudagrass", region="Georgia",
         state="GA", country="US", grade="Good",
         primary_use="Beef/Equine",
         bbox=[-85.6, 30.4, -80.8, 35.0]),
]

# ── Hay types for analysis ────────────────────────────────────────────
HAY_TYPES = ["Alfalfa", "Timothy", "Orchardgrass", "Mixed_Grass", "Bermudagrass"]

# ── USDA Hay Report URLs (Agricultural Marketing Service) ─────────────
USDA_HAY_REPORTS = {
    "National":          "https://www.ams.usda.gov/mnreports/lp_fv050.txt",
    "Columbia_Basin_WA": "https://www.ams.usda.gov/mnreports/ams_3058.pdf",
    "California":        "https://www.ams.usda.gov/mnreports/LA_FV900.txt",
    "Southwest":         "https://www.ams.usda.gov/mnreports/lp_fy770.txt",
    "Midwest":           "https://www.ams.usda.gov/mnreports/lp_fy720.txt",
    "Southeast":         "https://www.ams.usda.gov/mnreports/lp_fy640.txt",
    "Northern_Plains":   "https://www.ams.usda.gov/mnreports/lp_fy700.txt",
}

# ── Growing season by hay type ────────────────────────────────────────
HAY_SEASONS = {
    "Alfalfa":      ("04", "10"),   # 3-8 cuttings/year depending on region
    "Timothy":      ("04", "08"),   # 1-2 cuttings
    "Orchardgrass": ("04", "09"),   # 2-3 cuttings
    "Mixed_Grass":  ("04", "09"),
    "Bermudagrass": ("05", "10"),   # warm-season, southern US
}

# ── Cutting schedule thresholds ───────────────────────────────────────
# NDVI thresholds that indicate cutting readiness by hay type
CUTTING_NDVI_THRESHOLDS = {
    "Alfalfa":      {"ready": 0.65, "peak": 0.75, "past_peak": 0.60},
    "Timothy":      {"ready": 0.60, "peak": 0.72, "past_peak": 0.55},
    "Orchardgrass": {"ready": 0.58, "peak": 0.70, "past_peak": 0.53},
    "Mixed_Grass":  {"ready": 0.55, "peak": 0.68, "past_peak": 0.50},
    "Bermudagrass": {"ready": 0.52, "peak": 0.65, "past_peak": 0.48},
}

# ── Planetary Computer settings ───────────────────────────────────────
PC_STAC_URL   = "https://planetarycomputer.microsoft.com/api/stac/v1"
COLLECTION    = "sentinel-2-l2a"
MAX_CLOUD_PCT = 20   # tighter than commodity crops — hay quality is cloud-sensitive
RESOLUTION    = 20   # 20m for better field-level detection

# ── Sentinel-2 band indices for hay ──────────────────────────────────
# Beyond NDVI: hay analysis uses additional indices for quality estimation
BAND_INDICES = {
    "NDVI":  "NIR-Red / NIR+Red",              # vegetation vigor
    "NDRE":  "NIR-RedEdge / NIR+RedEdge",      # chlorophyll / protein proxy
    "NDWI":  "NIR-SWIR / NIR+SWIR",            # moisture stress
    "EVI":   "Enhanced Vegetation Index",       # canopy structure
    "LSWI":  "Land Surface Water Index",        # water content
}
