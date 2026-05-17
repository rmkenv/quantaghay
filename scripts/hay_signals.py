"""
QuantAgri Hay — Signal Generator
==================================
Reads hay NDVI/NDRE data + prices, calls Ollama Cloud to generate
structured hay market signals for each production node.

Hay signals differ from commodity crop signals:
- Cutting readiness is a key output (not just bullish/bearish)
- Quality estimation via NDRE (protein proxy)
- Regional supply impact from cutting timing + weather windows
- Export grade vs domestic grade distinction

Output: data/hay/signals/latest.json
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from hay_config import HAY_DIR, HAY_NODES, HAY_TYPES
from ollama_client import chat_json

SIG_DIR  = HAY_DIR / "signals"
NDVI_DIR = HAY_DIR / "ndvi"
SIG_DIR.mkdir(parents=True, exist_ok=True)

HAY_SYSTEM = """You are QuantAgri Hay Intelligence Engine.
You analyze Sentinel-2 NDVI/NDRE spectral data and hay market conditions
for North American hay production regions.
You understand hay-specific dynamics: multi-cut cycles, cutting readiness,
protein quality estimation via NDRE, weather window risk, and export grade standards.
Always respond with valid JSON only. No markdown. No preamble."""


def load_latest_ndvi(hay_type: str, region: str) -> dict | None:
    files = sorted(NDVI_DIR.glob(f"{hay_type}_{region}_*.json"))
    return json.loads(files[-1].read_text()) if files else None


def load_latest_prices() -> dict:
    path = HAY_DIR / "prices" / "latest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def build_hay_signal_prompt(node_data: dict, prices: dict) -> str:
    hay_type   = node_data["hay_type"]
    region     = node_data["region"].replace("_", " ")
    grade      = node_data.get("grade", "Good")
    primary_use = node_data.get("primary_use", "Beef/Dairy")
    source     = node_data.get("source", "unknown")
    composites = node_data.get("composites", [])[-6:]  # last 6 periods

    # Get relevant price benchmark
    hay_prices = prices.get("usda_benchmarks", {}).get(hay_type, {})
    price_str  = " | ".join([f"{g}: ${d['price']}/ton" for g, d in hay_prices.items()]) or "N/A"

    regional_price = prices.get("regional_prices", {}).get(
        node_data.get("region", ""), {}
    )
    regional_str = f"${regional_price.get('final_price', 'N/A')}/ton" if regional_price else "N/A"

    cattle_proxy = prices.get("proxy_tickers", {}).get("LE=F", {})
    cattle_str   = f"${cattle_proxy.get('price','N/A')} | Week: {cattle_proxy.get('weekChg','N/A')}%" if cattle_proxy else "N/A"

    return f"""Analyze {hay_type} hay production in {region}.

SPECTRAL DATA (Planetary Computer · Sentinel-2 L2A · {source}):
State/Province: {node_data.get('state')} | Country: {node_data.get('country')}
Grade: {grade} | Primary Use: {primary_use}
Current NDVI: {node_data.get('current_ndvi')} | Current NDRE: {node_data.get('current_ndre')}
Current Velocity: {node_data.get('current_velocity')} | Status: {node_data.get('current_status')}
Peak NDVI: {node_data.get('peak_ndvi')} | Peak NDRE: {node_data.get('peak_ndre')}
Estimated cuts/season: {node_data.get('estimated_cuts')}

RECENT COMPOSITES (8-day):
{json.dumps(composites, indent=2)}

MARKET PRICES:
USDA Benchmarks ({hay_type}): {price_str}
Regional Price ({region}): {regional_str}
Live Cattle Proxy (LE=F): {cattle_str}
Source: USDA AMS (https://www.ams.usda.gov/market-news/hay)

Return ONLY this JSON — no markdown, no preamble:
{{
  "hay_type": "{hay_type}",
  "region": "{region}",
  "state": "{node_data.get('state')}",
  "grade": "{grade}",
  "primary_use": "{primary_use}",
  "cutting_status": "post_cut"|"growing"|"approaching_ready"|"ready_to_cut"|"past_peak",
  "days_to_cutting": <integer or null>,
  "estimated_yield_tons_acre": <float e.g. 1.8>,
  "quality_outlook": "Supreme"|"Premium"|"Good"|"Fair",
  "protein_estimate_pct": <float e.g. 18.5>,
  "moisture_risk": "Low"|"Moderate"|"High",
  "supply_sentiment": "Tight"|"Balanced"|"Ample",
  "price_direction": "Rising"|"Stable"|"Declining",
  "confidence": <float 0.0-1.0>,
  "rationale": "<2-3 sentences covering NDVI velocity, NDRE protein signal, cutting timing, and regional supply impact>",
  "weather_window_risk": "<e.g. Low — dry pattern forecast | High — rain likely this week>",
  "export_grade_eligible": true|false,
  "key_risk": "<e.g. Second cut delayed by rain | Colorado River water cuts reducing acreage>",
  "spectral_velocity": "<e.g. +0.031/8d>",
  "ndre_trend": "Rising"|"Stable"|"Declining"
}}"""


def run():
    today    = datetime.now(timezone.utc)
    date_str = today.strftime("%Y-%m-%d")
    print(f"\n[HAY SIGNALS] {date_str} — {len(HAY_NODES)} nodes\n")

    prices   = load_latest_prices()
    results  = []
    errors   = []

    for node in HAY_NODES:
        hay_type = node["hay_type"]
        region   = node["region"]
        label    = f"{hay_type}/{region}"

        ndvi = load_latest_ndvi(hay_type, region)
        if ndvi is None:
            print(f"  [SKIP] {label} — no NDVI data")
            errors.append(label)
            continue

        print(f"  [LLM ] {label}")
        try:
            signal = chat_json(
                prompt = build_hay_signal_prompt(ndvi, prices),
                system = HAY_SYSTEM,
            )
            signal["generatedAt"] = today.isoformat()
            signal["ndviSource"]  = ndvi.get("source", "unknown")
            results.append(signal)
            status = signal.get("cutting_status", "?")
            conf   = signal.get("confidence", 0)
            print(f"  [OK  ] {label} → {status} | {conf:.0%} conf")
        except Exception as e:
            print(f"  [ERR ] {label}: {e}")
            errors.append(label)

    snapshot = {
        "generatedAt": today.isoformat(),
        "date":        date_str,
        "signalCount": len(results),
        "errors":      errors,
        "signals":     results,
    }

    (SIG_DIR / "latest.json").write_text(json.dumps(snapshot, indent=2))
    (SIG_DIR / f"{date_str}.json").write_text(json.dumps(snapshot, indent=2))

    print(f"\n[HAY SIGNALS] {len(results)} signals written")
    if errors:
        print(f"[HAY SIGNALS] {len(errors)} skipped: {errors}")


if __name__ == "__main__":
    run()
