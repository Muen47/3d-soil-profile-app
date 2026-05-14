"""
PDF boring-log extraction pipeline.

Workflow per PDF:
  1. Convert each page to a PNG image via PyMuPDF (fitz).
  2. Send each page image to the Anthropic API (claude-opus-4-7 vision).
  3. Parse the structured JSON the model returns.
  4. Derive consistency from Peck et al. 1974 (su first, SPT-N fallback).
  5. Append rows to the master CSV.

Usage
-----
    python pipeline/pdf_extract.py --pdf data/pdfs/OW-01.pdf
    python pipeline/pdf_extract.py --all          # process every PDF in data/pdfs/
    python pipeline/pdf_extract.py --all --resume # skip boreholes already in CSV
"""

import argparse
import base64
import csv
import json
import os
import re
import sys
import tempfile

import fitz  # PyMuPDF
import anthropic

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import derive_consistency

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT     = os.path.join(os.path.dirname(__file__), "..")
PDF_DIR   = os.path.join(_ROOT, "data", "pdfs")
CSV_PATH  = os.path.join(_ROOT, "data", "bangkok_boring_logs.csv")
IMG_DIR   = os.path.join(_ROOT, "data", "page_images")

CSV_COLS = [
    "borehole_id", "easting", "northing",
    "depth_m", "depth_top_m", "depth_bot_m",
    "soil_layer", "soil_desc", "consistency",
    "su_kpa", "su_method", "spt_n",
    "unit_weight", "plasticity_idx", "liquid_limit", "plastic_limit",
    "water_content", "source_file", "notes",
]

# ---------------------------------------------------------------------------
# Soil layer code normalisation
# ---------------------------------------------------------------------------
LAYER_ALIASES = {
    "FILL": "MG", "MADE GROUND": "MG", "TOPSOIL": "MG", "TOP SOIL": "MG",
    "VERY SOFT CLAY": "VSC", "VERY SOFT TO SOFT CLAY": "VSC",
    "SOFT CLAY": "SOC", "SOFT TO MEDIUM CLAY": "SOC",
    "MEDIUM CLAY": "SOC", "MEDIUM TO STIFF CLAY": "SC",
    "STIFF CLAY": "SC", "STIFF TO VERY STIFF CLAY": "SC",
    "VERY STIFF CLAY": "SC",
    "MEDIUM STIFF CLAY": "MSC", "HARD CLAY": "MSC",
    "VERY STIFF TO HARD CLAY": "MSC", "HARD SILTY CLAY": "MSC",
    "FIRM SAND": "FS", "SILTY SAND": "FS", "SANDY SILT": "FS",
    "SAND": "SS", "DENSE SAND": "SS", "SILTY SAND DENSE": "SS",
    "VERY DENSE SAND": "SS", "DENSE SILTY SAND": "SS",
    "DENSE TO VERY DENSE SAND": "SS", "GRAVEL": "SS",
}

VALID_LAYERS = {"MG", "VSC", "SOC", "SC", "MSC", "FS", "SS"}


def normalise_layer(raw: str) -> str:
    if not raw:
        return ""
    upper = raw.strip().upper()
    if upper in VALID_LAYERS:
        return upper
    for alias, code in LAYER_ALIASES.items():
        if alias in upper:
            return code
    return upper[:3]   # best-effort fallback


# ---------------------------------------------------------------------------
# PDF → images
# ---------------------------------------------------------------------------

def pdf_to_images(pdf_path: str, dpi: int = 200) -> list[str]:
    """Render each page to a PNG and return the file paths."""
    os.makedirs(IMG_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(pdf_path))[0]
    doc  = fitz.open(pdf_path)
    paths = []
    scale = dpi / 72
    mat   = fitz.Matrix(scale, scale)
    for i, page in enumerate(doc):
        pix  = page.get_pixmap(matrix=mat)
        path = os.path.join(IMG_DIR, f"{stem}_page{i+1:02d}.png")
        pix.save(path)
        paths.append(path)
    doc.close()
    return paths


# ---------------------------------------------------------------------------
# Vision extraction prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are a geotechnical data extraction assistant.
Extract boring log data from the image exactly as shown — do not infer or
interpolate values that are not visible.

Return ONLY a JSON object with this schema (no markdown fences):
{
  "borehole_id": "OW-XX",
  "easting": 123456,
  "northing": 1234567,
  "rows": [
    {
      "depth_m": 3.0,
      "depth_top_m": 2.5,
      "depth_bot_m": 3.5,
      "soil_layer": "VSC",
      "soil_desc": "Very soft CLAY, grey, high plasticity (CH)",
      "su_kpa": 12.7,
      "su_method": "ST",
      "spt_n": null,
      "unit_weight": 14.7,
      "plasticity_idx": 52,
      "liquid_limit": 75,
      "plastic_limit": 23,
      "water_content": 88,
      "notes": ""
    }
  ]
}

Rules:
- depth_m is the mid-depth of the sample interval.
- su_kpa is ONLY from Shelby Tube (ST) or Field Vane (FV) tests. su_method = "ST" or "FV".
- spt_n is ONLY from Split Spoon (SS) sampling. When su_kpa is present, spt_n must be null.
- soil_layer codes: MG=Fill, VSC=Very Soft Clay, SOC=Soft Clay, SC=Stiff Clay,
  MSC=Medium-Hard/Hard Clay, FS=Firm Sand transition, SS=Sand.
- For SPT refusal (e.g. "76/225mm"), record spt_n as the blow count only (76).
- Atterberg limits and unit weight: include only when shown in the log — do not estimate.
- Leave numeric fields null if not visible for that sample.
- Return null for missing borehole_id or coordinates if not visible on this page.
"""

USER_PROMPT = """Extract all boring log data visible on this page.
Include every sample row shown. For the header, extract borehole ID,
easting (Co-ordinate E), and northing (Co-ordinate N) if visible."""


def _encode_image(path: str) -> str:
    with open(path, "rb") as f:
        return base64.standard_b64encode(f.read()).decode("utf-8")


def extract_page(client: anthropic.Anthropic, image_path: str) -> dict:
    """Send one page image to Claude and return parsed JSON."""
    b64 = _encode_image(image_path)
    msg = client.messages.create(
        model="claude-opus-4-7",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64,
                    },
                },
                {"type": "text", "text": USER_PROMPT},
            ],
        }],
    )
    raw = msg.content[0].text.strip()
    # Strip accidental markdown fences
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Merge pages and build CSV rows
# ---------------------------------------------------------------------------

def merge_pages(pages: list[dict], source_file: str) -> list[dict]:
    """
    Combine extracted data from multiple pages of the same borehole.
    Header fields (borehole_id, easting, northing) are taken from the first
    page that provides them.
    """
    borehole_id = None
    easting     = None
    northing    = None

    for p in pages:
        borehole_id = borehole_id or p.get("borehole_id")
        easting     = easting     or p.get("easting")
        northing    = northing    or p.get("northing")

    all_rows = []
    for p in pages:
        for row in p.get("rows", []):
            row["borehole_id"] = borehole_id
            row["easting"]     = easting
            row["northing"]    = northing
            row["source_file"] = source_file
            row["soil_layer"]  = normalise_layer(row.get("soil_layer", ""))
            # Enforce mutual exclusivity of su_kpa and spt_n
            has_su = bool(row.get("su_kpa")) or bool(row.get("su_method"))
            if has_su:
                row["spt_n"] = None
            else:
                row["su_kpa"]    = None
                row["su_method"] = None
            # Derive consistency from Peck et al. 1974
            row["consistency"] = derive_consistency(
                row.get("su_kpa"), row.get("spt_n"), row.get("soil_layer", "")
            )
            all_rows.append(row)

    # Sort by depth
    all_rows.sort(key=lambda r: float(r.get("depth_m") or 0))
    return all_rows


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _existing_boreholes(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return {row["borehole_id"] for row in reader}


def _append_rows(rows: list[dict], csv_path: str) -> None:
    file_exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in CSV_COLS})


def _overwrite_rows(rows: list[dict], borehole_id: str, csv_path: str) -> None:
    """Replace all rows for a given borehole_id in the CSV."""
    existing: list[dict] = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["borehole_id"] != borehole_id]

    all_rows = existing + [{col: r.get(col, "") for col in CSV_COLS} for r in rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLS)
        writer.writeheader()
        writer.writerows(all_rows)


# ---------------------------------------------------------------------------
# Main extraction entry point
# ---------------------------------------------------------------------------

def process_pdf(
    pdf_path: str,
    client: anthropic.Anthropic,
    csv_path: str = CSV_PATH,
    keep_images: bool = False,
    overwrite: bool = True,
) -> list[dict]:
    source_file = os.path.basename(pdf_path)
    print(f"\n[{source_file}] Converting to images...")
    image_paths = pdf_to_images(pdf_path)
    print(f"  {len(image_paths)} page(s)")

    page_results = []
    for i, img_path in enumerate(image_paths, 1):
        print(f"  Page {i}: extracting via Claude vision...")
        try:
            result = extract_page(client, img_path)
            page_results.append(result)
            print(f"    -> {len(result.get('rows', []))} rows extracted")
        except Exception as e:
            print(f"    ERROR: {e}")
            page_results.append({"rows": []})
        finally:
            if not keep_images:
                os.remove(img_path)

    rows = merge_pages(page_results, source_file)
    print(f"  Total: {len(rows)} merged rows")

    if rows:
        if overwrite:
            bh = rows[0].get("borehole_id", source_file)
            _overwrite_rows(rows, bh, csv_path)
        else:
            _append_rows(rows, csv_path)
        print(f"  Saved to {csv_path}")

    return rows


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Extract boring log data from PDFs.")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--pdf",  help="Path to a single PDF file")
    group.add_argument("--all",  action="store_true", help="Process all PDFs in data/pdfs/")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip boreholes already present in the CSV")
    parser.add_argument("--keep-images", action="store_true",
                        help="Keep intermediate PNG files after extraction")
    parser.add_argument("--csv",  default=CSV_PATH, help="Output CSV path")
    args = parser.parse_args()

    client = anthropic.Anthropic()   # reads ANTHROPIC_API_KEY from env

    if args.pdf:
        pdfs = [args.pdf]
    else:
        pdfs = sorted(
            os.path.join(PDF_DIR, f)
            for f in os.listdir(PDF_DIR)
            if f.lower().endswith(".pdf")
        )
        print(f"Found {len(pdfs)} PDFs in {PDF_DIR}")

    skip = _existing_boreholes(args.csv) if args.resume else set()

    for pdf_path in pdfs:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        if stem in skip:
            print(f"[{stem}] Already extracted — skipping")
            continue
        try:
            process_pdf(
                pdf_path,
                client,
                csv_path=args.csv,
                keep_images=args.keep_images,
                overwrite=True,
            )
        except Exception as e:
            print(f"[{stem}] FAILED: {e}")


if __name__ == "__main__":
    main()
