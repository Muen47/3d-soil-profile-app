"""
YOLO-based boring log extractor — Format 2 PDFs (MRT Orange Line).

Workflow per page:
  1. Render PDF page to PNG at 200 DPI via PyMuPDF.
  2. Run best.pt YOLO detection on the PNG.
  3. Build pixel→depth mapping from Box Depth region +
     PyMuPDF text words within that region.
  4. Build pixel→value mapping from Scale N SuSPT detections
     (linear regression through detected tick positions).
  5. For each Data Su / Data SPT-N detection: read value from
     scale mapping (x-axis) and depth from depth mapping (y-axis).
     Assign ST/SS from nearest Symbol detection.
  6. Page 1 header: extract borehole_id, easting, northing via
     PyMuPDF text (same logic as extract_format1.py).
  7. Write rows to data/bangkok_boring_logs_yolo.csv.

Usage
-----
    python pipeline/extract_yolo.py --pdf data/pdfs/OW-36.pdf
    python pipeline/extract_yolo.py --all
    python pipeline/extract_yolo.py --all --resume

YOLO class map (best.pt)
------------------------
  0  Box Alterberg Limits    10 Scale 100 Att
  1  Box Depth               11 Scale 100 SuSPT
  2  Box Description         12 Scale 20 Att
  3  Box Sample Detail       13 Scale 20 SuSPT
  4  Box Suc and SPT-N       14 Scale 40 Att
  5  Box Unit Weight         15 Scale 40 SuSPT
  6  Data Alterberg Limit    16 Scale 60 Att
  7  Data SPT-N              17 Scale 60 SuSPT
  8  Data Su                 18 Scale 80 Att
  9  Data Unit weight        19 Scale 80 SuSPT
                             20 Symbol SS
                             21 Symbol ST
"""

import argparse
import csv
import os
import re
import sys

import fitz          # PyMuPDF
import numpy as np
from ultralytics import YOLO

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import derive_consistency

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------
_ROOT      = os.path.join(os.path.dirname(__file__), "..")
PDF_DIR    = os.path.join(_ROOT, "data", "pdfs")
CSV_PATH   = os.path.join(_ROOT, "data", "bangkok_boring_logs_yolo.csv")
IMG_DIR    = os.path.join(_ROOT, "data", "page_images")
MODEL_PATH = os.path.join(os.path.dirname(__file__), "best.pt")

DPI        = 200
PX_PER_PT  = DPI / 72.0       # multiply PyMuPDF (pt) → image (px)

CSV_COLS = [
    "borehole_id", "easting", "northing",
    "depth_m", "depth_top_m", "depth_bot_m",
    "soil_layer", "soil_desc", "consistency",
    "su_kpa", "su_method", "spt_n",
    "unit_weight", "plasticity_idx", "liquid_limit", "plastic_limit",
    "water_content", "source_file", "notes",
]

# Numeric value for each SuSPT scale marker class name
SUSPT_SCALE = {
    "Scale 20 SuSPT": 20.0,
    "Scale 40 SuSPT": 40.0,
    "Scale 60 SuSPT": 60.0,
    "Scale 80 SuSPT": 80.0,
    "Scale 100 SuSPT": 100.0,
}

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

def parse_detections(results, names):
    """Return {class_name: [(x1,y1,x2,y2,conf), ...]}."""
    by_class: dict[str, list] = {}
    for box in results.boxes:
        cls  = int(box.cls[0])
        name = names[cls]
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = float(box.conf[0])
        by_class.setdefault(name, []).append((x1, y1, x2, y2, conf))
    return by_class


def _cx(det): return (det[0] + det[2]) / 2.0
def _cy(det): return (det[1] + det[3]) / 2.0
def _best(dets): return max(dets, key=lambda d: d[4])


def merged_box(dets):
    """Union bounding box of a list of detections."""
    return (min(d[0] for d in dets), min(d[1] for d in dets),
            max(d[2] for d in dets), max(d[3] for d in dets))


# ---------------------------------------------------------------------------
# Scale calibration  (x-pixel → value)
# ---------------------------------------------------------------------------

def build_scale_fn(by_class, box_left_x=None):
    """
    Fit a linear function: value = a * x_pixel + b

    Uses all detected SuSPT scale markers; optionally anchors x=box_left_x
    to value=0. Returns None when fewer than 2 reference points are available.
    """
    pts = []  # (x_px, value)

    for cls_name, value in SUSPT_SCALE.items():
        if cls_name in by_class:
            # Among duplicate detections of the same tick, pick highest conf
            d = _best(by_class[cls_name])
            pts.append((_cx(d), value))

    # Anchor: left edge of "Box Suc and SPT-N" → value 0
    if box_left_x is not None:
        pts.append((box_left_x, 0.0))

    if len(pts) < 2:
        return None

    pts.sort(key=lambda p: p[0])
    xs   = np.array([p[0] for p in pts])
    vals = np.array([p[1] for p in pts])

    # Linear least-squares: value = slope * x + intercept
    coeffs = np.polyfit(xs, vals, 1)   # [slope, intercept]

    def fn(x_px: float) -> float:
        return float(np.polyval(coeffs, x_px))

    fn._pts    = pts
    fn._coeffs = coeffs
    return fn


# ---------------------------------------------------------------------------
# Depth calibration  (y-pixel → depth_m)
# ---------------------------------------------------------------------------

def _to_float(s: str):
    try:
        return float(str(s).strip().replace(",", ""))
    except ValueError:
        return None


def build_depth_fn(page_words, depth_box_px, y_page_max_px):
    """
    Build y_pixel → depth_m linear function from numeric text within the
    depth column (Box Depth region).

    page_words : result of page.get_text('words') — coords in PDF points
    depth_box_px : (x1, y1, x2, y2) in image pixels
    y_page_max_px : image height in pixels (for bounds checking)
    """
    bx1, by1, bx2, by2 = depth_box_px
    # Convert depth-box pixel bounds to PDF-point bounds with padding
    pad_h = 20 / PX_PER_PT      # 20 px horizontal padding in points
    x1_pt = (bx1 - pad_h * PX_PER_PT) / PX_PER_PT   # same as (bx1/PX_PER_PT - pad_h)
    x2_pt = (bx2 + pad_h * PX_PER_PT) / PX_PER_PT

    pairs = []   # (y_px, depth_m)
    for w in page_words:
        wx0, wy0, wx1, wy1, word = w[0], w[1], w[2], w[3], w[4]
        wcx_pt = (wx0 + wx1) / 2.0
        wcy_pt = (wy0 + wy1) / 2.0
        if not (x1_pt <= wcx_pt <= x2_pt):
            continue
        v = _to_float(word)
        if v is None or v < 0 or v > 200:
            continue
        y_px = wcy_pt * PX_PER_PT
        # Accept words broadly within the depth column y-range (+ 100 px slack)
        if -100 <= y_px - by1 <= (by2 - by1) + 100:
            pairs.append((y_px, v))

    if len(pairs) < 2:
        return None

    # Deduplicate by y-bucket (5 px bucket)
    buckets: dict[int, tuple] = {}
    for y_px, v in pairs:
        k = int(round(y_px / 5.0))
        if k not in buckets:
            buckets[k] = (y_px, v)
    pairs = sorted(buckets.values())

    if len(pairs) < 2:
        return None

    ys   = np.array([p[0] for p in pairs])
    ds   = np.array([p[1] for p in pairs])
    coeffs = np.polyfit(ys, ds, 1)   # [slope, intercept]

    def fn(y_px: float) -> float:
        return max(0.0, float(np.polyval(coeffs, y_px)))

    fn._pairs  = pairs
    fn._coeffs = coeffs
    return fn


# ---------------------------------------------------------------------------
# Symbol matching
# ---------------------------------------------------------------------------

def nearest_symbol(y_px, st_dets, ss_dets, fallback="ST", tol=80):
    """Return 'ST' or 'SS' for the symbol closest in y to data point."""
    best_dist, best_type = tol + 1, fallback
    for d in st_dets:
        dist = abs(_cy(d) - y_px)
        if dist < best_dist:
            best_dist, best_type = dist, "ST"
    for d in ss_dets:
        dist = abs(_cy(d) - y_px)
        if dist < best_dist:
            best_dist, best_type = dist, "SS"
    return best_type


# ---------------------------------------------------------------------------
# Per-page extraction
# ---------------------------------------------------------------------------

def extract_page(fitz_page, yolo_model, img_path, conf_thresh=0.25):
    """
    Run YOLO + calibration on one rendered page image.
    Returns a list of raw row dicts (without borehole_id/easting/northing).
    """
    results  = yolo_model(img_path, conf=conf_thresh, verbose=False)[0]
    by_class = parse_detections(results, yolo_model.names)

    img_h = results.orig_shape[0]   # height of the image in pixels

    # ── Depth box ──────────────────────────────────────────────────────────
    if "Box Depth" not in by_class:
        return []

    depth_box_px = merged_box(by_class["Box Depth"])

    # PyMuPDF words for depth calibration
    page_words = fitz_page.get_text("words")
    depth_fn   = build_depth_fn(page_words, depth_box_px, img_h)
    if depth_fn is None:
        return []

    # ── SuSPT scale ────────────────────────────────────────────────────────
    box_left_x = None
    if "Box Suc and SPT-N" in by_class:
        box_left_x = min(d[0] for d in by_class["Box Suc and SPT-N"])

    scale_fn = build_scale_fn(by_class, box_left_x=box_left_x)

    # ── Symbols ────────────────────────────────────────────────────────────
    st_dets = by_class.get("Symbol ST", [])
    ss_dets = by_class.get("Symbol SS", [])

    rows = []

    # ── Data Su ────────────────────────────────────────────────────────────
    for det in by_class.get("Data Su", []):
        y_px  = _cy(det)
        x_px  = _cx(det)
        depth = depth_fn(y_px)
        su    = round(max(0.0, scale_fn(x_px)), 1) if scale_fn else None
        sym   = nearest_symbol(y_px, st_dets, ss_dets, fallback="ST")
        rows.append({
            "depth_m":     round(depth, 2),
            "depth_top_m": round(depth - 0.45, 2),
            "depth_bot_m": round(depth + 0.45, 2),
            "su_kpa":      su if su is not None else "",
            "su_method":   sym,
            "spt_n":       "",
            "consistency": derive_consistency(su, None, "") or "",
            "_det_conf":   det[4],
        })

    # ── Data SPT-N ─────────────────────────────────────────────────────────
    for det in by_class.get("Data SPT-N", []):
        y_px  = _cy(det)
        x_px  = _cx(det)
        depth = depth_fn(y_px)
        raw   = scale_fn(x_px) if scale_fn else None
        spt   = max(0, int(round(raw))) if raw is not None else None
        sym   = nearest_symbol(y_px, st_dets, ss_dets, fallback="SS")
        rows.append({
            "depth_m":     round(depth, 2),
            "depth_top_m": round(depth - 0.45, 2),
            "depth_bot_m": round(depth + 0.45, 2),
            "su_kpa":      "",
            "su_method":   "",
            "spt_n":       spt if spt is not None else "",
            "consistency": derive_consistency(None, spt, "") or "",
            "_det_conf":   det[4],
        })

    return rows


# ---------------------------------------------------------------------------
# Header extraction (page 1, PyMuPDF)
# ---------------------------------------------------------------------------

def extract_header(fitz_page) -> dict:
    words = fitz_page.get_text("words")
    header = {"borehole_id": None, "easting": None, "northing": None}

    # Group words into text lines by similar y
    hw = sorted(words, key=lambda w: (round(w[1] / 5) * 5, w[0]))
    lines, cur_y, cur_line = [], -999.0, []
    for w in hw:
        if abs(w[1] - cur_y) > 5:
            if cur_line:
                lines.append(" ".join(cur_line))
            cur_line = [w[4]]
            cur_y = w[1]
        else:
            cur_line.append(w[4])
    if cur_line:
        lines.append(" ".join(cur_line))

    for line in lines:
        ll = line.lower()
        if header["borehole_id"] is None and ("number" in ll or " no" in ll or "borehole" in ll):
            m = re.search(r"(OW-\d+|BH-\d+|\bB\d+\b)", line, re.I)
            if m:
                header["borehole_id"] = m.group(1).upper()
        if header["easting"] is None and "co-ordinate" in ll and " e" in ll:
            for n in re.findall(r"[\d,]+", line):
                v = _to_float(n)
                if v and 600000 < v < 700000:
                    header["easting"] = v; break
        if header["northing"] is None and "co-ordinate" in ll and " n" in ll:
            for n in re.findall(r"[\d,]+", line):
                v = _to_float(n)
                if v and 1400000 < v < 1700000:
                    header["northing"] = v; break

    # Fallback: scan every word for coordinate-shaped numbers
    if header["easting"] is None or header["northing"] is None:
        for w in words:
            v = _to_float(w[4])
            if v is None:
                continue
            if header["easting"]  is None and 600000 < v < 700000:
                header["easting"] = v
            if header["northing"] is None and 1400000 < v < 1700000:
                header["northing"] = v

    return header


# ---------------------------------------------------------------------------
# Per-PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str, yolo_model, conf=0.25,
                keep_images=False) -> list[dict]:
    source = os.path.basename(pdf_path)
    stem   = os.path.splitext(source)[0]
    os.makedirs(IMG_DIR, exist_ok=True)

    doc = fitz.open(pdf_path)
    mat = fitz.Matrix(PX_PER_PT, PX_PER_PT)

    # Page 1 header (borehole metadata)
    header   = extract_header(doc[0])
    bh_id    = header.get("borehole_id") or stem.replace("_", "-")
    easting  = header.get("easting")  or ""
    northing = header.get("northing") or ""

    all_rows = []
    for page_no, page in enumerate(doc):
        img_path = os.path.join(IMG_DIR, f"{stem}_yolo_p{page_no+1:02d}.png")
        pix = page.get_pixmap(matrix=mat)
        pix.save(img_path)

        try:
            page_rows = extract_page(page, yolo_model, img_path, conf_thresh=conf)
        except Exception as e:
            print(f"  [page {page_no+1}] ERROR: {e}")
            page_rows = []
        finally:
            if not keep_images:
                try:
                    os.remove(img_path)
                except OSError:
                    pass

        print(f"  [page {page_no+1}] {len(page_rows)} detections")
        all_rows.extend(page_rows)

    doc.close()

    # Sort by depth, deduplicate (keep highest-confidence detection per depth bucket)
    # First, bucket by 0.5 m
    buckets: dict[float, dict] = {}
    for r in all_rows:
        k = round(r["depth_m"] * 2) / 2.0   # 0.5 m buckets
        if k not in buckets or r["_det_conf"] > buckets[k]["_det_conf"]:
            buckets[k] = r

    final = []
    for r in sorted(buckets.values(), key=lambda x: x["depth_m"]):
        r.pop("_det_conf", None)
        r.update({
            "borehole_id": bh_id,
            "easting":     easting,
            "northing":    northing,
            "soil_layer":  "",
            "soil_desc":   "",
            "source_file": source,
            "unit_weight": "",
            "plasticity_idx": "",
            "liquid_limit":   "",
            "plastic_limit":  "",
            "water_content":  "",
            "notes":       "",
        })
        final.append(r)

    return final


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------

def save_rows(rows: list[dict], csv_path: str, bh_id: str) -> None:
    existing: list[dict] = []
    if os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["borehole_id"] != bh_id]
    all_rows = existing + [{col: r.get(col, "") for col in CSV_COLS} for r in rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(all_rows)


def existing_boreholes(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, encoding="utf-8") as f:
        return {r["borehole_id"] for r in csv.DictReader(f)}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_rows(rows: list[dict]) -> None:
    print(f"\n  {'depth_m':>7}  {'type':4}  {'su_kpa':>7}  {'spt_n':>5}  consistency")
    print(f"  {'-'*7}  {'-'*4}  {'-'*7}  {'-'*5}  {'-'*15}")
    for r in rows:
        stype = r.get("su_method") or ("SS" if r.get("spt_n") not in ("", None) else "?")
        print(f"  {r['depth_m']:7.2f}  {stype:4}  "
              f"{str(r.get('su_kpa',''))  :>7}  "
              f"{str(r.get('spt_n',''))   :>5}  "
              f"{r.get('consistency','')}")


def main():
    parser = argparse.ArgumentParser(
        description="YOLO-based boring log extractor (Format 2 PDFs).")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pdf",  help="Path to a single PDF")
    grp.add_argument("--all",  action="store_true",
                     help="All PDFs in data/pdfs/")
    parser.add_argument("--resume",      action="store_true",
                        help="Skip boreholes already in the output CSV")
    parser.add_argument("--conf",        type=float, default=0.25,
                        help="YOLO confidence threshold (default 0.25)")
    parser.add_argument("--csv",         default=CSV_PATH,
                        help="Output CSV path")
    parser.add_argument("--keep-images", action="store_true",
                        help="Keep intermediate PNG files")
    parser.add_argument("--show",        action="store_true",
                        help="Print extracted rows to stdout")
    parser.add_argument("--model",       default=MODEL_PATH,
                        help="Path to YOLO model weights")
    args = parser.parse_args()

    print(f"Loading YOLO model from {args.model} …")
    yolo_model = YOLO(args.model)

    pdfs = ([args.pdf] if args.pdf else
            sorted(os.path.join(PDF_DIR, f)
                   for f in os.listdir(PDF_DIR)
                   if f.lower().endswith(".pdf")))

    skip = existing_boreholes(args.csv) if args.resume else set()

    total_rows = 0
    for pdf_path in pdfs:
        stem     = os.path.splitext(os.path.basename(pdf_path))[0]
        bh_guess = stem.replace("_", "-")
        if bh_guess in skip:
            print(f"[{stem}] Skipping (already extracted)")
            continue

        print(f"\n[{os.path.basename(pdf_path)}]")
        try:
            rows = extract_pdf(pdf_path, yolo_model,
                               conf=args.conf,
                               keep_images=args.keep_images)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not rows:
            print("  WARNING: no data rows extracted")
            continue

        bh_id = rows[0]["borehole_id"]
        print(f"  → {bh_id}  E={rows[0]['easting']}  N={rows[0]['northing']}"
              f"  {len(rows)} rows")

        if args.show or args.pdf:
            _print_rows(rows)

        save_rows(rows, args.csv, bh_id)
        total_rows += len(rows)

    print(f"\n{'='*60}")
    print(f"Total: {total_rows} rows  →  {args.csv}")


if __name__ == "__main__":
    main()
