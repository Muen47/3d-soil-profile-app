"""
Format-1 boring log extractor — digital PDFs (MRT Orange Line project).

Uses PyMuPDF positional word extraction to parse fixed-column tables:
  x≈97   : sample type (ST / SS)
  x≈117  : sample number
  x≈140-295 : soil description (layer boundary rows)
  x≈305  : layer boundary depth value
  x≈460-510 : Suc kPa (ST) or SPT-N (SS)

Output: ONE ROW PER LAYER BOUNDARY INTERVAL.
  depth_top_m = depth label at boundary[i]
  depth_bot_m = depth label at boundary[i+1]  (last layer → total borehole depth)
  depth_m     = midpoint of the interval
  su_kpa/spt_n = mean of all test samples whose midpoint depth falls in the interval

Depth is derived from the y-pixel position of each word using a linear
scale calibrated from the first two boundary-depth markers on each page.

Usage
-----
    python pipeline/extract_format1.py --pdf data/pdfs/OW-01.pdf
    python pipeline/extract_format1.py --all
    python pipeline/extract_format1.py --all --resume
"""

import argparse
import csv
import os
import re
import sys

import fitz  # PyMuPDF

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import derive_consistency

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ROOT    = os.path.join(os.path.dirname(__file__), "..")
PDF_DIR  = os.path.join(_ROOT, "data", "pdfs")
CSV_PATH = os.path.join(_ROOT, "data", "bangkok_boring_logs_real.csv")

CSV_COLS = [
    "borehole_id", "easting", "northing",
    "depth_m", "depth_top_m", "depth_bot_m",
    "soil_layer", "soil_desc", "consistency",
    "su_kpa", "su_method", "spt_n",
    "unit_weight", "plasticity_idx", "liquid_limit", "plastic_limit",
    "water_content", "source_file", "notes",
]

# ---------------------------------------------------------------------------
# Soil layer classification from description text
# ---------------------------------------------------------------------------
_LAYER_RULES = [
    # Most-specific first
    (["very soft to soft", "very soft clay"],                    "VSC"),
    (["soft to medium", "soft clay", "soft to soft"],           "SOC"),
    (["very stiff to hard", "hard clay", "hard silty clay",
      "hard silt"],                                              "MSC"),
    (["stiff to very stiff", "very stiff clay", "stiff clay"],  "SC"),
    (["medium stiff", "medium clay", "medium to stiff"],        "SC"),
    (["fill", "topsoil", "top soil", "made ground"],            "MG"),
    (["firm sand", "sandy clay transition", "transition"],       "FS"),
    (["dense sand", "very dense sand", "dense silty sand",
      "sand with silt", "silty sand", " sand "],                "SS"),
]

def classify_layer(desc: str) -> str:
    low = desc.lower()
    for keywords, code in _LAYER_RULES:
        if any(kw in low for kw in keywords):
            return code
    if "sand" in low:
        return "SS"
    if "clay" in low:
        return "SOC"
    return "MG"

# ---------------------------------------------------------------------------
# Number helpers
# ---------------------------------------------------------------------------
_NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)*$")

def _to_float(s: str):
    s = s.strip().replace(",", "")
    try:
        return float(s)
    except ValueError:
        return None

def _is_numeric(s: str) -> bool:
    return bool(_NUM_RE.match(s.strip()))

# ---------------------------------------------------------------------------
# Header extraction (y < 290 on page 1)
# ---------------------------------------------------------------------------

def _extract_header(words: list) -> dict:
    """Return {borehole_id, easting, northing, total_depth} from page-1 header words."""
    header = {"borehole_id": None, "easting": None, "northing": None, "total_depth": None}
    hw = [w for w in words if w[1] < 290]
    hw.sort(key=lambda w: (round(w[1] / 5), w[0]))

    # Reconstruct lines by grouping words with similar y
    lines: list[list] = []
    cur_y = -999
    cur_line: list = []
    for w in hw:
        if abs(w[1] - cur_y) > 4:
            if cur_line:
                lines.append(cur_line)
            cur_line = [(w[0], w[4])]
            cur_y = w[1]
        else:
            cur_line.append((w[0], w[4]))
    if cur_line:
        lines.append(cur_line)

    for line in lines:
        text = " ".join(tok for _, tok in line)
        tl = text.lower()

        # Borehole id
        if "number" in tl or " no:" in tl or "borehole" in tl:
            m = re.search(r"(OW-\d+|BH-\d+|\bB\d+\b)", text, re.I)
            if m:
                header["borehole_id"] = m.group(1).upper()

        # Easting
        if "co-ordinate" in tl and " e" in tl:
            nums = re.findall(r"[\d,]+", text)
            for n in nums:
                v = _to_float(n)
                if v and 600000 < v < 700000:
                    header["easting"] = v
                    break

        # Northing
        if "co-ordinate" in tl and " n" in tl:
            nums = re.findall(r"[\d,]+", text)
            for n in nums:
                v = _to_float(n)
                if v and 1400000 < v < 1700000:
                    header["northing"] = v
                    break

        # Total borehole depth — look for "total depth", "depth of borehole", etc.
        if header["total_depth"] is None and (
            "total depth" in tl or "depth of borehole" in tl
            or ("depth" in tl and "total" in tl)
        ):
            nums = re.findall(r"\d+(?:\.\d+)?", text)
            for n in nums:
                v = _to_float(n)
                if v and 5.0 < v < 300.0:
                    header["total_depth"] = v
                    break

    # Fallback numeric scan for easting / northing
    if header["easting"] is None or header["northing"] is None:
        all_nums = []
        for w in hw:
            v = _to_float(w[4])
            if v is not None:
                all_nums.append(v)
        for v in all_nums:
            if header["easting"] is None and 600000 < v < 700000:
                header["easting"] = v
            if header["northing"] is None and 1400000 < v < 1700000:
                header["northing"] = v

    return header

# ---------------------------------------------------------------------------
# Depth-scale calibration from boundary-depth column (x≈305)
# ---------------------------------------------------------------------------
BOUNDARY_X_MIN = 298
BOUNDARY_X_MAX = 318

def _calibrate_depth_scale(words: list) -> tuple:
    """
    Return (y_surface, px_per_meter) from the first two numeric depth-boundary
    markers found at x≈305.  Returns (None, None) if calibration fails.
    """
    markers = []
    for w in words:
        x0, y0, word = w[0], w[1], w[4]
        if BOUNDARY_X_MIN <= x0 <= BOUNDARY_X_MAX:
            v = _to_float(word)
            if v is not None and 0.0 <= v <= 200.0:
                markers.append((y0, v))

    markers.sort(key=lambda t: t[0])
    for i in range(len(markers) - 1):
        y1, d1 = markers[i]
        y2, d2 = markers[i + 1]
        if abs(d2 - d1) > 0.5 and abs(y2 - y1) > 5:
            px_per_m = (y2 - y1) / (d2 - d1)
            y_surface = y1 - d1 * px_per_m
            return y_surface, px_per_m

    return None, None

def _y_to_depth(y, y_surface, px_per_m) -> float:
    return round((y - y_surface) / px_per_m, 2)

# ---------------------------------------------------------------------------
# Column x-ranges
# ---------------------------------------------------------------------------
TYPE_X_MIN, TYPE_X_MAX   =  90, 112   # ST / SS column
NUM_X_MIN,  NUM_X_MAX    = 110, 132   # sample number
DESC_X_MIN, DESC_X_MAX   = 135, 295   # soil description
VALUE_X_MIN, VALUE_X_MAX = 455, 520   # Suc / SPT-N

DATA_Y_MIN = 290   # ignore header area

# ---------------------------------------------------------------------------
# Unit weight chart — graphical dot extraction
# ---------------------------------------------------------------------------
# The "Unit Weight (kN/m³)" column is a small dot-chart, not a text column.
# PyMuPDF page.get_drawings() returns the individual filled-rect elements
# that make up each plotted dot.  We cluster them by y-position to find one
# UW reading per sample, then convert the cluster's mean x to kN/m³ using
# the scale labels "5" and "15" printed at the chart's x-axis.
UW_CHART_X_MIN = 428   # left edge of dot chart
UW_CHART_X_MAX = 462   # right edge (< SPT-N column at x≈465)
UW_SCALE_X_MIN = 425   # x-search window for scale labels
UW_SCALE_X_MAX = 455
UW_SCALE_Y_MIN = 265   # y-search window for scale labels (header row)
UW_SCALE_Y_MAX = 280
UW_DOT_MAX_DIM =  10   # px — max w/h of one drawing element forming a dot
UW_CLUSTER_WIN =  12   # px — y-window for grouping elements into one dot
UW_VAL_MIN     =  10.0  # kN/m³ sanity floor
UW_VAL_MAX     =  30.0  # kN/m³ sanity ceiling


def _calibrate_uw_scale(words: list) -> tuple:
    """
    Return (x_at_5, x_at_15) centre-x of the '5' and '15' kN/m³ scale
    labels, or (None, None) if not found on this page.
    Only page 1 carries these labels; caller should cache the result.
    """
    x_at_5 = x_at_15 = None
    for w in words:
        if UW_SCALE_X_MIN <= w[0] <= UW_SCALE_X_MAX and UW_SCALE_Y_MIN <= w[1] <= UW_SCALE_Y_MAX:
            try:
                v = float(w[4])
            except ValueError:
                continue
            cx = (w[0] + w[2]) / 2
            if v == 5.0:
                x_at_5 = cx
            elif v == 15.0:
                x_at_15 = cx
    return x_at_5, x_at_15


def _extract_page_uw(
    drawings: list,
    x_at_5: float,
    x_at_15: float,
    y_surface: float,
    px_per_m: float,
) -> list:
    """
    Extract unit weight readings from graphical dot clusters on one page.

    Returns list of (depth_m: float, uw_kn_m3: float).
    """
    if x_at_5 is None or x_at_15 is None:
        return []

    px_per_uw = (x_at_15 - x_at_5) / (15.0 - 5.0)  # px per kN/m³

    # Collect centroids of small drawing elements in the UW chart x-band
    pts: list = []
    for d in drawings:
        r = d["rect"]
        cx = (r[0] + r[2]) / 2
        cy = (r[1] + r[3]) / 2
        w  = r[2] - r[0]
        h  = r[3] - r[1]
        if (UW_CHART_X_MIN <= cx <= UW_CHART_X_MAX
                and cy >= DATA_Y_MIN
                and w < UW_DOT_MAX_DIM
                and h < UW_DOT_MAX_DIM):
            pts.append((cy, cx))

    if not pts:
        return []

    # Cluster by y-position (each dot = several sub-elements close in y)
    pts.sort()
    clusters: list = [[pts[0]]]
    for pt in pts[1:]:
        if pt[0] - clusters[-1][0][0] < UW_CLUSTER_WIN:
            clusters[-1].append(pt)
        else:
            clusters.append([pt])

    results: list = []
    for cluster in clusters:
        mean_cx = sum(x for _, x in cluster) / len(cluster)
        mean_cy = sum(y for y, _ in cluster) / len(cluster)
        depth = max(0.0, _y_to_depth(mean_cy, y_surface, px_per_m))
        uw    = 5.0 + (mean_cx - x_at_5) / px_per_uw
        if UW_VAL_MIN <= uw <= UW_VAL_MAX:
            results.append((round(depth, 2), round(uw, 1)))

    return results

# ---------------------------------------------------------------------------
# Per-page data extraction  (returns depth-space data, not pixel-space)
# ---------------------------------------------------------------------------

def _extract_page_data(
    words: list,
    y_surface: float,
    px_per_m: float,
) -> tuple[list, list]:
    """
    Returns
    -------
    boundaries : list of (depth_m: float, desc: str)
        One entry per depth-column marker, with description text attached.
    samples    : list of (depth_m: float, stype: str, value: float | None)
        One entry per ST or SS sample row.
    """
    data_words = [w for w in words if w[1] >= DATA_Y_MIN]

    # ── Boundary depths ──────────────────────────────────────────────────────
    raw_bounds: list[tuple[float, float]] = []   # (y_px, depth_m_from_label)
    for w in data_words:
        if BOUNDARY_X_MIN <= w[0] <= BOUNDARY_X_MAX:
            v = _to_float(w[4])
            if v is not None and 0.0 <= v <= 200.0:
                raw_bounds.append((w[1], v))

    # Attach soil description to each boundary row
    _EOB_RE = re.compile(r"end\s+of\s+borehole", re.IGNORECASE)
    boundaries: list[tuple[float, str]] = []
    for b_y, b_depth in raw_bounds:
        desc_words = [
            w[4] for w in data_words
            if DESC_X_MIN <= w[0] <= DESC_X_MAX and abs(w[1] - b_y) <= 20
        ]
        raw_desc = " ".join(desc_words)
        # Strip "End of Borehole" footer text that leaks into the last boundary
        clean_desc = _EOB_RE.sub("", raw_desc).strip(" ,./")
        boundaries.append((b_depth, clean_desc))

    # ── Test samples ─────────────────────────────────────────────────────────
    type_words: list[tuple[float, str]] = []   # (y_px, "ST"/"SS")
    for w in data_words:
        if TYPE_X_MIN <= w[0] <= TYPE_X_MAX and w[4] in ("ST", "SS"):
            type_words.append((w[1], w[4]))

    value_words: list[tuple[float, float]] = []   # (y_px, value)
    for w in data_words:
        if VALUE_X_MIN <= w[0] <= VALUE_X_MAX:
            v = _to_float(w[4])
            if v is not None and 0.0 < v <= 500.0:
                value_words.append((w[1], v))

    samples: list[tuple[float, str, float | None]] = []
    for sample_y, stype in type_words:
        depth_m = max(0.0, _y_to_depth(sample_y, y_surface, px_per_m))

        best_v, best_dist = None, 9999
        for v_y, v in value_words:
            dist = abs(v_y - sample_y)
            if dist < best_dist and dist <= 25:
                best_dist = dist
                best_v = v

        samples.append((depth_m, stype, best_v))

    return boundaries, samples

# ---------------------------------------------------------------------------
# Build layer-interval rows from aggregated page data
# ---------------------------------------------------------------------------

def _build_layer_rows(
    all_boundaries: list,
    all_samples: list,
    total_depth: float | None,
    all_uw_readings: list | None = None,
) -> list[dict]:
    """
    One row per consecutive boundary interval.
    Test-sample values are averaged per interval.
    """
    if not all_boundaries:
        return []

    # Deduplicate + sort boundaries by depth
    seen_depths: set[float] = set()
    unique: list[tuple[float, str]] = []
    for depth, desc in sorted(all_boundaries, key=lambda x: x[0]):
        key = round(depth, 1)
        if key not in seen_depths:
            seen_depths.add(key)
            unique.append((depth, desc))

    # Determine bottom of the last layer
    if total_depth is None:
        # Infer from deepest sample or deepest boundary + generous margin
        all_depths = [d for d, _, _ in all_samples] + [d for d, _ in unique]
        total_depth = round(max(all_depths) + 1.5, 2) if all_depths else unique[-1][0] + 1.5

    uw_list = all_uw_readings or []

    rows: list[dict] = []
    for i, (top_depth, desc) in enumerate(unique):
        bot_depth = unique[i + 1][0] if i + 1 < len(unique) else total_depth
        # Skip zero-width intervals (e.g. "End of Borehole" marker == total_depth)
        if round(bot_depth - top_depth, 3) <= 0:
            continue
        bot_depth = round(bot_depth, 2)
        mid_depth = round((top_depth + bot_depth) / 2, 2)
        soil_layer = classify_layer(desc) if desc.strip() else "MG"

        # Samples whose midpoint depth falls in [top_depth, bot_depth)
        interval = [
            (stype, val)
            for s_depth, stype, val in all_samples
            if top_depth <= s_depth < bot_depth and val is not None
        ]

        su_vals  = [val for stype, val in interval if stype == "ST"]
        spt_vals = [val for stype, val in interval if stype == "SS"]

        su_kpa    = round(sum(su_vals)  / len(su_vals),  1) if su_vals  else None
        spt_n     = int(round(sum(spt_vals) / len(spt_vals))) if spt_vals else None
        su_method = "ST" if su_kpa is not None else None

        # Unit weight — average of graphical dot readings within interval
        uw_vals = [
            uw for uw_depth, uw in uw_list
            if top_depth <= uw_depth < bot_depth
        ]
        unit_weight = round(sum(uw_vals) / len(uw_vals), 1) if uw_vals else None

        consistency = derive_consistency(su_kpa, spt_n, soil_layer)

        rows.append({
            "depth_m":      mid_depth,
            "depth_top_m":  round(top_depth, 2),
            "depth_bot_m":  bot_depth,
            "soil_layer":   soil_layer,
            "soil_desc":    desc.strip(),
            "consistency":  consistency or "",
            "su_kpa":       round(su_kpa, 1) if su_kpa  is not None else "",
            "su_method":    su_method or "",
            "spt_n":        spt_n        if spt_n        is not None else "",
            "unit_weight":  unit_weight  if unit_weight  is not None else "",
            "plasticity_idx": "",
            "liquid_limit":   "", "plastic_limit": "",
            "water_content":  "", "notes": "",
        })

    # Depth guard: MG below 5 m is geologically impossible — reclassify to
    # the nearest preceding layer type (fallback: SOC).
    last_non_mg = "SOC"
    for row in rows:
        if row["soil_layer"] != "MG":
            last_non_mg = row["soil_layer"]
        elif row["depth_top_m"] > 5:
            row["soil_layer"] = last_non_mg
            row["consistency"] = derive_consistency(
                row["su_kpa"] if row["su_kpa"] != "" else None,
                row["spt_n"]  if row["spt_n"]  != "" else None,
                last_non_mg,
            ) or ""

    return rows

# ---------------------------------------------------------------------------
# Per-PDF extraction
# ---------------------------------------------------------------------------

def extract_pdf(pdf_path: str) -> list[dict]:
    source = os.path.basename(pdf_path)
    doc    = fitz.open(pdf_path)

    header: dict = {}
    all_boundaries:  list = []
    all_samples:     list = []
    all_uw_readings: list = []

    # UW scale is only on page 1; cache it for subsequent pages
    uw_x_at_5: float | None  = None
    uw_x_at_15: float | None = None

    for page_no, page in enumerate(doc):
        words    = page.get_text("words")   # (x0,y0,x1,y1,word,block,line,word_no)
        drawings = page.get_drawings()

        if page_no == 0:
            header = _extract_header(words)
            uw_x_at_5, uw_x_at_15 = _calibrate_uw_scale(words)

        y_surface, px_per_m = _calibrate_depth_scale(words)
        if y_surface is None:
            print(f"  [page {page_no+1}] WARNING: could not calibrate depth scale — skipping")
            continue

        bounds, samples = _extract_page_data(words, y_surface, px_per_m)
        all_boundaries.extend(bounds)
        all_samples.extend(samples)

        uw_page = _extract_page_uw(drawings, uw_x_at_5, uw_x_at_15, y_surface, px_per_m)
        all_uw_readings.extend(uw_page)

    doc.close()

    rows = _build_layer_rows(
        all_boundaries, all_samples, header.get("total_depth"), all_uw_readings
    )

    # Stamp header fields
    stem_id  = os.path.splitext(source)[0].replace("_", "-")
    bh_id    = stem_id
    easting  = header.get("easting")  or ""
    northing = header.get("northing") or ""

    for r in rows:
        r["borehole_id"] = bh_id
        r["easting"]     = easting
        r["northing"]    = northing
        r["source_file"] = source

    return rows

# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------

def _existing_boreholes(csv_path: str) -> set[str]:
    if not os.path.exists(csv_path):
        return set()
    with open(csv_path, encoding="utf-8") as f:
        return {r["borehole_id"] for r in csv.DictReader(f)}


def _save_rows(rows: list[dict], csv_path: str, overwrite_bh: str | None) -> None:
    existing: list[dict] = []
    if overwrite_bh and os.path.exists(csv_path):
        with open(csv_path, encoding="utf-8") as f:
            existing = [r for r in csv.DictReader(f) if r["borehole_id"] != overwrite_bh]

    all_rows = existing + [{col: r.get(col, "") for col in CSV_COLS} for r in rows]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS)
        w.writeheader()
        w.writerows(all_rows)


def _append_rows(rows: list[dict], csv_path: str) -> None:
    exists = os.path.exists(csv_path)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
        if not exists:
            w.writeheader()
        for r in rows:
            w.writerow({col: r.get(col, "") for col in CSV_COLS})

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_rows(rows: list[dict]) -> None:
    print(f"\n  {'top_m':>6} {'bot_m':>6} {'mid_m':>6}  {'layer':5}  {'su_kpa':>6}  {'spt_n':>5}  {'uw':>6}  {'desc'}")
    print(f"  {'-'*6} {'-'*6} {'-'*6}  {'-'*5}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*40}")
    prev_bot = None
    for r in rows:
        top = r["depth_top_m"]
        bot = r["depth_bot_m"]
        mid = r["depth_m"]
        gap = ""
        if prev_bot is not None and abs(float(top) - float(prev_bot)) > 0.01:
            gap = f"  *** GAP {prev_bot}→{top} ***"
        su    = f"{r['su_kpa']:>6}" if r["su_kpa"]      != "" else "      "
        spt   = f"{r['spt_n']:>5}" if r["spt_n"]        != "" else "     "
        uw    = f"{r['unit_weight']:>6}" if r["unit_weight"] != "" else "      "
        desc  = str(r["soil_desc"])[:40]
        print(f"  {float(top):6.2f} {float(bot):6.2f} {float(mid):6.2f}  {r['soil_layer']:5}  {su}  {spt}  {uw}  {desc}{gap}")
        prev_bot = bot
    if rows:
        gaps = sum(
            1 for i in range(1, len(rows))
            if abs(float(rows[i]["depth_top_m"]) - float(rows[i-1]["depth_bot_m"])) > 0.01
        )
        print(f"\n  {len(rows)} layer rows  |  gaps between rows: {gaps}")


def main():
    parser = argparse.ArgumentParser(description="Extract Format-1 boring log PDFs.")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pdf",  help="Path to a single PDF")
    grp.add_argument("--all",  action="store_true", help="All PDFs in data/pdfs/")
    parser.add_argument("--resume", action="store_true",
                        help="Skip boreholes already in output CSV")
    parser.add_argument("--csv", default=CSV_PATH, help="Output CSV path")
    parser.add_argument("--show", action="store_true",
                        help="Print extracted rows to stdout")
    args = parser.parse_args()

    pdfs = ([args.pdf] if args.pdf else
            sorted(os.path.join(PDF_DIR, f)
                   for f in os.listdir(PDF_DIR) if f.lower().endswith(".pdf")))

    skip = _existing_boreholes(args.csv) if args.resume else set()
    total_rows = 0

    for pdf_path in pdfs:
        stem = os.path.splitext(os.path.basename(pdf_path))[0]
        if stem in skip:
            print(f"[{stem}] Already extracted — skipping")
            continue

        print(f"\n[{os.path.basename(pdf_path)}] Extracting...")
        try:
            rows = extract_pdf(pdf_path)
        except Exception as e:
            print(f"  ERROR: {e}")
            continue

        if not rows:
            print(f"  WARNING: no layers extracted — skipping")
            continue
        bh_id = rows[0]["borehole_id"]
        print(f"  Borehole: {bh_id}  |  E={rows[0]['easting']}  N={rows[0]['northing']}"
              f"  |  {len(rows)} layer intervals")

        if args.show or args.pdf:
            _print_rows(rows)

        if rows:
            _save_rows(rows, args.csv, overwrite_bh=bh_id)
            total_rows += len(rows)

    print(f"\n{'='*60}")
    print(f"Total: {total_rows} layer rows saved to {args.csv}")


if __name__ == "__main__":
    main()
