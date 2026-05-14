"""
Generate realistic dummy Bangkok boring log data for 10 boreholes.
Statistics based on Nguyen et al. 2023 and typical MRT Orange Line subsoil.
"""
import csv, os, sys, random
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from preprocess import derive_consistency

random.seed(7)
np.random.seed(7)

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "bangkok_boring_logs.csv")

CSV_COLS = [
    "borehole_id","easting","northing",
    "depth_m","depth_top_m","depth_bot_m",
    "soil_layer","soil_desc","consistency",
    "su_kpa","su_method","spt_n",
    "unit_weight","plasticity_idx","liquid_limit","plastic_limit",
    "water_content","source_file","notes",
]

# ── 10 boreholes spread across Bangkok UTM 47N ────────────────────────────────
BOREHOLES = [
    ("OW-01", 658871, 1522280),
    ("OW-02", 663200, 1519500),
    ("OW-03", 655100, 1526800),
    ("OW-04", 671300, 1513200),
    ("OW-05", 651800, 1532400),
    ("OW-06", 668500, 1528100),
    ("OW-07", 652400, 1515600),
    ("OW-08", 675200, 1521800),
    ("OW-09", 660100, 1510800),
    ("OW-10", 650300, 1537900),
]

# ── Nguyen et al. 2023 / Bangkok reference statistics ─────────────────────────
# (mean, std) unless noted
STATS = {
    # Very Soft Clay  (depth 2–7 m)
    "VSC": dict(
        su_base=10.0, su_grad=1.8, su_std=2.5,          # Su = base + grad*z ± std
        gamma=(14.9, 0.3),
        ll=(80, 8), pl=(24, 2), pi=(56, 7), wc=(95, 10),
    ),
    # Soft–Medium Clay (depth 7–14 m)
    "SOC": dict(
        su_base=22.0, su_grad=2.8, su_std=4.0,
        gamma=(15.6, 0.3),
        ll=(70, 7), pl=(23, 2), pi=(47, 6), wc=(72, 8),
    ),
    # Stiff–Very Stiff Clay (depth 14–26 m)
    "SC": dict(
        su_base=80.0, su_grad=7.0, su_std=12.0,         # ST samples
        spt_base=16,  spt_grad=1.5, spt_std=3.0,        # SS samples
        gamma=(18.2, 0.4),
        ll=(53, 6), pl=(20, 2), pi=(33, 5), wc=(33, 5),
    ),
    # Dense Sand — first layer (depth 26–42 m)
    "SS1": dict(
        spt_base=32, spt_grad=2.0, spt_std=5.0,
        gamma=(19.8, 0.3),
        wc=(17, 2),
    ),
    # Hard Clay (depth 40–56 m)
    "MSC": dict(
        spt_base=35, spt_grad=2.5, spt_std=7.0,
        gamma=(20.1, 0.2),
        ll=(40, 5), pl=(19, 2), pi=(21, 4), wc=(21, 3),
    ),
    # Firm Sand transition (thin)
    "FS": dict(
        spt_base=22, spt_std=5.0,
        gamma=(19.0, 0.3),
        wc=(20, 2),
    ),
    # Dense–Very Dense Sand — second layer (depth 52–70 m)
    "SS2": dict(
        spt_base=48, spt_grad=1.5, spt_std=7.0,
        gamma=(20.5, 0.2),
        wc=(15, 2),
    ),
}

def rnd(x, digits=1):
    return round(float(x), digits)

def gauss(mu, sigma, lo=None, hi=None):
    v = np.random.normal(mu, sigma)
    if lo is not None: v = max(lo, v)
    if hi is not None: v = min(hi, v)
    return v

def make_row(bh_id, easting, northing, depth, top, bot, layer, desc,
             su=None, su_method=None, spt=None,
             gamma=None, pi=None, ll=None, pl=None, wc=None, notes=""):
    su_kpa = rnd(su, 1) if su is not None else ""
    spt_n  = int(round(spt)) if spt is not None else ""
    consistency = derive_consistency(
        su if su is not None else None,
        spt if spt is not None else None,
        layer,
    ) or ""
    return {
        "borehole_id": bh_id, "easting": easting, "northing": northing,
        "depth_m": rnd(depth, 2), "depth_top_m": rnd(top, 2), "depth_bot_m": rnd(bot, 2),
        "soil_layer": layer, "soil_desc": desc, "consistency": consistency,
        "su_kpa": su_kpa, "su_method": su_method or "", "spt_n": spt_n,
        "unit_weight": rnd(gamma, 1) if gamma is not None else "",
        "plasticity_idx": int(round(pi)) if pi is not None else "",
        "liquid_limit":   int(round(ll)) if ll is not None else "",
        "plastic_limit":  int(round(pl)) if pl is not None else "",
        "water_content":  int(round(wc)) if wc is not None else "",
        "source_file": f"{bh_id}.pdf", "notes": notes,
    }

def atterberg(key):
    s = STATS[key]
    ll = gauss(*s["ll"], lo=s["ll"][0]-20, hi=s["ll"][0]+25)
    pl = gauss(*s["pl"], lo=12, hi=30)
    pi = ll - pl
    wc = gauss(*s["wc"], lo=5)
    return ll, pl, pi, wc

# ── Generate rows ─────────────────────────────────────────────────────────────
all_rows = []

for bh_id, easting, northing in BOREHOLES:
    # Slightly randomise layer boundaries per borehole
    fill_bot = rnd(random.uniform(1.5, 3.0), 1)
    vsc_bot  = rnd(random.uniform(5.5, 8.5),  1)
    soc_bot  = rnd(random.uniform(11.5, 15.5), 1)
    sc_bot   = rnd(random.uniform(22.0, 27.0), 1)
    ss1_bot  = rnd(random.uniform(38.0, 44.0), 1)
    msc_bot  = rnd(random.uniform(50.0, 57.0), 1)
    fs_bot   = rnd(msc_bot + random.uniform(1.0, 2.5), 1)
    total    = rnd(random.uniform(60.0, 68.0), 1)

    rows = []

    # MG fill ─────────────────────────────────────────────────────────────────
    gamma = gauss(17.2, 0.5, lo=16.0, hi=19.0)
    rows.append(make_row(
        bh_id, easting, northing,
        fill_bot/2, 0.0, fill_bot,
        "MG", "Fill Material / Made Ground", gamma=gamma,
        spt=gauss(8, 3, lo=3, hi=20),
        wc=gauss(30, 6, lo=15),
    ))

    # VSC — ST samples every ~1.5 m ───────────────────────────────────────────
    d = fill_bot + 0.75
    while d <= vsc_bot + 0.1:
        z = d - fill_bot
        s = STATS["VSC"]
        su  = gauss(s["su_base"] + s["su_grad"]*z, s["su_std"], lo=5, hi=30)
        g   = gauss(*s["gamma"], lo=13.5, hi=16.0)
        ll, pl, pi, wc = atterberg("VSC")
        rows.append(make_row(
            bh_id, easting, northing, d, d-0.45, d+0.45,
            "VSC", "Very soft to soft CLAY grey high plasticity (CH)",
            su=su, su_method="ST", gamma=g,
            pi=pi, ll=ll, pl=pl, wc=wc,
        ))
        d += 1.5

    # SOC — ST samples every ~1.5 m ───────────────────────────────────────────
    d = vsc_bot + 0.75
    while d <= soc_bot + 0.1:
        z = d - vsc_bot
        s = STATS["SOC"]
        su  = gauss(s["su_base"] + s["su_grad"]*z, s["su_std"], lo=15, hi=65)
        g   = gauss(*s["gamma"], lo=14.5, hi=17.0)
        ll, pl, pi, wc = atterberg("SOC")
        rows.append(make_row(
            bh_id, easting, northing, d, d-0.45, d+0.45,
            "SOC", "Soft to medium CLAY grey high plasticity (CH)",
            su=su, su_method="ST", gamma=g,
            pi=pi, ll=ll, pl=pl, wc=wc,
        ))
        d += 1.5

    # SC — alternate ST (su) and SS (SPT) every 1.5 m ────────────────────────
    d = soc_bot + 0.75
    toggle = 0
    while d <= sc_bot + 0.1:
        z = d - soc_bot
        s = STATS["SC"]
        g = gauss(*s["gamma"], lo=16.0, hi=20.5)
        ll, pl, pi, wc = atterberg("SC")
        if toggle % 2 == 0:   # ST
            su = gauss(s["su_base"] + s["su_grad"]*z, s["su_std"], lo=40, hi=250)
            rows.append(make_row(
                bh_id, easting, northing, d, d-0.45, d+0.45,
                "SC", "Stiff to very stiff CLAY light grey to grey high plasticity (CH)",
                su=su, su_method="ST", gamma=g,
                pi=pi, ll=ll, pl=pl, wc=wc,
            ))
        else:                  # SS
            spt = gauss(s["spt_base"] + s["spt_grad"]*z, s["spt_std"], lo=8, hi=50)
            rows.append(make_row(
                bh_id, easting, northing, d, d-0.3, d+0.3,
                "SC", "Stiff to very stiff CLAY light grey to grey high plasticity (CH)",
                spt=spt, gamma=g,
                pi=pi, ll=ll, pl=pl, wc=wc,
                notes="SS sample in stiff clay",
            ))
        toggle += 1
        d += 1.5

    # First SS — dense sand, SPT every 1.5 m ─────────────────────────────────
    d = sc_bot + 0.75
    while d <= ss1_bot + 0.1:
        z = d - sc_bot
        s = STATS["SS1"]
        spt = gauss(s["spt_base"] + s["spt_grad"]*z, s["spt_std"], lo=20, hi=100)
        g   = gauss(*s["gamma"], lo=18.5, hi=21.5)
        wc  = gauss(*s["wc"], lo=8)
        rows.append(make_row(
            bh_id, easting, northing, d, d-0.3, d+0.3,
            "SS", "Dense to very dense silty SAND grey fine to medium grained (SM)",
            spt=spt, gamma=g, wc=wc,
        ))
        d += 1.5

    # MSC — hard clay, SPT every 1.5 m ───────────────────────────────────────
    d = ss1_bot + 0.75
    while d <= msc_bot + 0.1:
        z = d - ss1_bot
        s = STATS["MSC"]
        spt = gauss(s["spt_base"] + s["spt_grad"]*z, s["spt_std"], lo=25, hi=100)
        g   = gauss(*s["gamma"], lo=19.0, hi=21.5)
        ll, pl, pi, wc = atterberg("MSC")
        rows.append(make_row(
            bh_id, easting, northing, d, d-0.3, d+0.3,
            "MSC", "Hard silty CLAY light grey to grey low plasticity (CL)",
            spt=spt, gamma=g,
            pi=pi, ll=ll, pl=pl, wc=wc,
        ))
        d += 1.5

    # FS — thin transition, single sample ─────────────────────────────────────
    s = STATS["FS"]
    rows.append(make_row(
        bh_id, easting, northing,
        (msc_bot+fs_bot)/2, msc_bot, fs_bot,
        "FS", "Firm SAND / Sandy CLAY transition (SM-SC)",
        spt=gauss(s["spt_base"], s["spt_std"], lo=10, hi=45),
        gamma=gauss(*s["gamma"], lo=18.0, hi=20.5),
        wc=gauss(*s["wc"], lo=8),
        notes="Transition layer",
    ))

    # Second SS — very dense sand, SPT every 1.5 m ───────────────────────────
    d = fs_bot + 0.75
    while d <= total + 0.1:
        z = d - fs_bot
        s = STATS["SS2"]
        spt = gauss(s["spt_base"] + s["spt_grad"]*z, s["spt_std"], lo=35, hi=100)
        g   = gauss(*s["gamma"], lo=19.5, hi=21.5)
        wc  = gauss(*s["wc"], lo=5)
        rows.append(make_row(
            bh_id, easting, northing, d, d-0.3, d+0.3,
            "SS", "Dense to very dense SAND with silt grey fine to medium grained (SM SP-SM)",
            spt=spt, gamma=g, wc=wc,
        ))
        d += 1.5

    print(f"  {bh_id}: {len(rows)} rows  "
          f"(fill {fill_bot}m | soft clay {vsc_bot}/{soc_bot}m | "
          f"stiff {sc_bot}m | sand1 {ss1_bot}m | hard {msc_bot}m | "
          f"fs {fs_bot}m | total {total}m)")
    all_rows.extend(rows)

# ── Write CSV ─────────────────────────────────────────────────────────────────
with open(OUT, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=CSV_COLS, extrasaction="ignore")
    w.writeheader()
    w.writerows(all_rows)

print(f"\nWrote {len(all_rows)} rows across {len(BOREHOLES)} boreholes -> {OUT}")
