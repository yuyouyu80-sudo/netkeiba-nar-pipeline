# -*- coding: utf-8 -*-
"""col3_variant_search.jsonのベースライン+上位候補について、3variant(score5_30/score4_30/
score4_40)×4母集団でブロックブートストラップ95%CIを計算する。"""
import sys, json, csv, pathlib
import numpy as np
sys.path.insert(0, r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
import jra_dataset

SP = pathlib.Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad")
PROJECT = pathlib.Path(r"C:\Users\yuyou\Desktop\新しい作業場所")

data = jra_dataset.load(rebuild=False)
races_ds, actual = data["races"], data["actual"]
actual_by_id = {r["race_id"]: actual.get(r["race_id"], {}) for r in races_ds}

data2 = json.loads((SP / "data2_races.json").read_text(encoding="utf-8"))
marks5 = json.loads((SP / "low_hit_marks5.json").read_text(encoding="utf-8"))
TANSHO, FUKU, WIDE = marks5["tansho"], marks5["fuku"], marks5["wide"]
UMAREN, SANRENPUKU = marks5["umaren"], marks5["sanrenpuku"]

conf_path = PROJECT / "data" / "jra_pipeline" / "confidence_per_race.csv"
with open(conf_path, encoding="utf-8-sig") as f:
    conf_rows = list(csv.DictReader(f))
target_ids = {r["race_id"] for r in data2}
gap_by_id = {r["race_id"]: float(r["gap_top2"]) for r in conf_rows if r["race_id"] in target_ids}
vals = sorted(gap_by_id.values())
n = len(vals)
LO_EDGE, HI_EDGE = vals[n // 3], vals[(2 * n) // 3]


def tier_of(race_id):
    v = gap_by_id.get(race_id)
    if v is None:
        return "mid"
    if v >= HI_EDGE:
        return "high"
    if v <= LO_EDGE:
        return "low"
    return "mid"


MARK_LABEL = {"none": "マークなし", "t": "外し", "f": "複外", "w": "W外し", "ur": "馬連外し", "sr": "3連複外し"}

races_out = []
for r in data2:
    race_id = r["race_id"]
    block = f"{r['kaisai_date']}_{r['racecourse']}"
    tier = tier_of(race_id)
    horses = []
    for h in r["horses"]:
        umaban = h.get("umaban")
        key = f"{race_id}/{umaban}"
        rank4 = h.get("rank4")
        sc5 = h.get("score5")
        sc4 = h.get("score4")
        horses.append({
            "u": umaban, "b": bool(rank4 is not None and rank4 <= 4),
            "t": key in TANSHO, "f": key in FUKU, "w": key in WIDE,
            "ur": key in UMAREN, "sr": key in SANRENPUKU,
            "s5": (sc5 * 100 if sc5 is not None else None),
            "s4": (sc4 * 100 if sc4 is not None else None),
        })
    table = actual_by_id.get(race_id, {}).get("3連複", {})
    pay = {tuple(sorted(c)): p for c, p in table.items() if p > 0}
    races_out.append({"id": race_id, "blk": block, "tier": tier, "h": horses, "pay": pay})


def col_sets(horses, c1s, c2s, c3s, score_field, score_min):
    c1, c2, c3 = set(), set(), set()
    for h in horses:
        if h["b"] or (c1s != "none" and h[c1s]):
            c1.add(h["u"])
    for h in horses:
        if not h["b"]:
            continue
        if c2s != "none" and h[c2s]:
            continue
        c2.add(h["u"])
    for h in horses:
        v = h[score_field]
        if v is None or v < score_min:
            continue
        if c3s != "none" and h[c3s]:
            continue
        c3.add(h["u"])
    return c1, c2, c3


def formation_combos(c1, c2, c3):
    combos = set()
    for x in c1:
        for y in c2:
            if y == x:
                continue
            for z in c3:
                if z == x or z == y:
                    continue
                combos.add(tuple(sorted((x, y, z))))
    return combos


def block_bootstrap_ci(stakes_by_block, rets_by_block, n_boot=3000, seed=2026):
    blocks = list(stakes_by_block.keys())
    stakes = np.array([stakes_by_block[b] for b in blocks], dtype=float)
    rets = np.array([rets_by_block[b] for b in blocks], dtype=float)
    nB = len(blocks)
    rng = np.random.default_rng(seed)
    rates = []
    for _ in range(n_boot):
        idx = rng.integers(0, nB, size=nB)
        s, r = stakes[idx].sum(), rets[idx].sum()
        rates.append(r / s * 100 if s > 0 else 0.0)
    rates.sort()
    lo = rates[int(0.025 * len(rates))]
    hi = rates[int(0.975 * len(rates))]
    return lo, hi, nB


def evaluate(c1s, c2s, c3s, pop, score_field, score_min):
    stakes_b, rets_b = {}, {}
    n_bet = hit_races = stake_sum = ret_sum = 0
    for r in races_out:
        if pop != "all" and r["tier"] != pop:
            continue
        c1, c2, c3 = col_sets(r["h"], c1s, c2s, c3s, score_field, score_min)
        combos = formation_combos(c1, c2, c3)
        if not combos:
            continue
        stake = len(combos) * 100
        ret = sum(r["pay"].get(k, 0) for k in combos)
        n_bet += 1
        stake_sum += stake
        ret_sum += ret
        if ret > 0:
            hit_races += 1
        b = r["blk"]
        stakes_b[b] = stakes_b.get(b, 0) + stake
        rets_b[b] = rets_b.get(b, 0) + ret
    hit_rate = hit_races / n_bet * 100 if n_bet else 0.0
    return_rate = ret_sum / stake_sum * 100 if stake_sum else 0.0
    ci_lo, ci_hi, nB = block_bootstrap_ci(stakes_b, rets_b)
    return {
        "n_bet": n_bet, "hit_races": hit_races, "hit_rate_pct": round(hit_rate, 2),
        "stake": stake_sum, "return": ret_sum, "return_rate_pct": round(return_rate, 2),
        "ci_lo": round(ci_lo, 1), "ci_hi": round(ci_hi, 1), "n_blocks": nB,
    }


# variant定義 + baseline/top1候補(sim_col3_variant_search.pyの出力から採取)
VARIANTS = [
    ("score5_30", "s5", 30, "スコア5(BOX5モデル予測)>=30【既存】"),
    ("score4_30", "s4", 30, "スコア4(BOX4モデル予測)>=30"),
    ("score4_40", "s4", 40, "スコア4(BOX4モデル予測)>=40"),
]

CANDIDATES_BY_VARIANT = {
    "score5_30": [
        ("baseline (none/none/none)", "none", "none", "none"),
        ("all-top1 (none/w/t)", "none", "w", "t"),
        ("high-top1 (t/w/sr)", "t", "w", "sr"),
        ("mid-top1 (none/w/none)", "none", "w", "none"),
        ("low-top1 (none/t/w)", "none", "t", "w"),
    ],
    "score4_30": [
        ("baseline (none/none/none)", "none", "none", "none"),
        ("all-top1 (none/t/w)", "none", "t", "w"),
        ("high-top1 (none/w/ur)", "none", "w", "ur"),
        ("mid-top1 (none/w/none)", "none", "w", "none"),
        ("low-top1 (none/t/w)", "none", "t", "w"),
    ],
    "score4_40": [
        ("baseline (none/none/none)", "none", "none", "none"),
        ("all-top1 (none/t/w)", "none", "t", "w"),
        ("high-top1 (t/w/sr)", "t", "w", "sr"),
        ("mid-top1 (none/sr/none)", "none", "sr", "none"),
        ("low-top1 (none/t/w)", "none", "t", "w"),
    ],
}


def label(m):
    return MARK_LABEL[m]


out = {}
for vname, field, thresh, vlabel in VARIANTS:
    out[vname] = {"label": vlabel, "field": field, "threshold": thresh, "candidates": []}
    print(f"\n########## variant {vname} ({vlabel}) ##########")
    for lbl, c1s, c2s, c3s in CANDIDATES_BY_VARIANT[vname]:
        row = {"label": lbl, "c1": c1s, "c2": c2s, "c3": c3s,
               "c1_label": label(c1s), "c2_label": label(c2s), "c3_label": label(c3s)}
        for pop in ("all", "high", "mid", "low"):
            row[pop] = evaluate(c1s, c2s, c3s, pop, field, thresh)
        out[vname]["candidates"].append(row)
        print(f"\n=== {lbl} ===")
        for pop in ("all", "high", "mid", "low"):
            p = row[pop]
            print(f"  [{pop:4s}] n={p['n_bet']:3d} hit={p['hit_rate_pct']:5.1f}% return={p['return_rate_pct']:6.1f}% "
                  f"CI=[{p['ci_lo']:.1f}%, {p['ci_hi']:.1f}%] (blocks={p['n_blocks']})")

(SP / "col3_variant_diagnosis.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("\nwrote col3_variant_diagnosis.json")
