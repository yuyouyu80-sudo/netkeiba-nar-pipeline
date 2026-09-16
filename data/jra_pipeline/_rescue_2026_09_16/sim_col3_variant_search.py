# -*- coding: utf-8 -*-
"""3連複フォーメーション3列目の基準を変えた場合の比較。
既存版: 3列目 = スコア5(BOX5モデル予測)>=30 のプール − 該当マーク除外
今回の比較対象:
  variant A (score5_30): スコア5>=30 (既存・ベースライン)
  variant B (score4_30): スコア4(BOX4モデル予測)>=30
  variant C (score4_40): スコア4>=40
1列目・2列目は既存のformation_mark_search.pyと完全に同じ(BOX4上位4頭ベース)。
各variantについて、単一マーク6状態×3列独立=216パターンを、全319レース+高/中/低確信度の
4母集団で実測する。
"""
import sys, json, csv, itertools, pathlib
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
LO_EDGE = vals[n // 3]
HI_EDGE = vals[(2 * n) // 3]
print(f"gap_top2 tercile edges: lo={LO_EDGE:.4f} hi={HI_EDGE:.4f} (n={n})")


def tier_of(race_id):
    v = gap_by_id.get(race_id)
    if v is None:
        return "mid"
    if v >= HI_EDGE:
        return "high"
    if v <= LO_EDGE:
        return "low"
    return "mid"


MARK_KEYS = ["t", "f", "w", "ur", "sr"]
MARK_LABEL = {"none": "マークなし", "t": "外し", "f": "複外", "w": "W外し", "ur": "馬連外し", "sr": "3連複外し"}
MARK_STATES = ["none", "t", "f", "w", "ur", "sr"]

races_out = []
tier_count = {"high": 0, "mid": 0, "low": 0}
for r in data2:
    race_id = r["race_id"]
    block = f"{r['kaisai_date']}_{r['racecourse']}"
    tier = tier_of(race_id)
    tier_count[tier] += 1
    horses = []
    for h in r["horses"]:
        umaban = h.get("umaban")
        key = f"{race_id}/{umaban}"
        rank4 = h.get("rank4")
        sc5 = h.get("score5")
        sc4 = h.get("score4")
        horses.append({
            "u": umaban,
            "b": bool(rank4 is not None and rank4 <= 4),
            "t": key in TANSHO, "f": key in FUKU, "w": key in WIDE,
            "ur": key in UMAREN, "sr": key in SANRENPUKU,
            "s5": (sc5 * 100 if sc5 is not None else None),
            "s4": (sc4 * 100 if sc4 is not None else None),
        })
    table = actual_by_id.get(race_id, {}).get("3連複", {})
    pay = {tuple(sorted(c)): p for c, p in table.items() if p > 0}
    races_out.append({"id": race_id, "blk": block, "tier": tier, "h": horses, "pay": pay})

print("tier counts:", tier_count, "total:", len(races_out))

VARIANTS = [
    ("score5_30", "s5", 30, "スコア5(BOX5モデル予測)>=30【既存】"),
    ("score4_30", "s4", 30, "スコア4(BOX4モデル予測)>=30"),
    ("score4_40", "s4", 40, "スコア4(BOX4モデル予測)>=40"),
]


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


def new_acc():
    return {"n_bet": 0, "hit_races": 0, "stake": 0, "return": 0}


def finalize(acc):
    n_bet, hit, stake, ret = acc["n_bet"], acc["hit_races"], acc["stake"], acc["return"]
    return {
        "n_bet": n_bet, "hit_races": hit,
        "hit_rate_pct": round(hit / n_bet * 100, 2) if n_bet else 0.0,
        "stake": stake, "return": ret,
        "return_rate_pct": round(ret / stake * 100, 2) if stake else 0.0,
    }


combos_total = list(itertools.product(MARK_STATES, repeat=3))
print(f"patterns per variant: {len(combos_total)}")

variant_results = {}
pool_stats = {}
for vname, field, thresh, label in VARIANTS:
    # プール規模の診断(マーク除外前、col3の母集団サイズ)
    pool_sizes = []
    empty_races = 0
    for r in races_out:
        n_pool = sum(1 for h in r["h"] if h[field] is not None and h[field] >= thresh)
        pool_sizes.append(n_pool)
        if n_pool == 0:
            empty_races += 1
    pool_stats[vname] = {
        "label": label, "field": field, "threshold": thresh,
        "avg_pool_size": round(sum(pool_sizes) / len(pool_sizes), 2),
        "min_pool_size": min(pool_sizes), "max_pool_size": max(pool_sizes),
        "empty_races": empty_races, "n_races": len(races_out),
    }
    print(f"\n=== variant {vname} ({label}) ===")
    print(f"  avg pool size: {pool_stats[vname]['avg_pool_size']}, "
          f"empty races: {empty_races}/{len(races_out)}")

    results = []
    for c1s, c2s, c3s in combos_total:
        accs = {"all": new_acc(), "high": new_acc(), "mid": new_acc(), "low": new_acc()}
        for r in races_out:
            c1, c2, c3 = col_sets(r["h"], c1s, c2s, c3s, field, thresh)
            combos = formation_combos(c1, c2, c3)
            if not combos:
                continue
            stake = len(combos) * 100
            ret = sum(r["pay"].get(k, 0) for k in combos)
            is_hit = ret > 0
            for pop in ("all", r["tier"]):
                a = accs[pop]
                a["n_bet"] += 1
                a["stake"] += stake
                a["return"] += ret
                if is_hit:
                    a["hit_races"] += 1
        results.append({
            "c1": c1s, "c2": c2s, "c3": c3s,
            "all": finalize(accs["all"]), "high": finalize(accs["high"]),
            "mid": finalize(accs["mid"]), "low": finalize(accs["low"]),
        })
    variant_results[vname] = results

    for pop in ("all", "high", "mid", "low"):
        cand = [r for r in results if r[pop]["n_bet"] >= 20]
        cand.sort(key=lambda r: -r[pop]["return_rate_pct"])
        print(f"  --- top3 (母集団={pop}, n_bet>=20) ---")
        for r in cand[:3]:
            p = r[pop]
            print(f"    1列={MARK_LABEL[r['c1']]:6s} 2列={MARK_LABEL[r['c2']]:6s} 3列={MARK_LABEL[r['c3']]:6s} "
                  f"n={p['n_bet']:3d} hit={p['hit_rate_pct']:5.1f}% return={p['return_rate_pct']:6.1f}%")

out = {
    "meta": {
        "n_races": len(races_out), "tier_count": tier_count,
        "gap_top2_edges": {"lo": LO_EDGE, "hi": HI_EDGE},
        "mark_label": MARK_LABEL, "variants": [{"name": v[0], "field": v[1], "threshold": v[2], "label": v[3]} for v in VARIANTS],
        "pool_stats": pool_stats,
    },
    "results": variant_results,
}
out_path = SP / "col3_variant_search.json"
out_path.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print(f"\nwrote {out_path} ({out_path.stat().st_size:,} bytes)")
