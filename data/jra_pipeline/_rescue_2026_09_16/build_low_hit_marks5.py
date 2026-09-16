# -*- coding: utf-8 -*-
"""build_low_hit_marks3.pyの拡張版。「馬柱データ2」対象の全319レースに対して、単勝(外し)・
複勝(複外)・ワイド(W外し)・馬連(馬連外し、新規)・3連複(3連複外し、新規)の5種類の
低的中率パターン一致マークをまとめて計算する。
出力: low_hit_marks5.json  { "tansho": {...}, "fuku": {...}, "wide": {...}, "umaren": {...}, "sanrenpuku": {...} }
"""
import sys, json
sys.path.insert(0, r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad")
from factor_engine import load_data

SP = r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"


def eval_option_ext(horse, group, opt):
    v = horse.get(group["source"])
    if v is None:
        return False
    kind = group["kind"]
    if kind == "rank_le":
        try:
            return v <= opt["params"]["le"]
        except TypeError:
            return False
    if kind == "threshold_ge":
        try:
            return v >= opt["params"]["ge"]
        except TypeError:
            return False
    if kind == "threshold_lt":
        try:
            return v < opt["params"]["lt"]
        except TypeError:
            return False
    if kind == "category_in":
        return v in opt["params"]["in"]
    return False


def horse_qualifies_ext(horse, selected, factor_groups):
    for gid, opt_id in selected.items():
        group = factor_groups[gid]
        opt = next((o for o in group["options"] if o["id"] == opt_id), None)
        if opt is None or not eval_option_ext(horse, group, opt):
            return False
    return True


def non_redundant_roots(rows, keep_fn):
    cand = [r for r in rows if keep_fn(r)]
    cand.sort(key=lambda r: (len(r["selected"]), -r["n_races"]))
    patterns, root_sets = [], []
    for r in cand:
        s = frozenset(r["selected"].items())
        if any(rs.issubset(s) for rs in root_sets):
            continue
        patterns.append(r)
        root_sets.append(s)
    return patterns, len(cand)


def compute_marks(target_races, patterns, fg):
    marks = {}
    n_checked = n_matched = 0
    for race in target_races:
        race_id = race["race_id"]
        for h in race["horses"]:
            umaban = h.get("umaban")
            if umaban is None:
                continue
            n_checked += 1
            best, n_match = None, 0
            for p in patterns:
                if horse_qualifies_ext(h, p["selected"], fg):
                    n_match += 1
                    if best is None or (p["hit_rate_pct"], p["return_rate_pct"]) < (best["hit_rate_pct"], best["return_rate_pct"]):
                        best = p
            if best is not None:
                n_matched += 1
                marks[f"{race_id}/{umaban}"] = {
                    "n_match": n_match,
                    "best_hit_rate_pct": round(best["hit_rate_pct"], 2),
                    "best_return_rate_pct": round(best["return_rate_pct"], 2),
                    "best_n_races": best["n_races"],
                    "best_conditions": [f"{c['group']}: {c['value']}" for c in best["conditions"]],
                }
    return marks, n_checked, n_matched


print("loading artifact1_data.json ...")
data = load_data(SP + r"\artifact1_data.json")
fg = dict(data["factor_groups"])
universe = json.loads(open(SP + r"\low_search_universe.json", encoding="utf-8").read())
fg.update(universe["new_groups"])

data2_races = json.loads(open(SP + r"\data2_races.json", encoding="utf-8").read())
target_ids = {r["race_id"] for r in data2_races}
target_races = [r for r in data["races"] if r["race_id"] in target_ids]
print(f"data2 races: {len(target_ids)}, matched in artifact1_data: {len(target_races)}")
missing = target_ids - {r["race_id"] for r in target_races}
if missing:
    print(f"WARNING: {len(missing)} data2 race_id(s) not found in artifact1_data.json: {sorted(missing)[:5]}...")

tansho_rows = json.loads(open(SP + r"\tansho_low_report_data.json", encoding="utf-8").read())["rows"]
fuku_rows = json.loads(open(SP + r"\fuku_low_report_data.json", encoding="utf-8").read())["rows"]
wide_rows = json.loads(open(SP + r"\wide_low_report_data.json", encoding="utf-8").read())["rows"]
umaren_rows = json.loads(open(SP + r"\umaren_low_report_data.json", encoding="utf-8").read())["rows"]
sanrenpuku_rows = json.loads(open(SP + r"\sanrenpuku_low_report_data.json", encoding="utf-8").read())["rows"]

tansho_pat, tansho_cand = non_redundant_roots(tansho_rows, lambda r: r["hit_rate_pct"] == 0.0)
fuku_pat, fuku_cand = non_redundant_roots(fuku_rows, lambda r: r["hit_rate_pct"] < 5.0)
wide_pat, wide_cand = non_redundant_roots(wide_rows, lambda r: r["hit_rate_pct"] == 0.0)
umaren_pat, umaren_cand = non_redundant_roots(umaren_rows, lambda r: r["hit_rate_pct"] == 0.0)
sanrenpuku_pat, sanrenpuku_cand = non_redundant_roots(sanrenpuku_rows, lambda r: r["hit_rate_pct"] == 0.0)
print(f"tansho     root patterns: {len(tansho_pat)} / {tansho_cand} candidates / {len(tansho_rows)} total")
print(f"fuku       root patterns: {len(fuku_pat)} / {fuku_cand} candidates / {len(fuku_rows)} total")
print(f"wide       root patterns: {len(wide_pat)} / {wide_cand} candidates / {len(wide_rows)} total")
print(f"umaren     root patterns: {len(umaren_pat)} / {umaren_cand} candidates / {len(umaren_rows)} total")
print(f"sanrenpuku root patterns: {len(sanrenpuku_pat)} / {sanrenpuku_cand} candidates / {len(sanrenpuku_rows)} total")

out = {}
for label, pat in (("tansho", tansho_pat), ("fuku", fuku_pat), ("wide", wide_pat),
                   ("umaren", umaren_pat), ("sanrenpuku", sanrenpuku_pat)):
    marks, n_checked, n_matched = compute_marks(target_races, pat, fg)
    print(f"[{label}] horses checked: {n_checked}, matched: {n_matched}")
    out[label] = marks

with open(SP + r"\low_hit_marks5.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print("wrote low_hit_marks5.json")
