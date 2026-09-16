# -*- coding: utf-8 -*-
"""build_low_hit_marks3.pyの兄弟版。馬柱データ予想(c7416ae0)対象レースに、
2026-09-15再構築の馬連的中率3%未満台帳・3連複的中率3%未満台帳(いずれも現行290atom
FACTOR_GROUPS準拠)のうち「的中率0.0%」パターンだけを非冗長化した集合で、
「馬連外し」「3連複外し」マークを計算する。
現行FACTOR_GROUPSをそのまま使えるため、旧169atom版のようなラベル変換(eval_option_ext)は
不要で、jra_factor_registry.horse_qualifies()をそのまま使う。
"""
import json
import sys
from pathlib import Path

LIB_DIR = Path(r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model")
DATA_DIR = LIB_DIR.parent.parent / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_factor_registry as FR  # noqa: E402

OUT = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\umaren_sanrenpuku_low_hit_marks.json")


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


def compute_marks(races, patterns):
    marks = {}
    n_checked = n_matched = 0
    for race in races:
        race_id = race["race_id"]
        for h in race["horses"]:
            umaban = h.get("umaban")
            if umaban is None:
                continue
            n_checked += 1
            best, n_match = None, 0
            for p in patterns:
                sel = {gid: {oid} for gid, oid in p["selected"].items()}
                if FR.horse_qualifies(h, sel):
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


print("factor_database.json読み込み中...")
fdb = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
races = fdb["races"]
print(f"  {len(races)}レース")

ledger = json.loads((DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json").read_text(encoding="utf-8"))
umaren_rows = ledger["bet_types"]["馬連"]["rows"]
sanrenpuku_rows = ledger["bet_types"]["3連複"]["rows"]

umaren_pat, umaren_cand = non_redundant_roots(umaren_rows, lambda r: r["hit_rate_pct"] == 0.0)
sanrenpuku_pat, sanrenpuku_cand = non_redundant_roots(sanrenpuku_rows, lambda r: r["hit_rate_pct"] == 0.0)
print(f"馬連   root patterns: {len(umaren_pat)} / {umaren_cand} candidates / {len(umaren_rows)} total")
print(f"3連複  root patterns: {len(sanrenpuku_pat)} / {sanrenpuku_cand} candidates / {len(sanrenpuku_rows)} total")

out = {}
for label, pat in (("umaren", umaren_pat), ("sanrenpuku", sanrenpuku_pat)):
    marks, n_checked, n_matched = compute_marks(races, pat)
    print(f"[{label}] horses checked: {n_checked}, matched: {n_matched}")
    out[label] = marks

OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
print("wrote", OUT)
