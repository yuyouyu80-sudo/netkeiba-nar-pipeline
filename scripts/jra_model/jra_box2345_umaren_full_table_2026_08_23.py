# -*- coding: utf-8 -*-
"""box2-5馬連探索の候補パターンについて、全券種(単勝/複勝/枠連/馬連/ワイド/馬単/3連複/3連単)の
的中率・回収率をfull_table()で計算し、現行モデル(またはbox2は市場)と横並び比較するJSONを出力する。
既存のjra_search_box2345_umaren_cap_2026_08_23_result.jsonの重みを再利用し、追加の探索は行わない
(参考記録の可視化専用、値は変更しない)。"""
import json
import sys
from pathlib import Path

import numpy as np

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

data = jra_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS
dfs = [r["df"] for r in races]
priors = JS.make_priors(dfs)
mats = JS.signal_matrices(races, priors, NAMES, JS.CLASS_ORDINAL)

search = json.loads((DATA_DIR / "jra_search_box2345_umaren_cap_2026_08_23_result.json").read_text(encoding="utf-8"))

PRODUCTION_FILES = {3: "winner_box3.json", 4: "winner_box4.json", 5: "winner_v3.json"}


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


out = {}
for box_n_str, r in search["results_by_box"].items():
    box_n = int(box_n_str)
    ev = JE.Evaluator(races, actual, box_n=box_n)

    cand_w = wvec(r["best_full_population"]["weights"])
    cand_picks = JE.score_picks(mats, cand_w, box_n)
    cand_table = ev.full_table(cand_picks)

    if box_n in PRODUCTION_FILES:
        winner = json.loads((DATA_DIR / PRODUCTION_FILES[box_n]).read_text(encoding="utf-8"))
        base_w = wvec(winner["weights"])
        base_picks = JE.score_picks(mats, base_w, box_n)
        base_label = f"現行モデル(BOX{box_n}, pattern{winner['pattern_id']})"
    else:
        base_picks = JE.market_picks(races, box_n)
        base_label = f"市場(上位{box_n}人気BOX)"
    base_table = ev.full_table(base_picks)

    rows = []
    for bt in ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]:
        b = base_table[base_table["bet_type"] == bt].iloc[0]
        c = cand_table[cand_table["bet_type"] == bt].iloc[0]
        rows.append({
            "bet_type": bt,
            "base_hits": int(b["hit_races"]), "base_races": int(b["races"]),
            "base_hit_pct": float(b["hit_rate_pct"]), "base_return_pct": float(b["return_rate_pct"]),
            "cand_hits": int(c["hit_races"]), "cand_races": int(c["races"]),
            "cand_hit_pct": float(c["hit_rate_pct"]), "cand_return_pct": float(c["return_rate_pct"]),
        })
    out[box_n] = {
        "base_label": base_label,
        "cand_label": f"馬連探索候補(pattern#{r['best_full_population']['pattern_index']}・不採用/参考)",
        "rows": rows,
    }
    print(f"box_n={box_n}  base={base_label}  cand=pattern#{r['best_full_population']['pattern_index']}")
    for row in rows:
        print(f"  {row['bet_type']:>4s}  base {row['base_hits']:3d}/{row['base_races']} ({row['base_hit_pct']:5.1f}%) "
              f"{row['base_return_pct']:7.1f}%   cand {row['cand_hits']:3d}/{row['cand_races']} "
              f"({row['cand_hit_pct']:5.1f}%) {row['cand_return_pct']:7.1f}%")

OUT = DATA_DIR / "jra_box2345_umaren_full_table_2026_08_23.json"
OUT.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print("\nwrote", OUT.name)
