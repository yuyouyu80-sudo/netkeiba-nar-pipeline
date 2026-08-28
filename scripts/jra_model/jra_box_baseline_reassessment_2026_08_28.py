# -*- coding: utf-8 -*-
"""JRA Stage2 Phase J0(5): jra_box_baseline_reassessment_2026_08_22.pyの246レース版・
ペアCI追加版。

2026-08-22版は現行本番モデルのfit母集団(105レース)を除外したheld-out評価まで行ったが、
held-out集合の絶対回収率のCI(block_bootstrap)しか出しておらず、「市場を統計的に上回るか」
を判定できるペア差分CI(block_bootstrap_diff、model picks vs market picks)が無かった
(NAR Stage1で発見した「絶対回収率のCIは市場超過ゲートに使えない」のと同型の不備)。

本スクリプトは同じロジックを現時点の246レース(2026-08-22時点の211から32日→14日分・
35レース増)に適用し、held-out集合上でmodel vs marketのペア差分95%CIを追加する。
既存jra_eval.py/jra_box_baseline_reassessment_2026_08_22.pyは無改造で参照するのみ。
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
NAMES = JS.LEGACY_SIGNALS
OUT_JSON = DATA_DIR / "jra_box_baseline_reassessment_2026_08_28_result.json"
OUT_TXT = DATA_DIR / "jra_box_baseline_reassessment_2026_08_28_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
log(f"races={len(races)}(2026-08-22版は211、35レース増)")


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


results_by_box = {}
for BOX_N in BOX_NS:
    log(f"\n{'=' * 60}\nbox_n={BOX_N}\n{'=' * 60}")
    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W = wvec(winner["weights"])
    fitted_on = winner["fitted_on"]

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    picks = JE.score_picks(mats_all, W, BOX_N)
    market_picks = JE.market_picks(races, BOX_N)

    r_all = ev.evaluate(picks)
    held_out_blocks = JE.held_out_block_subset(fitted_on, races)
    held_out_idx = np.where(np.isin(ev.blocks, held_out_blocks))[0]
    r_held_out = ev.evaluate(picks, idx=held_out_idx)
    boot_held_out_abs = ev.block_bootstrap(picks, n=2000, seed=9100 + BOX_N,
                                           block_subset=held_out_blocks)
    # 2026-08-28追加: held-out集合上でのmodel vs marketペア差分95%CI(市場超過ゲート本体)。
    boot_held_out_vs_market = ev.block_bootstrap_diff(
        picks, market_picks, n=2000, seed=9200 + BOX_N, block_subset=held_out_blocks)

    breakeven = JE.breakeven_pct(BOX_N)

    log(f"  fit母集団: {fitted_on['search_dates'] + fitted_on['holdout_dates']} "
        f"({fitted_on['n_races']}レース)")
    log(f"  held-outブロック数={len(held_out_blocks)}  held-outレース数={len(held_out_idx)}")
    log(f"  理論ブレークイーブン({'+'.join(JE.OBJ_BETS)})={breakeven:.2f}%")
    log(f"  (a) 全{len(races)}レース: model={r_all['model']:.2f}%  market={r_all['market']:.2f}%  "
        f"市場差={r_all['excess']:+.2f}pt")
    log(f"  (b) held-out{len(held_out_idx)}レース: model={r_held_out['model']:.2f}%  "
        f"market={r_held_out['market']:.2f}%  市場差={r_held_out['excess']:+.2f}pt")
    log(f"      絶対回収率95%CI=[{boot_held_out_abs['lo']:.2f},{boot_held_out_abs['hi']:.2f}]%"
        f"(参考値、市場超過ゲートには使わない)")
    log(f"      model vs marketペア差分95%CI=[{boot_held_out_vs_market['lo']:+.2f},"
        f"{boot_held_out_vs_market['hi']:+.2f}]  n_blocks={boot_held_out_vs_market['n_blocks']}  "
        f"{'市場超過が統計的に有意' if boot_held_out_vs_market['lo'] > 0 else '非有意(0を跨ぐか負)'}")

    results_by_box[BOX_N] = {
        "model_file": WINNER_FILES[BOX_N],
        "fitted_on_dates": fitted_on["search_dates"] + fitted_on["holdout_dates"],
        "fitted_on_n_races": fitted_on["n_races"],
        "n_held_out_blocks": len(held_out_blocks),
        "n_held_out_races": int(len(held_out_idx)),
        "breakeven_pct": breakeven,
        "all_races": {"model": r_all["model"], "market": r_all["market"], "excess": r_all["excess"]},
        "held_out_only": {
            "model": r_held_out["model"], "market": r_held_out["market"],
            "excess": r_held_out["excess"],
            "bootstrap_ci_absolute": boot_held_out_abs,
            "bootstrap_ci_vs_market": boot_held_out_vs_market,
            "above_breakeven": bool(r_held_out["model"] > breakeven),
            "market_excess_significant": bool(boot_held_out_vs_market["lo"] > 0),
        },
    }

OUT_JSON.write_text(json.dumps({
    "n_races": len(races),
    "note": "jra_box_baseline_reassessment_2026_08_22.pyの246レース版。held_out_onlyの"
            "bootstrap_ci_vs_marketが新規追加(model vs marketのペア差分CI、市場超過ゲート本体)。"
            "bootstrap_ci_absoluteは参考値でありゲートに使わない(NAR Stage1で確認済みの"
            "絶対CI≠市場超過ゲートの原則)。",
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\nwrote {OUT_JSON}")
log(f"wrote {OUT_TXT}")
