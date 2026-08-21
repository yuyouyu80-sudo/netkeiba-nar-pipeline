# -*- coding: utf-8 -*-
"""Step0(2026-08-22): BOX側評価基盤の是正。

これまでのbox5/4/3探索レポート(jra_search500_2026_08_21_v3signals.py等)は「現行モデル、
全211レースで直接評価: 市場差+16.58pt」を比較対象のヘッドライン値として使ってきたが、
winner_v3.json/winner_box4.json/winner_box3.jsonの`fitted_on`を見ると、この重みは105レース
(2026-07-11/07-12/07-18を探索、07-19/07-25/07-26をholdout)でfitされており、この6日は
現在の211レース評価用データセットに丸ごと含まれる(2026-08-22、Opus 5サブエージェントの
調査により発見、本スクリプトの実行前に3ファイルとも当該日付一致を直接確認済み)。つまり
ヘッドライン値の約半分がin-sampleの数字であり、公正な比較対象ではなかった。軸流し側の
レポート(jra_axis_search500_2026_08_21_v3signals.py)は既にfit母集団除外評価を行っていたが、
BOX側だけ抜けていた。

本スクリプトは、fit母集団と重複しない真に未見のブロックだけで現行モデルを再評価し、
理論ブレークイーブン回収率(jra_eval.breakeven_pct、控除率から算出)と併記することで、
「今後の正しい現状認識」を1箇所にまとめる。既存の5本の探索レポートのREJECTED判定自体は
current modelとのblock_bootstrap_diffに基づいており、ヘッドライン数値の訂正では覆らないため
遡って書き換えない。

出力: data/jra_pipeline/jra_box_baseline_reassessment_2026_08_22_result.json
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
OUT_JSON = DATA_DIR / "jra_box_baseline_reassessment_2026_08_22_result.json"

data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
print(f"races={len(races)}")


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


results_by_box = {}

for BOX_N in BOX_NS:
    print(f"\n{'=' * 60}\nbox_n={BOX_N}\n{'=' * 60}")
    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W = wvec(winner["weights"])
    fitted_on = winner["fitted_on"]

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    picks = JE.score_picks(mats_all, W, BOX_N)

    # (a) 全211レース(既存ヘッドライン値、参考として残す)
    r_all = ev.evaluate(picks)

    # (b) fit母集団と重複しない真に未見のブロックだけで再評価
    held_out_blocks = JE.held_out_block_subset(fitted_on, races)
    held_out_idx = np.where(np.isin(ev.blocks, held_out_blocks))[0]
    r_held_out = ev.evaluate(picks, idx=held_out_idx)
    boot_held_out = ev.block_bootstrap(picks, n=2000, seed=9100 + BOX_N, block_subset=held_out_blocks)

    breakeven = JE.breakeven_pct(BOX_N)

    print(f"  fit母集団: {fitted_on['search_dates'] + fitted_on['holdout_dates']} "
        f"({fitted_on['n_races']}レース)")
    print(f"  held-outブロック数={len(held_out_blocks)}  held-outレース数={len(held_out_idx)}")
    print(f"  理論ブレークイーブン({'+'.join(JE.OBJ_BETS)})={breakeven:.2f}%")
    print(f"  (a) 全211レース: model={r_all['model']:.2f}%  market={r_all['market']:.2f}%  "
        f"市場差={r_all['excess']:+.2f}pt")
    print(f"  (b) held-out{len(held_out_idx)}レース: model={r_held_out['model']:.2f}%  "
        f"market={r_held_out['market']:.2f}%  市場差={r_held_out['excess']:+.2f}pt  "
        f"95%CI=[{boot_held_out['lo']:.2f}, {boot_held_out['hi']:.2f}]  "
        f"ブレークイーブン比={'超過' if r_held_out['model'] > breakeven else '未達'}")

    results_by_box[BOX_N] = {
        "model_file": WINNER_FILES[BOX_N],
        "fitted_on_dates": fitted_on["search_dates"] + fitted_on["holdout_dates"],
        "fitted_on_n_races": fitted_on["n_races"],
        "n_held_out_blocks": len(held_out_blocks),
        "n_held_out_races": int(len(held_out_idx)),
        "breakeven_pct": breakeven,
        "all_211_races": {"model": r_all["model"], "market": r_all["market"], "excess": r_all["excess"]},
        "held_out_only": {"model": r_held_out["model"], "market": r_held_out["market"],
                          "excess": r_held_out["excess"], "bootstrap_ci": boot_held_out,
                          "above_breakeven": bool(r_held_out["model"] > breakeven)},
    }

OUT_JSON.write_text(json.dumps({
    "n_races": len(races),
    "note": "Step0(2026-08-22): BOX側レポートのヘッドライン値(全211レース直接評価)は"
            "約半分がin-sample(現行重みのfit母集団と重複)。held_out_onlyが公正な比較対象。"
            "既存5本の探索レポートのREJECTED判定自体は覆らない(current modelとの"
            "block_bootstrap_diffに基づくため)が、今後はこちらを現状認識の基準とする。",
    "results_by_box": results_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {OUT_JSON}")
