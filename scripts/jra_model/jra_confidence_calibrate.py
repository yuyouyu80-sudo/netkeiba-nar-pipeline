# -*- coding: utf-8 -*-
"""JRA「段階的的中確率のはしご」(topk_ladder)較正スクリプト。

2026-08-12、JRA/NAR確信度統一の一環でscratchpadから昇格し、以下を変更した:
  * `BASE / "predict.py"`(scratchpadハードコードパス)への動的importをやめ、
    `jra_signals.py`(単一の真実の源)+`winner_v3.json`を直接使用する。
  * 較正ロジック本体は`scripts/common/confidence_calibrate.py`(JRA/NAR共通)に委譲する。
    主手法はロジスティック回帰(Platt scaling)。旧来の4分位バケット法は診断比較用に残す。
  * 出力先を`data/jra_pipeline/`(git管理下)に変更し、NAR側(`data/nar_pipeline/`)と対称にした。
  * `beats_trivial_baseline`は単純なOOF Brier点推定の比較ではなく、ブロック単位ブートストラップ
    95%CIで判定する(共通モジュール側の実装)。ブロック総数がMIN_BLOCKS_FOR_PCT未満の場合は
    `show_pct=False`となり、レポート側は較正済み%でなく3段階(高/中/低)表示にフォールバックする
    設計にした(現在のJRAは約30ブロックしかなく、この閾値を満たさない見込み。意図した挙動)。

的中定義: 上位K頭picksの中に実際の1着馬が含まれているか(単勝払戻、K=5,4,3,2,1)。NAR側とも
この単勝ベース定義で統一した(box_n×place/profit較正テーブルはNAR側で廃止)。

出力: data/jra_pipeline/confidence_calibration.json / confidence_calibration_report.txt
"""
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
import confidence_calibrate as CC  # noqa: E402
import jra_dataset as JD  # noqa: E402
import jra_signals as JS  # noqa: E402

LADDER_KS = [5, 4, 3, 2, 1]
OUT_JSON = DATA_DIR / "confidence_calibration.json"
OUT_TXT = DATA_DIR / "confidence_calibration_report.txt"
lines: list = []


def log(s: str = "") -> None:
    print(s)
    lines.append(str(s))


data = JD.load(rebuild=True)
races, actual = data["races"], data["actual"]
log(f"レース数: {len(races)}  日付: {data['dates']}")

winner = json.loads((DATA_DIR / "winner_v3.json").read_text(encoding="utf-8"))
WEIGHTS, PRIORS, CLASS_ORDINAL = winner["weights"], winner["priors"], winner["class_ordinal"]

rows = []
for r in races:
    win_payouts = actual.get(r["race_id"], {}).get("単勝", {})
    winners = set(win_payouts.keys())
    if not winners:
        continue
    current_class = JS._class_ordinal(r["race_name"], CLASS_ORDINAL)
    scored = JS.score_race(r["df"], current_class, WEIGHTS, PRIORS, CLASS_ORDINAL)
    score = scored["_score"].to_numpy(dtype=float)
    score = np.where(np.isnan(score), -1e18, score)
    umaban = scored["umaban"].astype(int).to_numpy()
    order = np.argsort(-score, kind="stable")
    sorted_scores = score[order]
    sorted_umaban = umaban[order]

    gaps = CC.gap_features(sorted_scores, LADDER_KS)
    rec = {"race_id": r["race_id"], "kaisai_date": r["kaisai_date"], "racecourse": r["racecourse"],
          "field_size": len(scored), **gaps}
    for k in LADDER_KS:
        rec[f"hit_{k}"] = 1.0 if (set(sorted_umaban[:k].tolist()) & winners) else 0.0
    rows.append(rec)

full_df = pd.DataFrame(rows)
full_df["_block"] = full_df["kaisai_date"] + "_" + full_df["racecourse"]
blocks_arr = full_df["_block"].to_numpy()
log(f"対象レース数: {len(full_df)}  ブロック数(開催日×競馬場): {len(set(blocks_arr))}")

log("\n" + "#" * 72)
log("段階的的中確率のはしご(topk_ladder): 上位K頭picksに実際の1着馬が入っていたか")
log("単勝ベース・logistic較正が主手法。bucket法(旧手法)は参考診断として併記する。")
log("#" * 72)

ladder_results = {}
for K in LADDER_KS:
    log("\n" + "-" * 60)
    log(f"K={K}(上位{K}頭に実際の1着馬が入っている確率)")
    log("-" * 60)
    sub = full_df[[f"gap_boundary_{K}", f"hit_{K}"]].rename(
        columns={f"gap_boundary_{K}": "gap_boundary_k", f"hit_{K}": "hit"})
    log(f"  実測hit率: {sub['hit'].mean() * 100:.1f}%")

    res_logistic = CC.lobo_bucket_calibrate(
        sub, blocks_arr, "hit", feature_candidates=("gap_boundary_k",), method="logistic",
        hit_definition_label="単勝: 1着馬が上位K頭picksに入っているか")
    res_bucket = CC.lobo_bucket_calibrate(
        sub, blocks_arr, "hit", feature_candidates=("gap_boundary_k",), method="bucket",
        hit_definition_label="単勝: 1着馬が上位K頭picksに入っているか")

    log(f"  [logistic(採用)] OOF Brier={res_logistic['chosen_oof_brier']:.4f}"
        f" vs 自明基準{res_logistic['trivial_baseline_oof_brier']:.4f}"
        f"  Brier差95%CI=[{res_logistic['brier_gain_ci95'][0]:+.4f}, {res_logistic['brier_gain_ci95'][1]:+.4f}]"
        f"  beats_trivial={res_logistic['beats_trivial_baseline']}"
        f"  min_blocks_ok={res_logistic['min_blocks_ok']}(n_blocks={res_logistic['n_blocks_total']})"
        f"  show_pct={res_logistic['show_pct']}")
    log(f"  [bucket(参考診断)] OOF Brier={res_bucket['chosen_oof_brier']:.4f}"
        f"  Brier差95%CI=[{res_bucket['brier_gain_ci95'][0]:+.4f}, {res_bucket['brier_gain_ci95'][1]:+.4f}]"
        f"  beats_trivial={res_bucket['beats_trivial_baseline']}"
        f"  shrink_k={res_bucket['fit_params']['shrink_k']}")

    ladder_results[K] = res_logistic
    ladder_results[K]["bucket_diagnostic"] = {
        "oof_brier": res_bucket["chosen_oof_brier"],
        "brier_gain_ci95": res_bucket["brier_gain_ci95"],
        "beats_trivial_baseline": res_bucket["beats_trivial_baseline"],
        "shrink_k": res_bucket["fit_params"]["shrink_k"],
    }

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "fitted_on": {"n_races": len(full_df), "dates": data["dates"],
                 "n_blocks": len(set(blocks_arr))},
    "method": "lobo_logistic_calibration_with_bootstrap_ci_gate",
    "topk_ladder": {
        "weights_source": "winner_v3.json",
        "hit_definition": "上位K頭picksの中に実際の1着馬が含まれているか(単勝払戻)",
        "ks": LADDER_KS,
        "results_by_k": {str(k): v for k, v in ladder_results.items()},
    },
    "created_at": datetime.now().isoformat(timespec="seconds"),
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
