# -*- coding: utf-8 -*-
"""NAR「段階的的中確率のはしご」(topk_ladder)較正スクリプト。

2026-08-12、JRA/NAR確信度統一の一環で全面書き換えた。旧来のbox_n(5/4/3)×hit_mode(place/profit)
較正テーブルは廃止し、JRAと同じ単勝ベースtopk_ladder(K=5..1)のみに統一した。較正ロジック本体は
`scripts/common/confidence_calibrate.py`(JRA/NAR共通)に委譲する。旧来の候補特徴量`gap_pct`・
`spread`も廃止し、`gap_boundary_k`(用途ごとに事前登録、選択バイアスを排除)のみを使う。
主手法はロジスティック回帰(Platt scaling)。旧来の4分位バケット法は診断比較用に残す。
`beats_trivial_baseline`はOOF Brierの点推定比較ではなく、ブロック単位ブートストラップ95%CIで
判定する(共通モジュール側の実装)。

出力: data/nar_pipeline/confidence_calibration_nar.json / confidence_calibration_report.txt
(スキーマはJRA側 data/jra_pipeline/confidence_calibration.json と同一にした)
"""
import json
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
import confidence_calibrate as CC  # noqa: E402
import nar_backtest as NB  # noqa: E402
import nar_dataset  # noqa: E402
import nar_eval as NE  # noqa: E402
import nar_signals as NS  # noqa: E402

LADDER_KS = [5, 4, 3, 2, 1]
LADDER_WEIGHTS_JSON = "winner_box5_nar.json"  # 予想5頭表示(box5)と同じ重みを使う

OUT_JSON = DATA_DIR / "confidence_calibration_nar.json"
OUT_TXT = DATA_DIR / "confidence_calibration_report.txt"
lines: list = []


def log(s: str = "") -> None:
    print(s)
    lines.append(str(s))


data = nar_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
log(f"レース数: {len(races)}  日付: {data['dates']}")

priors = NS.make_priors(races)
NAMES = NS.ALL_SIGNALS
mats = NS.signal_matrices(races, priors, NAMES)
blocks = NE.blocks_of(races)
log(f"ブロック数(開催日×競馬場): {len(set(blocks))}")


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


ladder_w = json.loads((DATA_DIR / LADDER_WEIGHTS_JSON).read_text(encoding="utf-8"))["weights"]
LADDER_W = wvec(ladder_w)

log("\n" + "#" * 72)
log("段階的的中確率のはしご(topk_ladder): 上位K頭picksに実際の1着馬が入っていたか")
log("単勝ベース・logistic較正が主手法。bucket法(旧手法)は参考診断として併記する。")
log("#" * 72)

ladder_results = {}
for K in LADDER_KS:
    log("\n" + "-" * 60)
    log(f"K={K}(上位{K}頭に実際の1着馬が入っている確率)")
    log("-" * 60)
    ev = NE.Evaluator(races, actual, box_n=K)
    picks = NE.score_picks(mats, LADDER_W, K)
    st, rt = ev.settler.returns_for(picks)
    win_col = NB.BET_TYPES.index("単勝")
    hit = (rt[:, win_col] > 0).astype(float)

    rows = []
    for i, m in enumerate(mats):
        num, den = m["S"] @ LADDER_W, m["A"] @ LADDER_W
        score = np.where(den > 0, num / den, -1e18)
        order = np.argsort(-score, kind="stable")
        s_sorted = score[order]
        gaps = CC.gap_features(s_sorted, [K])
        rows.append({"race_id": races[i]["race_id"], "field_size": len(s_sorted),
                    "gap_boundary_k": gaps[f"gap_boundary_{K}"], "hit": float(hit[i])})
    df = pd.DataFrame(rows)
    log(f"  対象レース数: {len(df)}  実測hit率: {df['hit'].mean() * 100:.1f}%")

    res_logistic = CC.lobo_bucket_calibrate(
        df, blocks, "hit", feature_candidates=("gap_boundary_k",), method="logistic",
        hit_definition_label="単勝: 1着馬が上位K頭picksに入っているか")
    res_bucket = CC.lobo_bucket_calibrate(
        df, blocks, "hit", feature_candidates=("gap_boundary_k",), method="bucket",
        hit_definition_label="単勝: 1着馬が上位K頭picksに入っているか")

    log(f"  [logistic(採用)] OOF Brier={res_logistic['chosen_oof_brier']:.4f}"
        f" vs 自明基準{res_logistic['trivial_baseline_oof_brier']:.4f}"
        f"  Brier差95%CI=[{res_logistic['brier_gain_ci95'][0]:+.4f}, {res_logistic['brier_gain_ci95'][1]:+.4f}]"
        f"  beats_trivial={res_logistic['beats_trivial_baseline']}"
        f"  min_blocks_ok={res_logistic['min_blocks_ok']}(n_blocks={res_logistic['n_blocks_total']})"
        f"  show_pct={res_logistic['show_pct']}")
    log(f"  [bucket(参考診断)] OOF Brier={res_bucket['chosen_oof_brier']:.4f}"
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
    "fitted_on": {"n_races": len(races), "dates": data["dates"], "n_blocks": len(set(blocks))},
    "method": "lobo_logistic_calibration_with_bootstrap_ci_gate",
    "topk_ladder": {
        "weights_source": LADDER_WEIGHTS_JSON,
        "hit_definition": "上位K頭picksの中に実際の1着馬が含まれているか(単勝リターン>0)",
        "ks": LADDER_KS,
        "results_by_k": {str(k): v for k, v in ladder_results.items()},
    },
    "code_sha256": {
        name: sha256((LIB_DIR / name).read_bytes()).hexdigest()[:16]
        for name in ["nar_signals.py", "nar_eval.py", "nar_backtest.py", "nar_dataset.py",
                    "nar_confidence_calibrate.py"]
    },
    "created_at": datetime.now().isoformat(timespec="seconds"),
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
