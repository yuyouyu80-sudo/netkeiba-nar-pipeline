# -*- coding: utf-8 -*-
"""Phase 3(モデリング変更)の検出力(power)事前登録(2026-09-06、改善計画Phase 0-C)。

過去の重み再探索(2026-08-11, 08-12, 08-28)は、`jra_eval.selection_optimism()`が返す
`true_edge_pt / true_edge_sd >= 2.0`ゲートで3回ともREJECTEDになっている。この`true_edge_sd`は
n_rep回の半分割レプリケート間のばらつきで、ブロック数(開催日×競馬場の分割単位)にほぼ
反比例して縮む(sd_at_N ≈ sd_at_n0 * sqrt(n0/N))。

本スクリプトは、既存の探索結果JSON(`selection_optimism`を含む)から現在のブロック数n0での
sdを読み取り、(a) 現在のブロック数で検出可能な最小効果量(MDE、gate_ratio倍のsd)、
(b) 指定した目標効果量を検出するのに必要なブロック数、を計算して記録する。

**運用ルール**: Phase 3の各実験は、ここで計算したMDEを下回る改善しか実質的に見込めない
場合は実施しない(0-Aの過去データバックフィルでブロック数が増えるまで凍結する)。

出力: `data/jra_pipeline/power_precheck.json`
"""
from __future__ import annotations

import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RESULT_JSON = (
    PROJECT_ROOT / "data" / "jra_pipeline" / "jra_search500_2026_08_28_v4signals_result.json"
)
OUT_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "power_precheck.json"

GATE_RATIO = 2.0
TARGET_EFFECTS_PT = [2.0, 3.0, 5.0, 10.0]


def precheck(result_json: Path = DEFAULT_RESULT_JSON, gate_ratio: float = GATE_RATIO) -> dict:
    data = json.loads(result_json.read_text(encoding="utf-8"))
    n_blocks0 = data["n_blocks"]
    n_races0 = data.get("n_races")

    per_box = {}
    for box, r in data.get("results_by_box", {}).items():
        so = r.get("selection_optimism", {})
        sd0 = so.get("true_edge_sd")
        edge0 = so.get("true_edge_pt")
        if sd0 is None:
            continue
        current_mde_pt = gate_ratio * sd0
        required_n_for_target = {
            f"{eff}pt": math.ceil(n_blocks0 * ((gate_ratio * sd0 / eff) ** 2))
            for eff in TARGET_EFFECTS_PT
        }
        per_box[box] = {
            "observed_true_edge_pt": edge0,
            "observed_true_edge_sd": sd0,
            "current_n_blocks": n_blocks0,
            "current_mde_pt_at_gate_ratio_2": round(current_mde_pt, 2),
            "required_n_blocks_for_target_effect": required_n_for_target,
            "phase3_recommendation": (
                "凍結(標本不足)" if n_blocks0 < required_n_for_target.get("3.0pt", 0)
                else "着手可"
            ),
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "source": str(result_json.relative_to(PROJECT_ROOT)),
        "n_races0": n_races0,
        "n_blocks0": n_blocks0,
        "gate_ratio": gate_ratio,
        "per_box": per_box,
        "notes": (
            "sdはブロック数にほぼ反比例して縮む(sd_at_N ≈ sd_at_n0 * sqrt(n0/N))という近似に"
            "基づく概算。目標効果量3ptを『競馬の回収率改善として現実的な下限』の目安として"
            "phase3_recommendationの判定基準に使っている(恣意的な閾値であり、実際の意思決定は"
            "この数値だけでなくドメイン知識も併用すること)。"
        ),
    }
    OUT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-json", type=Path, default=DEFAULT_RESULT_JSON)
    parser.add_argument("--gate-ratio", type=float, default=GATE_RATIO)
    args = parser.parse_args()

    report = precheck(args.result_json, args.gate_ratio)
    print(f"n_races0={report['n_races0']}  n_blocks0={report['n_blocks0']}  gate_ratio={report['gate_ratio']}")
    for box, info in sorted(report["per_box"].items()):
        print(
            f"  box{box}: true_edge_pt={info['observed_true_edge_pt']:.2f}  "
            f"sd={info['observed_true_edge_sd']:.2f}  "
            f"current_MDE={info['current_mde_pt_at_gate_ratio_2']}pt  "
            f"required_n_for_3pt={info['required_n_blocks_for_target_effect']['3.0pt']}  "
            f"-> {info['phase3_recommendation']}"
        )
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
