# -*- coding: utf-8 -*-
"""2026-09-13の同日内トラックバイアス候補シグナル(track_waku_bias、JRAデータ資産棚卸し
レビューPhase3)がbox5/4/3全てで採否ゲート未満(REJECT、現行モデル維持)となった結論を、
winner_v3.json/winner_box4.json/winner_box3.jsonへ記録する一回限りのスクリプト。
_record_search_2026_09_11_rejection.pyと同型。"""
import json
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"

entries = {
    "winner_v3.json": {"box_n": 5, "best_pattern": 657, "in_sample_excess": 17.20,
                       "nested_oof_excess": -5.55, "true_edge": 1.76, "true_edge_sd": 5.84},
    "winner_box4.json": {"box_n": 4, "best_pattern": 48, "in_sample_excess": 21.24,
                         "nested_oof_excess": -6.75, "true_edge": 1.46, "true_edge_sd": 6.25},
    "winner_box3.json": {"box_n": 3, "best_pattern": 656, "in_sample_excess": 23.68,
                         "nested_oof_excess": 9.02, "true_edge": 2.67, "true_edge_sd": 8.24},
}

for fname, e in entries.items():
    p = DATA_DIR / fname
    d = json.loads(p.read_text(encoding="utf-8"))
    d["search_2026_09_13"] = {
        "trigger": "「JRAデータ資産棚卸し」監査(Opus5サブエージェント、9/13)がdata/race_resultsの"
                  "実測コーナー通過順位列(corner1〜4_rank等、469,779行・参照スクリプト0)を"
                  "未活用データとして指摘。対象レース自身のコーナー位置は発走前に知りえないため、"
                  "枠番(発走前既知)を軸に、同日の先行レースの実測複勝率から「開催日×競馬場×"
                  "サーフェス×枠グループ(内/中/外)」単位の時点参照トラックバイアスを新設"
                  "(track_waku_bias、race_numberより厳密に前のレースのみ使用、リーク防止済み・"
                  "fill率が1R=0%→6R以降ほぼ100%と設計通り増加することを確認済み)。"
                  "jra_search_pedigree_2026_08_30.pyと同型のNested LOBO OOF+選択バイアス診断で"
                  "box5/4/3全てを検証。",
        "n_signals": 35,
        "best_pattern_index": e["best_pattern"],
        "in_sample_excess_pt": e["in_sample_excess"],
        "nested_lobo_oof_excess_pt": e["nested_oof_excess"],
        "true_edge_pt": e["true_edge"], "true_edge_sd": e["true_edge_sd"],
        "true_edge_over_sd_ratio": round(e["true_edge"] / e["true_edge_sd"], 3),
        "decision_gate": "true_edge_pt / true_edge_sd >= 2.0",
        "decision": "REJECTED - 不採用、現行モデルを維持",
        "decision_reason": (
            "true_edge/sd比が採否ゲート2.0を大きく下回る(0.23〜0.32)。box5/4はNested LOBO OOF"
            "市場差自体がマイナス(-5.55pt/-6.75pt)で現行モデルにも劣る。box3のみNested LOBO OOF"
            "市場差がプラスかつ現行モデルを上回った(+9.02pt vs +7.89pt)が、選択バイアス診断の"
            "楽観バイアスが+31.9ptと大きく、true_edge/sd比では他box同様FAILのため採用しない"
            "(box5/4/3間で一貫してFAILという結果自体は過去の候補群と同じパターン)。"
            "track_waku_bias自体の最良パターンにおける重み配分は1.6%(box5)/15.4%(box4)/"
            "7.3%(box3)で、box4では相応の重みが付いたが、ゲート自体を通過していない以上、"
            "本番反映は見送る。"
        ),
        "full_reports": {
            "search": "jra_search_track_bias_2026_09_13_report.txt",
            "feature_build": "jra_track_bias_features_2026_09_13.py(標準出力のみ、"
                            "レポートファイル無し。fill率の検証ログは実行時ログ参照)",
        },
    }
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated", fname)
