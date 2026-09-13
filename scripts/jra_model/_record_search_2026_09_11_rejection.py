# -*- coding: utf-8 -*-
"""2026-09-11のbox5/4/3重み再探索(45信号)がプロ競馬予想家役・プロシステムエンジニア役の
2体のOpus5サブエージェント独立レビューを受けて不採用(現行モデル維持)となった結論を、
winner_v3.json/winner_box4.json/winner_box3.jsonへ記録する一回限りのスクリプト。"""
import json
import pathlib

DATA_DIR = pathlib.Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"

entries = {
    "winner_v3.json": {"box_n": 5, "best_pattern": 1695, "in_sample_excess": 13.99,
                       "nested_oof_excess": 4.48, "true_edge": 1.46, "true_edge_sd": 5.48},
    "winner_box4.json": {"box_n": 4, "best_pattern": 258, "in_sample_excess": 16.41,
                         "nested_oof_excess": 4.88, "true_edge": 3.05, "true_edge_sd": 6.78},
    "winner_box3.json": {"box_n": 3, "best_pattern": 1559, "in_sample_excess": 11.07,
                         "nested_oof_excess": -5.97, "true_edge": 2.81, "true_edge_sd": 5.55},
}

for fname, e in entries.items():
    p = DATA_DIR / fname
    d = json.loads(p.read_text(encoding="utf-8"))
    d["search_2026_09_11"] = {
        "trigger": "ユーザー依頼「BOX3の予想モデルを変更したい、現時点で最良の予想モデルを探して」を受け、"
                  "box_n=5/4/3全てについてjra_dataset.load(rebuild=True)の現時点全319レース(18開催日)・"
                  "45信号(LEGACY10+CANDIDATE v1〜v5、ALL_SIGNALS_V5+CANDIDATE_SIGNALS_V4)で2000パターン"
                  "Dirichlet探索+Nested LOBO OOF+選択バイアス診断を実施。ユーザー了承の上で採否ゲート未満でも"
                  "「参考値」として最良候補を検討したが、続けてプロ競馬予想家役・プロシステムエンジニア役の"
                  "2体のOpus5サブエージェントに独立レビューさせたところ、両者とも独立検証の末「現行モデル維持」"
                  "で一致し、それを採用した。",
        "n_signals": 45,
        "best_pattern_index": e["best_pattern"],
        "in_sample_excess_pt": e["in_sample_excess"],
        "nested_lobo_oof_excess_pt": e["nested_oof_excess"],
        "true_edge_pt": e["true_edge"], "true_edge_sd": e["true_edge_sd"],
        "true_edge_over_sd_ratio": round(e["true_edge"] / e["true_edge_sd"], 3),
        "decision_gate": "true_edge_pt / true_edge_sd >= 2.0",
        "decision": "REJECTED - 不採用、現行モデルを維持",
        "decision_reason": (
            "true_edge/sd比が採否ゲート2.0を大きく下回る(0.27〜0.51)。さらに2体のサブエージェント"
            "独立レビューで追加の欠陥が判明: (1)V5血統信号キャッシュが陳腐化しており本番投入時は"
            "in-sample市場差が大きく低下する、(2)held-out比較が候補側に有利な非対称設計だった"
            "(対称化するとbox5/4/3とも市場に負ける)、(3)候補の優位は少数の高配当ブロック"
            "(特に20260802新潟、race_id drift未解決フラグ有り)に強く依存、(4)LEGACY10のみの"
            "対照群で測り直すと現行モデルの「極端な高パーセンタイル」も緩和される(45信号プールが"
            "現行モデルの重みパターンを内包していないための人工物)。"
        ),
        "additional_finding_current_model_also_unproven": (
            "両レビューが独立に「現行モデル自体もfit母集団(105R)を除くと市場超過の証拠が無い」ことを"
            "発見(box5/box4/box3の市場差を期間分解: fit母集団では高いが、中間・最新レースでは"
            "box5/box3がマイナスに転じる)。今回は「新候補を採用しない」根拠にはなるが、"
            "「現行モデルが優れている」ことの証明にはならない点に注意。"
        ),
        "full_reports": {
            "search": "jra_search_box{}_2026_09_11_report.txt".format(e["box_n"]),
            "predictor_review": "review_2026_09_11_predictor_persona.md",
            "engineer_review": "review_2026_09_11_engineer_persona.md",
        },
    }
    p.write_text(json.dumps(d, ensure_ascii=False, indent=2), encoding="utf-8")
    print("updated", fname)
