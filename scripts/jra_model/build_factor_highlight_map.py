# -*- coding: utf-8 -*-
"""日次予想レポート(build_artifact.py)向けに、特定のファクター条件に一致するレース・
対象馬を抽出し、`data/jra_pipeline/factor_highlight_map*.json`へ書き出す。

対象は data/jra_pipeline/factor_database.json(結果確定済み、build_factor_dataset.py出力)
の全レース + `--pending-json`で渡すpending日分(jra_pending_factor_scoring.py出力、
結果未確定のため着順情報は無いが対象馬判定には不要)。

`--pattern`で選択可能な条件(2026-09-05、2パターン目を追加):
  - box5_pedigree_jt(既定): 「BOX5×血統×騎手厩舎 適合レース帳」(005c6aea)と同一の7条件。
  - mark_age_odds: 「予想印×年齢×オッズ帯 単勝適合レース帳」(2af83b24)と同一の7条件。

出力スキーマ(パターン共通):
{
  "filter_spec": [[グループラベル, 選択ラベル], ...],  # 表示用、元Artifactと同じ形式
  "races": {race_id: {"target_umabans": [umaban, ...], "kaisai_date": "..."}}
}
"""
import argparse
import json
import sys
from pathlib import Path

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(LIB_DIR))
from jra_factor_registry import FACTOR_GROUPS, horse_qualifies  # noqa: E402

PATTERNS = {
    # 「BOX5×血統×騎手厩舎 適合レース帳」(005c6aea)と同一の選択(2026-09-05時点)
    "box5_pedigree_jt": {
        "selected": {
            "race_type": {"rt_normal"},
            "score_rank": {"score_top3"},
            "radar_pedigree": {"radar_pedigree_top1"},
            "radar_jt": {"radar_jt_top5"},
            "waku": {"waku_3_6", "waku_7_8"},
            "score_rank_box4": {"score_box4_top5"},
            "score_rank_box3": {"score_box3_top5"},
        },
        "default_out": "factor_highlight_map.json",
    },
    # 「予想印×年齢×オッズ帯 単勝適合レース帳」(2af83b24)と同一の選択(2026-09-05追加)
    "mark_age_odds": {
        "selected": {
            "radar_mark": {"radar_mark_top1"},
            "horse_age": {"age_2", "age_5"},
            "odds_band": {"odds_ultra_fav", "odds_fav"},
            "course_size_band": {"csize_small", "csize_large"},
            "weight_carried_band": {"kin_52_54", "kin_55_56"},
            "jockey_area": {"jarea_ritto", "jarea_miho"},
            "jockey_lead_rank": {"jlead_top60"},
        },
        "default_out": "factor_highlight_map_2.json",
    },
}


def filter_spec_labels(selected_by_group: dict) -> list:
    out = []
    for gid, opt_ids in selected_by_group.items():
        group = FACTOR_GROUPS[gid]
        labels = [o["label"] for o in group["options"] if o["id"] in opt_ids]
        out.append([group["label"], "/".join(labels)])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pending-json", action="append", default=[],
                     help="jra_pending_factor_scoring.py出力(複数可)")
    ap.add_argument("--pattern", choices=sorted(PATTERNS), default="box5_pedigree_jt")
    ap.add_argument("--out", default=None,
                     help="省略時はパターンごとの既定ファイル名(data/jra_pipeline/配下)")
    args = ap.parse_args()

    selected_by_group = PATTERNS[args.pattern]["selected"]
    out_path = Path(args.out) if args.out else DATA_DIR / PATTERNS[args.pattern]["default_out"]

    fdb = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))

    races_out = {}
    n_checked = n_matched_races = 0
    for race in fdb["races"]:
        n_checked += 1
        targets = [h["umaban"] for h in race["horses"] if horse_qualifies(h, selected_by_group)]
        if targets:
            n_matched_races += 1
            races_out[race["race_id"]] = {"target_umabans": targets, "kaisai_date": race["kaisai_date"]}

    for pj_path in args.pending_json:
        pj = json.loads(Path(pj_path).read_text(encoding="utf-8"))
        for race_id, race in pj.items():
            n_checked += 1
            horses = [dict(h, race_type=race["race_type"]) for h in race["horses"]]
            targets = [h["umaban"] for h in horses if horse_qualifies(h, selected_by_group)]
            if targets:
                n_matched_races += 1
                races_out[race_id] = {"target_umabans": targets, "kaisai_date": race["kaisai_date"]}

    out = {"filter_spec": filter_spec_labels(selected_by_group), "races": races_out}
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{args.pattern}] checked {n_checked} races, matched {n_matched_races} races -> {out_path}")


if __name__ == "__main__":
    main()
