# -*- coding: utf-8 -*-
"""「予想力×高低差×ペース×斤量 馬連頭軸レース帳(実戦版)」Artifact
(https://claude.ai/code/artifact/7822331c-1ce4-4964-a60d-6624ee0aa6ca、2026-09-06パターン更新版)
の検証済みレース一覧(軸馬番+相手馬番)を、JRA日次予想レポート側で「対象レース・対象馬 その3」
として色分け表示するための恒久データに変換する。

このArtifact自体は別セッションで作成・検証されたもの(通常戦242レース中、直近2開催日
[20260829・20260830]をホールドアウト除外した母集団で、軸=本番予想スコア順位(BOX5モデル基準)
1位固定+相手2〜4頭を能力・調教評価順位×コース高低差×想定ペース×斤量で絞り込み、単勝1・単勝2
とは異なり馬連(軸+相手1頭ずつ)の的中パターンを検証したもの)。この検証済み(結果確定済み)側の
61レース・的中率39.3%等の統計値は、Artifact側で既に計算済みの結果をそのまま恒久化して使う
(再検証はしない、method-note等の数値の出典)。

2026-09-06追記(ユーザー依頼「単勝1・単勝2と同じく9/5・9/6分も計算してほしい」)で、
--pending-json オプションを追加。パターン1(box5_pedigree_jt)・パターン2(mark_age_odds)と
同様、結果未確定(pending)日についてはjra_pending_factor_scoring.py出力
(pred_rank・radar_rank_ability・course_elevation_band・race_pace_label・weight_carried_band、
いずれも発走前に確定済みでリークではない)から**同一の軸+相手フィルタ条件を自前で再計算**し、
検証済み61レースのマップへ動的にマージする(pending日分は当然、統計値[n_races/hit_rate_pct等]
には含めない。those remain the static verified-only numbers)。

出力: data/jra_pipeline/factor_highlight_map_3.json
  {
    "generated_at": "...",
    "source_artifact": "https://claude.ai/code/artifact/7822331c-1ce4-4964-a60d-6624ee0aa6ca",
    "bet_type": "馬連",
    "filter_spec": [[group_label, option_label], ...],
    "population_note": "...",
    "n_races": 61, "hit_races": 24, "hit_rate_pct": 39.3, "return_rate_pct": 274.1,
    "races": {race_id: {"axis_umaban": int, "partner_umabans": [int, ...]}}
  }

使い方: 検証済み側はArtifactの report-data JSON(<script id="report-data">)をUTF-8でダンプした
ファイルを --dump-json で渡す(Artifact本体はブラウザ実行前提でPythonから直接読めないため、
事前に `Artifact action:"read"` → 埋め込みJSONをUTF-8ファイルへ書き出す、という1回限りの
手動手順が必要)。pending側は --pending-json <pending_factor_{date}.json> を(複数可で)渡す。
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT = PROJECT_ROOT / "data" / "jra_pipeline" / "factor_highlight_map_3.json"
SOURCE_ARTIFACT_URL = "https://claude.ai/code/artifact/7822331c-1ce4-4964-a60d-6624ee0aa6ca"

# Artifact 7822331c の filter_spec(相手側の絞り込み条件)をpending日向けに再現するための
# しきい値。軸側(pred_rank==1)は別途ハードコード(このArtifact固有の「BOX5モデル1位固定」)。
PARTNER_ABILITY_RANK_MAX = 5
PARTNER_ELEVATION_BANDS = {"mid", "large"}
PARTNER_PACE_LABELS = {"H", "S"}
PARTNER_WEIGHT_BANDS = {"w55_56", "ge57"}
PARTNER_COUNT_RANGE = (2, 4)


def _match_pending_race(race: dict) -> dict | None:
    """1レース分(jra_pending_factor_scoring.py出力の1エントリ)を評価し、
    条件に一致すれば{"axis_umaban":..., "partner_umabans":[...]}を返す(不一致はNone)。"""
    if race.get("race_type") != "normal":
        return None
    horses = race.get("horses", [])
    axis = next((h for h in horses if h.get("pred_rank") == 1), None)
    if axis is None or axis.get("umaban") is None:
        return None
    partners = []
    for h in horses:
        if h is axis:
            continue
        if h.get("umaban") is None:
            continue
        ability = h.get("radar_rank_ability")
        if ability is None or ability > PARTNER_ABILITY_RANK_MAX:
            continue
        if h.get("course_elevation_band") not in PARTNER_ELEVATION_BANDS:
            continue
        if h.get("race_pace_label") not in PARTNER_PACE_LABELS:
            continue
        if h.get("weight_carried_band") not in PARTNER_WEIGHT_BANDS:
            continue
        partners.append(int(h["umaban"]))
    lo, hi = PARTNER_COUNT_RANGE
    if not (lo <= len(partners) <= hi):
        return None
    return {"axis_umaban": int(axis["umaban"]), "partner_umabans": partners}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dump-json", help="Artifactのreport-data JSONをダンプしたUTF-8ファイル(検証済み側)")
    ap.add_argument("--pending-json", action="append", default=[],
                     help="jra_pending_factor_scoring.py出力(複数可、pending側)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    if not args.dump_json and Path(args.out).exists():
        # --dump-jsonを省略した再実行(pending側だけ更新したい場合)は、既存出力から
        # 検証済み側のメタデータ・レース一覧を引き継ぐ。
        base = json.loads(Path(args.out).read_text(encoding="utf-8"))
    elif args.dump_json:
        dump = json.loads(Path(args.dump_json).read_text(encoding="utf-8"))
        races_out = {
            r["race_id"]: {
                "kaisai_date": r["kaisai_date"],
                "axis_umaban": r["axis_umaban"],
                "partner_umabans": r["partner_umabans"],
            }
            for r in dump["races"]
        }
        base = {
            "source_artifact": SOURCE_ARTIFACT_URL,
            "bet_type": dump["bet_type"],
            "filter_spec": dump["filter_spec"],
            "population_note": dump["population_note"],
            "n_races": dump["n_races"],
            "n_points": dump["n_points"],
            "hit_races": dump["hit_races"],
            "hit_rate_pct": dump["hit_rate_pct"],
            "return_rate_pct": dump["return_rate_pct"],
            "races": races_out,
        }
    else:
        raise SystemExit("--dump-json(初回)または既存の--outファイル(再実行)のいずれかが必要です")

    # pending側(結果未確定日)の動的再計算。検証済み側のrace_idと衝突した場合は
    # 検証済み側(backtest実績あり)を優先し上書きしない。
    pending_matched = 0
    pending_checked = 0
    for pj_path in args.pending_json:
        pending = json.loads(Path(pj_path).read_text(encoding="utf-8"))
        for race_id, race in pending.items():
            pending_checked += 1
            if race_id in base["races"]:
                continue
            m = _match_pending_race(race)
            if m is None:
                continue
            base["races"][race_id] = {
                "kaisai_date": race.get("kaisai_date"),
                "axis_umaban": m["axis_umaban"],
                "partner_umabans": m["partner_umabans"],
            }
            pending_matched += 1

    base["generated_at"] = datetime.now(timezone.utc).isoformat()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(base, ensure_ascii=False, indent=1), encoding="utf-8")
    print(
        f"wrote {out_path} ({len(base['races'])} races total; "
        f"pending checked={pending_checked} matched={pending_matched})"
    )


if __name__ == "__main__":
    main()
