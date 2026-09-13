# -*- coding: utf-8 -*-
"""JRAコースの内回り/外回り区分の静的マッピング表を生成する(2026-09-06、改善計画1-D)。

新規スクレイピングは不要(公開情報の静的な組み合わせ)。既存の`data/jra_course_master.csv`と
その併設README(`data/jra_course_master.README.md`)に、新潟・中山・京都・阪神の4場は
芝コースが内回り/外回り複数構成を持つことと、その一部の距離別内訳が既に人手確認済みの
`notes`列・README本文として記載されている。本スクリプトはその**既に確認済みの情報のみ**を
プログラム的に整理してCSV化する(新しい競馬事実の確認・推測は一切行わない)。

各行の`source_basis`列で根拠を明示する:
- "single_config": その場・サーフェスは単一構成(内外の区別が無い、turf_available_distances_m
  にinline markerが無い6場+全10場のダート)
- "inline_marker": `jra_course_master.csv`の`turf_available_distances_m`列に距離ごと
  `(内)`/`(外)`/`(内・外)`のように明記されている場合(中山・京都)
- "notes_text": `notes`列の自由記述に距離が明記されている場合(新潟・阪神・京都の一部)
- "unknown": 上記のいずれでも解決できない距離(**推測せず`unknown`のまま残す**。UI上は
  `jra_factor_registry.py`のoptionsに登録しないため常に非該当=絞り込み対象外として扱われ、
  誤った事実を混入させるより安全)

turn_typeの値: "単一"(内外の区別なし) / "内回り" / "外回り" / "内外" (同じ距離が内外
両方の設定で施行されうる) / "unknown"

出力: `data/jra_course_turn_type.csv`(git管理下、静的参照表)
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
COURSE_MASTER_PATH = PROJECT_ROOT / "data" / "jra_course_master.csv"
OUT_PATH = PROJECT_ROOT / "data" / "jra_course_turn_type.csv"

# venue_code -> {distance_m: (turn_type, source_basis)}
# 新潟(04): turf_available_distances_m は全て(外)注記付き(1400/1600/1800/2000/3000/3200)。
#   notesの「内回り発走距離は1200;1400;2000;2200;2400」より、1400と2000は内外両方で
#   施行されうる(内外)、1200/2200/2400は内回り限定、1600/1800/3000/3200は外回り限定。
#   1000は「直線1000m専用コース」(notes)であり回り自体が無いため対象外(turn_type無し)。
NIIGATA_TURF = {
    1200: ("内回り", "notes_text"),
    1400: ("内外", "notes_text"),
    1600: ("外回り", "notes_text"),
    1800: ("外回り", "notes_text"),
    2000: ("内外", "notes_text"),
    2200: ("内回り", "notes_text"),
    2400: ("内回り", "notes_text"),
    3000: ("外回り", "notes_text"),
    3200: ("外回り", "notes_text"),
}

# 中山(06): turf_available_distances_m にdistance毎の(外)/(内)/(外・内)inline markerあり。
NAKAYAMA_TURF = {
    1200: ("外回り", "inline_marker"),
    1600: ("外回り", "inline_marker"),
    1800: ("内回り", "inline_marker"),
    2000: ("内回り", "inline_marker"),
    2200: ("外回り", "inline_marker"),
    2500: ("内回り", "inline_marker"),
    2600: ("外回り", "inline_marker"),
    3200: ("内外", "inline_marker"),
    3600: ("内回り", "inline_marker"),
    4000: ("外回り", "inline_marker"),
}

# 京都(08): turf_available_distances_m にdistance毎の(内・外)/(外)inline markerあり。
#   notesの「内回り限定発走距離は1100;1200」は、turf_available_distances_m列に無い
#   距離(掲載頻度が低い距離と推測)についての追加情報として扱う。
KYOTO_TURF = {
    1100: ("内回り", "notes_text"),
    1200: ("内回り", "notes_text"),
    1400: ("内外", "inline_marker"),
    1600: ("内外", "inline_marker"),
    1800: ("外回り", "inline_marker"),
    2000: ("内外", "inline_marker"),
    2200: ("外回り", "inline_marker"),
    2400: ("外回り", "inline_marker"),
    3000: ("外回り", "inline_marker"),
    3200: ("外回り", "inline_marker"),
}

# 阪神(09): turf_available_distances_m にinline markerが無い(1400;1600;1800;2400;2600)。
#   notesは「内回り限定発走距離は1200」とのみ記載(1200はturf_available_distances_mに無い
#   距離)。turf_available_distances_mに載っている5距離は根拠不足のためunknownのまま残す
#   (誤った内外判定を混入させないことを優先)。
HANSHIN_TURF = {
    1200: ("内回り", "notes_text"),
    1400: (None, "unknown"),
    1600: (None, "unknown"),
    1800: (None, "unknown"),
    2400: (None, "unknown"),
    2600: (None, "unknown"),
}

MULTI_CONFIG_TURF = {"04": NIIGATA_TURF, "06": NAKAYAMA_TURF, "08": KYOTO_TURF, "09": HANSHIN_TURF}


def _parse_distances(cell: str) -> list[int]:
    if not isinstance(cell, str) or not cell.strip():
        return []
    out = []
    for token in cell.split(";"):
        digits = "".join(ch for ch in token if ch.isdigit())
        if digits:
            out.append(int(digits))
    return out


def build() -> pd.DataFrame:
    master = pd.read_csv(COURSE_MASTER_PATH, dtype={"venue_code": str})
    rows = []
    for _, row in master.iterrows():
        venue_code = str(row["venue_code"]).zfill(2)
        venue = row["venue"]

        # --- ダート: 全10場とも単一構成(README記載通り、内外区別の言及なし) ---
        for d in _parse_distances(row.get("dirt_available_distances_m", "")):
            rows.append({
                "venue_code": venue_code, "venue": venue, "surface": "ダート",
                "distance_m": d, "turn_type": "単一", "source_basis": "single_config",
            })

        # --- 芝 ---
        if venue_code in MULTI_CONFIG_TURF:
            table = MULTI_CONFIG_TURF[venue_code]
            for d, (turn_type, basis) in table.items():
                rows.append({
                    "venue_code": venue_code, "venue": venue, "surface": "芝",
                    "distance_m": d, "turn_type": turn_type or "unknown", "source_basis": basis,
                })
            # 新潟の直線1000m専用コースのように、multi-config表に無い距離が
            # turf_available_distances_mに残っていればunknownとして拾っておく。
            covered = set(table.keys())
            for d in _parse_distances(row.get("turf_available_distances_m", "")):
                if d not in covered:
                    rows.append({
                        "venue_code": venue_code, "venue": venue, "surface": "芝",
                        "distance_m": d, "turn_type": "unknown", "source_basis": "unknown",
                    })
        else:
            for d in _parse_distances(row.get("turf_available_distances_m", "")):
                rows.append({
                    "venue_code": venue_code, "venue": venue, "surface": "芝",
                    "distance_m": d, "turn_type": "単一", "source_basis": "single_config",
                })

    df = pd.DataFrame(rows).drop_duplicates(subset=["venue_code", "surface", "distance_m"])
    df = df.sort_values(["venue_code", "surface", "distance_m"]).reset_index(drop=True)
    return df


def main() -> None:
    df = build()
    df.to_csv(OUT_PATH, index=False, encoding="utf-8")
    print(f"wrote {OUT_PATH.relative_to(PROJECT_ROOT)} ({len(df)} rows)")
    print(df["turn_type"].value_counts())
    unknown = df[df["turn_type"] == "unknown"]
    if len(unknown):
        print(f"\nunknown ({len(unknown)}行、意図的に推測せず残しています):")
        print(unknown[["venue", "surface", "distance_m"]].to_string(index=False))


if __name__ == "__main__":
    main()
