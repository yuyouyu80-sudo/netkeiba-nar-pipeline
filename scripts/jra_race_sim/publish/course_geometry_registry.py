# -*- coding: utf-8 -*-
"""競馬場・馬場ごとの実測コース諸元(周長m, 直線m)。旧`build_all_races_multi.py`/
`step4_regenerate_display.py`で重複定義されていたものを共通化。新潟/中京/札幌は
build_all_races_20260802.py(初出典)と同一値。函館/小倉/福島はJRA公式コース紹介ページ
(https://www.jra.go.jp/facilities/race/{hakodate,fukushima,kokura}/course/)で2026-08-08検証済み。

新潟芝の内回り/外回り: 1200m/1400mは内回り(1623.0m/359.0m、Wikipedia「新潟競馬場」記事)。
1600/1800/2000mは内回り・外回りどちらもあり得る(クラスに依存し、レース単位の確証が取れない)
ため外回り(2223.0m/658.7m)を暫定適用した上でgeometry_uncertainフラグを立てる。
"""
GEOMETRY = {
    ("新潟", "ダート"): (1472.0, 353.9),
    ("新潟", "芝", "outer"): (2223.0, 658.7),
    ("新潟", "芝", "inner"): (1623.0, 359.0),
    ("中京", "芝"): (1705.9, 412.5),
    ("中京", "ダート"): (1530.0, 410.7),
    ("札幌", "芝"): (1640.9, 266.1),
    ("札幌", "ダート"): (1487.0, 264.3),
    ("函館", "芝"): (1626.6, 262.1),
    ("函館", "ダート"): (1475.8, 260.3),
    ("福島", "芝"): (1600.0, 292.0),
    ("福島", "ダート"): (1444.6, 295.7),
    ("小倉", "芝"): (1615.1, 293.0),
    ("小倉", "ダート"): (1445.4, 291.3),
}

SURFACE_MAP = {"芝": "芝", "ダ": "ダート"}


def is_straight_course(row):
    return row["racecourse"] == "新潟" and row["surface"] == "芝" and float(row["distance_m"]) == 1000.0


def niigata_loop(distance_m):
    """新潟芝の内回り/外回り判定(距離ベースの暫定ルール)。戻り値: (loop, uncertain)。"""
    d = float(distance_m)
    if d in (1200.0, 1400.0):
        return "inner", False
    if d in (1600.0, 1800.0, 2000.0):
        return "outer", True  # クラス依存で確証なし
    return "outer", False  # 2200m以上・その他


def resolve_geometry(racecourse, surface, distance_m):
    """(racecourse, surface, distance_m) -> (circumference_m, home_stretch_m, geometry_note)"""
    if racecourse == "新潟" and surface == "芝":
        loop, uncertain = niigata_loop(distance_m)
        circ, home = GEOMETRY[("新潟", "芝", loop)]
        note = "niigata_%s%s" % (loop, "_uncertain" if uncertain else "")
        return circ, home, note
    key = (racecourse, surface)
    if key not in GEOMETRY:
        return None, None, None
    circ, home = GEOMETRY[key]
    return circ, home, "fixed"
