# -*- coding: utf-8 -*-
"""JRA10場のコース仕様テーブル(位置取り・展開予想機能=pace_predict.py用)。

値は一般的な競馬知識に基づく暫定値であり、JRA公式のコース図と1件ずつ突き合わせて
確認できたものから下記チェックリストの verified を True に更新すること。
未検証(verified=False)のエントリが実際に使われた場合、get_course_spec() は
warnings.warn() で警告する(黙って不正確な値を使わないため)。

検証チェックリスト(JRA公式コースガイドと突き合わせ、確認できたら [ ] -> [x]):
- [ ] 札幌 芝
- [ ] 札幌 ダート
- [ ] 函館 芝
- [ ] 函館 ダート
- [ ] 福島 芝
- [ ] 福島 ダート
- [ ] 新潟 芝 直線(1000m)
- [ ] 新潟 芝 内回り
- [ ] 新潟 芝 外回り
- [ ] 新潟 ダート
- [ ] 東京 芝
- [ ] 東京 ダート
- [ ] 中山 芝 内回り
- [ ] 中山 芝 外回り
- [ ] 中山 ダート
- [ ] 中京 芝
- [ ] 中京 ダート
- [ ] 京都 芝 内回り
- [ ] 京都 芝 外回り
- [ ] 京都 ダート
- [ ] 阪神 芝 内回り
- [ ] 阪神 芝 外回り
- [ ] 阪神 ダート
- [ ] 小倉 芝
- [ ] 小倉 ダート

実装メモ: 当初案では`run_to_first_turn`(第1コーナーまでの距離)も競馬場×距離ごとに
個別の値を持たせる予定だったが、venue×distanceの全組み合わせを正確に暗記/検証するのは
非現実的(実装者の記憶に頼った30通り以上の値を検証不能なまま埋め込むことになる)ため、
より単純で説明可能な代理指標に変更した: **距離そのもの**から
`run_to_first_turn_bucket(distance_m)`で短/中/長を導出する(短距離ほど発走から
最初のコーナーまでが短いという、JRAコース設計上よく知られる一般則を使う)。
`turn_tightness`(コーナーの急さ)は内回り/外回りの別やコース自体の特性として
CourseSpecに静的に持たせる(これは競馬場×コースという少数の組み合わせなので
チェックリストでの検証に無理がない)。
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass


@dataclass(frozen=True)
class CourseSpec:
    racecourse: str
    surface: str
    loop: str  # "single" | "inner" | "outer" | "straight"
    circumference_m: float
    home_stretch_m: float
    turn_direction: str  # "left" | "right" | "none"(直線コース)
    home_stretch_slope: str  # "uphill_steep" | "uphill_gentle" | "flat"
    turn_tightness: str  # "tight" | "gradual" | "none"(直線コース)
    verified: bool = False


TURN_TIGHTNESS_FACTOR = {"tight": 1.0, "gradual": 0.5, "none": 0.0}
RUN_TO_FIRST_TURN_FACTOR = {"short": 1.0, "medium": 0.6, "long": 0.3}

# 内外の別が無い6場+全場ダート
_BASE_SPECS: dict[tuple[str, str], CourseSpec] = {
    ("札幌", "芝"): CourseSpec("札幌", "芝", "single", 1640, 266, "left", "flat", "gradual"),
    ("札幌", "ダート"): CourseSpec("札幌", "ダート", "single", 1487, 264, "left", "flat", "gradual"),
    ("函館", "芝"): CourseSpec("函館", "芝", "single", 1627, 262, "right", "flat", "tight"),
    ("函館", "ダート"): CourseSpec("函館", "ダート", "single", 1475.8, 260, "right", "flat", "tight"),
    ("福島", "芝"): CourseSpec("福島", "芝", "single", 1600, 296, "right", "uphill_gentle", "tight"),
    ("福島", "ダート"): CourseSpec("福島", "ダート", "single", 1444, 292, "right", "flat", "tight"),
    ("新潟", "ダート"): CourseSpec("新潟", "ダート", "single", 1472, 353.9, "left", "flat", "gradual"),
    ("東京", "芝"): CourseSpec("東京", "芝", "single", 2083, 525.9, "left", "uphill_gentle", "gradual"),
    ("東京", "ダート"): CourseSpec("東京", "ダート", "single", 1899, 501.6, "left", "uphill_gentle", "gradual"),
    ("中山", "ダート"): CourseSpec("中山", "ダート", "single", 1493, 308, "right", "uphill_steep", "tight"),
    ("中京", "芝"): CourseSpec("中京", "芝", "single", 1705, 412.5, "left", "uphill_gentle", "gradual"),
    ("中京", "ダート"): CourseSpec("中京", "ダート", "single", 1530, 410.7, "left", "uphill_gentle", "gradual"),
    ("京都", "ダート"): CourseSpec("京都", "ダート", "single", 1607, 329.1, "right", "flat", "gradual"),
    ("阪神", "ダート"): CourseSpec("阪神", "ダート", "single", 1517.6, 357.7, "right", "uphill_gentle", "tight"),
    ("小倉", "芝"): CourseSpec("小倉", "芝", "single", 1615, 293, "right", "flat", "tight"),
    ("小倉", "ダート"): CourseSpec("小倉", "ダート", "single", 1445, 291, "right", "flat", "tight"),
}

# 内回り/外回りの別がある4場の芝
_DUAL_LOOP_SPECS: dict[tuple[str, str], CourseSpec] = {
    ("新潟", "straight"): CourseSpec("新潟", "芝", "straight", 1000, 1000, "none", "flat", "none"),
    ("新潟", "inner"): CourseSpec("新潟", "芝", "inner", 1998, 358.7, "left", "flat", "tight"),
    ("新潟", "outer"): CourseSpec("新潟", "芝", "outer", 2223, 358.7, "left", "flat", "gradual"),
    ("中山", "inner"): CourseSpec("中山", "芝", "inner", 1667, 308, "right", "uphill_steep", "tight"),
    ("中山", "outer"): CourseSpec("中山", "芝", "outer", 1840, 310, "right", "uphill_steep", "gradual"),
    ("京都", "inner"): CourseSpec("京都", "芝", "inner", 1782, 328.4, "right", "flat", "tight"),
    ("京都", "outer"): CourseSpec("京都", "芝", "outer", 1894, 404.6, "right", "flat", "gradual"),
    ("阪神", "inner"): CourseSpec("阪神", "芝", "inner", 1913, 356.5, "right", "uphill_gentle", "tight"),
    ("阪神", "outer"): CourseSpec("阪神", "芝", "outer", 2089, 473.6, "right", "uphill_gentle", "gradual"),
}

# 競馬場×距離(m) -> loop。暫定値、要検証(チェックリスト参照)。
_LOOP_RULES: dict[str, list[tuple[int, str]]] = {
    "新潟": [(1000, "straight"), (1400, "inner"), (2000, "inner"), (1600, "outer"), (1800, "outer"), (2200, "outer"), (2400, "outer")],
    "中山": [(1600, "inner"), (1800, "inner"), (1200, "outer"), (2000, "outer"), (2200, "outer"), (2500, "outer"), (2600, "outer")],
    "京都": [(1200, "inner"), (1800, "inner"), (2200, "inner"), (1400, "outer"), (1600, "outer"), (2000, "outer"), (2400, "outer"), (3000, "outer"), (3200, "outer")],
    "阪神": [(1200, "inner"), (1800, "inner"), (2000, "inner"), (2600, "inner"), (1400, "outer"), (1600, "outer"), (2200, "outer"), (2400, "outer")],
}

_DEFAULT_SPEC = CourseSpec("不明", "芝", "single", 1600, 300, "left", "flat", "gradual", verified=False)


def run_to_first_turn_bucket(distance_m: float) -> str:
    """距離から第1コーナーまでの長さの目安(短/中/長)を導出する(競馬場非依存の簡易ルール)。"""
    if distance_m <= 1200:
        return "short"
    if distance_m <= 1800:
        return "medium"
    return "long"


def _resolve_loop(racecourse: str, distance_m: float) -> str | None:
    rules = _LOOP_RULES.get(racecourse)
    if not rules:
        return None
    for d, loop in rules:
        if int(distance_m) == d:
            return loop
    # 完全一致が無い場合は最も近い距離のルールを流用する
    nearest = min(rules, key=lambda dl: abs(dl[0] - distance_m))
    return nearest[1]


def get_course_spec(racecourse: str, surface: str, distance_m: float) -> CourseSpec:
    """(競馬場, 芝/ダート, 距離)からCourseSpecを引く。未検証エントリ使用時は警告する。"""
    spec: CourseSpec | None = None
    if surface == "芝" and racecourse in _LOOP_RULES:
        loop = _resolve_loop(racecourse, distance_m)
        if loop is not None:
            spec = _DUAL_LOOP_SPECS.get((racecourse, loop))
    if spec is None:
        spec = _BASE_SPECS.get((racecourse, surface))
    if spec is None:
        warnings.warn(
            f"course_specs: {racecourse}/{surface}/{distance_m}m の仕様が見つからないため既定値を使用します"
        )
        return _DEFAULT_SPEC
    if not spec.verified:
        warnings.warn(
            f"course_specs: 未検証のコース仕様を使用しています "
            f"({racecourse}/{surface}/{spec.loop}/{distance_m}m) — "
            "JRA公式コース図で要確認(course_specs.py先頭のチェックリスト参照)"
        )
    return spec
