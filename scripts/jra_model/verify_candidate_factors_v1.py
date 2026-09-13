# -*- coding: utf-8 -*-
"""candidate_factors_v1で追加した新規フィールドの簡易サニティチェック(一時スクリプト)。
分布が退化していないか(全部None/全部同じ値になっていないか)・値域が妥当かだけを見る。

2026-09-02: 当初「h.get(f) is None」でNone判定していたが、pandas Seriesの暗黙型強制で
一部のNoneがnp.nanへ化けてJSON上は非準拠のNaNトークンになっていたバグを発見(json標準の
NaN許容パースにより`math.isnan`でしか検出できなかった)。それ自体はbuild_factor_dataset.py
側で修正済みだが、このスクリプトも`is None`だけに頼らずmath.isnanも見るよう修正し、
かつ生JSONテキストにNaNトークンが残っていないかも直接チェックする(再発検知用)。"""
import json
import math
from collections import Counter
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline" / "factor_database.json"
_raw_text = DB_PATH.read_text(encoding="utf-8")
n_nan_tokens = _raw_text.count(":NaN,") + _raw_text.count(":NaN}")
print(f"raw JSON中の非準拠NaNトークン数: {n_nan_tokens} (0が正常、1件でもあればブラウザで"
      f"JSON.parseがクラッシュする)")

DB = json.loads(_raw_text)


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))

FIELDS_NUM = ["career_place_rate", "distance_band_place_rate", "turn_apt_place_rate",
              "course_size_apt_place_rate", "slope_apt_place_rate", "turf_type_apt_place_rate"]
FIELDS_CAT = ["grade_best", "first_time_flag", "jockey_switch", "trainer_change",
              "stable_multi_entry", "kaisai_late"]

horses = [h for r in DB["races"] for h in r["horses"]]
print(f"total horses: {len(horses)}  total races: {len(DB['races'])}")

for f in FIELDS_NUM:
    vals = [h[f] for h in horses if not _is_missing(h.get(f))]
    n_none = sum(1 for h in horses if _is_missing(h.get(f)))
    if vals:
        print(f"{f}: n={len(vals)} none={n_none} min={min(vals):.1f} max={max(vals):.1f} "
              f"mean={sum(vals)/len(vals):.1f}")
    else:
        print(f"{f}: ALL NONE (n={n_none}) <-- 要確認")

for f in FIELDS_CAT:
    c = Counter(h.get(f) for h in horses)
    print(f"{f}: {dict(c)}")

# 開催後半フラグはレース単位で1つのはず(全馬同じ値)。ばらつきがあればバグ。
bad = 0
for r in DB["races"]:
    vals = {h.get("kaisai_late") for h in r["horses"]}
    if len(vals) > 1:
        bad += 1
print(f"kaisai_late inconsistent within a race: {bad} races (0が正常)")

# grade_best=="no_history" と first_time_flag=="both_first" は共に「n_runs==0(初出走)」の
# ときだけ立つはずなので、両者の該当馬集合は完全一致するはず(不一致があればロジックの矛盾=バグ)。
no_hist = {(r["race_id"], h["umaban"]) for r in DB["races"] for h in r["horses"] if h.get("grade_best") == "no_history"}
both_first = {(r["race_id"], h["umaban"]) for r in DB["races"] for h in r["horses"] if h.get("first_time_flag") == "both_first"}
print(f"no_history count={len(no_hist)}  both_first count={len(both_first)}  "
      f"mismatch={len(no_hist ^ both_first)}(0が正常)")

# 新馬戦(race_type=shinba)は定義上、全馬が初出走のはず。grade_best=="no_history"の
# 割合が新馬戦でどれだけ高いかを確認する(以前は0%だったバグの再発チェック)。
shinba_horses = [h for r in DB["races"] for h in r["horses"] if h.get("race_type") == "shinba"]
shinba_no_hist = sum(1 for h in shinba_horses if h.get("grade_best") == "no_history")
print(f"shinba(新馬戦)horses={len(shinba_horses)}  no_history among them={shinba_no_hist} "
      f"({100*shinba_no_hist/len(shinba_horses):.1f}%、本来は100%に近いはず)")
