# -*- coding: utf-8 -*-
"""candidate_factors_v2・BOX4/BOX3 pred_rankで追加した新規フィールドの簡易サニティチェック。
分布が退化していないか(全部None/全部同じ値になっていないか)・値域が妥当かだけを見る。
verify_candidate_factors_v1.pyと同じ方針(_is_missing・生JSONのNaNトークン直接チェック)。"""
import json
import math
from collections import Counter
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline" / "factor_database.json"
_raw_text = DB_PATH.read_text(encoding="utf-8")
n_nan_tokens = _raw_text.count(":NaN,") + _raw_text.count(":NaN}")
print(f"raw JSON中の非準拠NaNトークン数: {n_nan_tokens} (0が正常)")

DB = json.loads(_raw_text)


def _is_missing(v):
    return v is None or (isinstance(v, float) and math.isnan(v))


FIELDS_NUM = ["pred_rank_box4", "pred_rank_box3", "class_form_place_rate", "kaisai_day_num"]
FIELDS_CAT = ["class_challenge_flag", "layoff_flag", "second_after_layoff_flag",
              "main_jockey_return_flag", "bad_run_popularity_drop_flag"]

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

# pred_rank_box4/box3は通常戦(normal)にのみ存在するはず(新馬・未勝利はNone)。
for f in ("pred_rank_box4", "pred_rank_box3"):
    by_type = Counter()
    for h in horses:
        by_type[(h.get("race_type"), _is_missing(h.get(f)))] += 1
    print(f"{f} missing-by-race_type: {dict(by_type)} (normalはFalseのみ、shinba/mishoubiはTrueのみが正常)")

# 通常戦の各レースで、pred_rank/pred_rank_box4/pred_rank_box3が1..出走頭数の順列になっているか
bad_perm = 0
n_normal_races = 0
for r in DB["races"]:
    if r["race_type"] != "normal":
        continue
    n_normal_races += 1
    n = len(r["horses"])
    for f in ("pred_rank", "pred_rank_box4", "pred_rank_box3"):
        ranks = sorted(h[f] for h in r["horses"])
        if ranks != list(range(1, n + 1)):
            bad_perm += 1
            print(f"  NG: race_id={r['race_id']} field={f} ranks={ranks}")
print(f"normal races checked: {n_normal_races}  permutation violations: {bad_perm} (0が正常)")

# pred_rank(BOX5)とpred_rank_box4/box3は重みが違うだけで同じシグナル入力のはずなので、
# 完全一致ではないが強い正の相関(概ね一致する順位)が期待値。1位馬同士の一致率を参考表示。
top1_match_box4 = top1_match_box3 = 0
for r in DB["races"]:
    if r["race_type"] != "normal":
        continue
    top5 = next((h for h in r["horses"] if h["pred_rank"] == 1), None)
    top4 = next((h for h in r["horses"] if h["pred_rank_box4"] == 1), None)
    top3 = next((h for h in r["horses"] if h["pred_rank_box3"] == 1), None)
    if top5 and top4 and top5["umaban"] == top4["umaban"]:
        top1_match_box4 += 1
    if top5 and top3 and top5["umaban"] == top3["umaban"]:
        top1_match_box3 += 1
print(f"BOX5と1位が一致する割合: box4={top1_match_box4}/{n_normal_races} "
      f"box3={top1_match_box3}/{n_normal_races} (参考値、完全一致は想定していない)")

# layoff_flagとsecond_after_layoff_flagはpast1/past2_dateの有無に依存するため、
# no_history/insufficient_historyの割合が新馬戦でほぼ100%になるはず(初出走はpast1_date無し)。
shinba_horses = [h for r in DB["races"] for h in r["horses"] if h.get("race_type") == "shinba"]
shinba_layoff_nohist = sum(1 for h in shinba_horses if h.get("layoff_flag") == "no_history")
print(f"shinba(新馬戦)horses={len(shinba_horses)}  layoff_flag==no_history among them="
      f"{shinba_layoff_nohist} ({100*shinba_layoff_nohist/len(shinba_horses):.1f}%、"
      f"本来は100%に近いはず)")
