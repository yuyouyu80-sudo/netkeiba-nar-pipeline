# -*- coding: utf-8 -*-
"""地方競馬(NAR)通常戦モデルのBOX5回収率検証。box_return.py(JRA)を移植し、
predictions_nar_{date}.csv(predict_pattern29.py --circuit nar の出力)と
data/payouts/nar/2026/{date}.csv を突き合わせる。

JRAとの違い:
  - 対象日付は検証済みnewspaperがある日付を自動検出する(ハードコードしない)。
  - 金沢の一部レース(202646072603-607)では「枠単」のpayout行が「枠連」として
    誤ラベル付けされたまま保存されており、combinationが「→」区切りになっている。
    この行を枠連としてparseすると誤って合算され回収率が水増しされるため、
    search_patterns_nar.pyと同じロジックで除外する。
  - waku(枠番)列は地方競馬の馬柱データに存在しない(厩舎コメント欄由来のため)ので
    枠連は常にstake=0(検証対象外)になる。
"""
import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
UNIT = 100  # 1点あたりの金額(地方競馬も最低100円単位)

all_pred_frames = []
for p in sorted(DATA_DIR.glob("predictions_nar_*.csv")):
    all_pred_frames.append(pd.read_csv(p, dtype=str))
all_pred = pd.concat(all_pred_frames, ignore_index=True)

# payoutsが存在する(=レースが終了し結果確定済みの)日付だけを対象にする。pending日
# (結果未確定)の予想まで含めると、payoutsが無いため「全レース不的中」として計上され
# 回収率が不当に押し下げられてしまう。
all_dates = sorted(all_pred["kaisai_date"].unique())
DATES = [d for d in all_dates if (PROJECT_ROOT / "data" / "payouts" / "nar" / "2026" / f"{d}.csv").exists()]
skipped = sorted(set(all_dates) - set(DATES))
if skipped:
    print(f"payouts未確定のためスキップ(検証待ち日付): {skipped}")

pred = all_pred[all_pred["kaisai_date"].isin(DATES)].reset_index(drop=True)
pred["umaban"] = pred["umaban"].astype(int)
pred["waku"] = pd.to_numeric(pred["waku"], errors="coerce")

print(f"predictions dates: {DATES}  races: {pred['race_id'].nunique()}")

payout_frames = []
for d in DATES:
    p = PROJECT_ROOT / "data" / "payouts" / "nar" / "2026" / f"{d}.csv"
    if p.exists():
        payout_frames.append(pd.read_csv(p, dtype=str))
payouts = pd.concat(payout_frames, ignore_index=True)
payouts["payout"] = payouts["payout"].astype(int)


def parse_combo(bet_type: str, combo_text: str):
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
    # NARの一部レース(金沢 202646072603-607)では「枠単」が「枠連」に誤ラベル付け
    # されたままcombinationだけ「→」区切りで保存されている。本来の枠連ではないため
    # Noneを返しsettle側で除外する(search_patterns_nar.pyと同じ対処)。
    if "→" in combo_text:
        return None
    return frozenset(int(x) for x in combo_text.split("-"))


BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]

results = {bt: {"stake": 0, "return": 0, "hit_races": 0, "race_count": 0} for bt in BET_TYPES}
per_race_rows = []

for race_id, g in pred.groupby("race_id", sort=False):
    umabans = g["umaban"].tolist()
    wakus = sorted(set(int(w) for w in g["waku"].dropna().tolist()))
    race_payouts = payouts[payouts["race_id"] == race_id]

    race_row = {"race_id": race_id}

    def settle(bt, combos):
        stake = len(combos) * UNIT
        rows = race_payouts[race_payouts["bet_type"] == bt]
        actual_map = {}
        for c, p in zip(rows["combination"], rows["payout"]):
            key = parse_combo(bt, c)
            if key is None:
                continue
            actual_map[key] = actual_map.get(key, 0) + p
        ret = sum(actual_map.get(c, 0) for c in combos)
        results[bt]["stake"] += stake
        results[bt]["return"] += ret
        results[bt]["race_count"] += 1
        results[bt]["hit_races"] += 1 if ret > 0 else 0
        race_row[bt] = (stake, ret)

    # 単勝 / 複勝: box = bet each of the 5 selections individually
    settle("単勝", umabans)
    settle("複勝", umabans)
    # 枠連: box over the distinct waku (frame) numbers represented by the 5 picks
    settle("枠連", [frozenset(c) for c in itertools.combinations(wakus, 2)])
    # 馬連 / ワイド: unordered pairs of the 5 selected horses
    settle("馬連", [frozenset(c) for c in itertools.combinations(umabans, 2)])
    settle("ワイド", [frozenset(c) for c in itertools.combinations(umabans, 2)])
    # 馬単: ordered pairs
    settle("馬単", list(itertools.permutations(umabans, 2)))
    # 3連複: unordered triples
    settle("3連複", [frozenset(c) for c in itertools.combinations(umabans, 3)])
    # 3連単: ordered triples
    settle("3連単", list(itertools.permutations(umabans, 3)))

    per_race_rows.append(race_row)

summary = []
for bt in BET_TYPES:
    r = results[bt]
    rate = (r["return"] / r["stake"] * 100) if r["stake"] else 0.0
    summary.append(
        {
            "bet_type": bt,
            "races": r["race_count"],
            "hit_races": r["hit_races"],
            "hit_rate_pct": round(r["hit_races"] / r["race_count"] * 100, 1) if r["race_count"] else 0,
            "total_stake": r["stake"],
            "total_return": r["return"],
            "return_rate_pct": round(rate, 1),
        }
    )

summary_df = pd.DataFrame(summary)
summary_df.to_csv(DATA_DIR / "box_return_summary_nar.csv", index=False, encoding="utf-8-sig")
print(summary_df.to_string(index=False))

# per-race detail for the artifact
detail_rows = []
for row in per_race_rows:
    race_id = row["race_id"]
    for bt in BET_TYPES:
        stake, ret = row[bt]
        detail_rows.append({"race_id": race_id, "bet_type": bt, "stake": stake, "return": ret})
detail_df = pd.DataFrame(detail_rows)
detail_df.to_csv(DATA_DIR / "box_return_detail_nar.csv", index=False, encoding="utf-8-sig")
print("races processed:", len(per_race_rows))
