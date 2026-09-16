# -*- coding: utf-8 -*-
"""「馬柱データ2」用のデータ生成。現行本番の重み(winner_v3.json=pattern83/BOX5、
winner_box4.json=pattern19/BOX4、winner_box3.json=pattern95/BOX3、2026-09-11のBOX3変更検討は
2体のサブエージェントレビューの結果、現行維持で決着)を使い、jra_model_scoring.pyの本番同一
ロジックでBOX5/4/3それぞれのスコア・順位を計算する。対象は現時点でnewspaper CSVが揃っている
全319レース(18開催日、2026-07-11〜09-06、通常戦のみ)。

出力: data2_races.json (レースごとの馬柱データ+3モデルのスコア/順位+実際の着順・払戻)
      data2_meta.json (集計・母集団情報)
"""
import sys, json, pathlib
import numpy as np
import pandas as pd

LIB_DIR = r"c:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_model"
PROJECT_ROOT = r"c:\Users\yuyou\Desktop\新しい作業場所"
sys.path.insert(0, LIB_DIR)
sys.path.insert(0, PROJECT_ROOT)
import jra_dataset
import jra_model_scoring as JMS
import jra_signals as JS
import jra_eval as JE
import jra_backtest as JB

SP = pathlib.Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad")
DATA_DIR = pathlib.Path(PROJECT_ROOT) / "data" / "jra_pipeline"

print("loading jra_dataset...")
data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
print(f"races: {len(races)}  dates: {data['dates']}")

BOX_CFG = {
    5: {"race_type": "normal", "winner_file": "winner_v3.json"},
    4: {"race_type": "normal_box4", "winner_file": "winner_box4.json"},
    3: {"race_type": "normal_box3", "winner_file": "winner_box3.json"},
}
weights_by_box, priors_by_box, meta_by_box = {}, {}, {}
for n, cfg in BOX_CFG.items():
    w = json.loads((DATA_DIR / cfg["winner_file"]).read_text(encoding="utf-8"))
    weights_by_box[n] = JMS.load_weights(cfg["race_type"])
    priors_by_box[n] = w["priors"]
    meta_by_box[n] = {"pattern_id": w["pattern_id"], "fitted_on": w["fitted_on"]}

# ------------------------------------------------------------ 実際の着順・レース情報(race_results)を読み込む
print("loading race_results for finish_pos & race meta...")
results_dir = pathlib.Path(PROJECT_ROOT) / "data" / "race_results" / "2026"
finish_by_race = {}
race_meta_by_id = {}
for d in data["dates"]:
    p = results_dir / f"{d}.csv"
    if not p.exists():
        continue
    df = pd.read_csv(p, dtype=str)
    for race_id, g in df.groupby("race_id"):
        m = {}
        for _, row in g.iterrows():
            try:
                m[int(row["umaban"])] = {
                    "finish_pos": row["finish_pos"], "jockey_name": row.get("jockey_name", ""),
                    "trainer_name": row.get("trainer_name", ""),
                }
            except (ValueError, TypeError):
                continue
        finish_by_race[race_id] = m
        first = g.iloc[0]
        race_meta_by_id[race_id] = {
            "race_number": first.get("race_number"), "surface": first.get("surface"),
            "distance_m": first.get("distance_m"), "weather": first.get("weather"),
            "going": first.get("going"), "start_time": first.get("start_time"),
        }

# ------------------------------------------------------------ レースごとにスコア計算(race["df"]へ列追加、in-place)
n_scored = 0
for race in races:
    df = race["df"]
    race_name = race["race_name"]
    for n in (5, 4, 3):
        score = JMS.score_race(df, race_name, BOX_CFG[n]["race_type"], priors_by_box[n], weights_by_box[n])
        df[f"_score_box{n}"] = score
        df[f"_rank_box{n}"] = JMS.pred_rank_of(score)
    n_scored += 1
print(f"scored: {n_scored} races")

# ------------------------------------------------------------ box5/4/3各モデルのpicksをpayout settle(集計用)
def picks_for(box_n, race, score_col):
    df = race["df"]
    s = df[score_col]
    order = np.argsort(-s.fillna(-1e18).to_numpy(), kind="stable")
    return order[:box_n]

box_return_summary = {}
for n in (5, 4, 3):
    ev = JE.Evaluator(races, actual, box_n=n)
    mkt = ev.evaluate(JE.market_picks(races, n))
    all_picks = [picks_for(n, r, f"_score_box{n}") for r in races]
    model = ev.evaluate(all_picks)
    full_table = ev.full_table(all_picks)
    box_return_summary[n] = {
        "market_excess_pt": model["excess"], "market_model": model["model"], "market": mkt["model"],
        "full_table": full_table.to_dict("records"),
    }

# ------------------------------------------------------------ 馬柱データ組立(実着順・払戻とマージ)
out_races = []
for race in races:
    df = race["df"]
    race_id, race_name = race["race_id"], race["race_name"]
    finishes = finish_by_race.get(race_id, {})
    fuku_payout = actual.get(race_id, {}).get("複勝", {})
    horses = []
    for _, row in df.iterrows():
        try:
            umaban = int(row["umaban"])
        except (ValueError, TypeError):
            continue
        fin = finishes.get(umaban, {})
        horses.append({
            "umaban": umaban,
            "waku": row.get("waku"),
            "name": row.get("horse_name") or row.get("bias_horse_name"),
            "ninki": row.get("bias_ninki"),
            "odds": row.get("bias_win_odds"),
            "weight": row.get("bias_horse_weight"),
            "jockey": row.get("bias_jockey") or fin.get("jockey_name"),
            "trainer": row.get("bias_trainer") or fin.get("trainer_name"),
            "sex_age": row.get("bias_sex_age"),
            "kinryo": row.get("bias_weight_carried"),
            "score5": row.get("_score_box5"), "rank5": row.get("_rank_box5"),
            "score4": row.get("_score_box4"), "rank4": row.get("_rank_box4"),
            "score3": row.get("_score_box3"), "rank3": row.get("_rank_box3"),
            "finish_pos": fin.get("finish_pos"),
            "fuku_payout": fuku_payout.get(umaban, 0),
        })

    rmeta = race_meta_by_id.get(race_id, {})
    out_races.append({
        "race_id": race_id, "kaisai_date": race["kaisai_date"], "racecourse": race["racecourse"],
        "race_name": race_name, "horses": horses,
        "race_number": rmeta.get("race_number"), "surface": rmeta.get("surface"),
        "distance_m": rmeta.get("distance_m"), "weather": rmeta.get("weather"),
        "going": rmeta.get("going"), "start_time": rmeta.get("start_time"),
    })

print(f"scored: {n_scored} races")

meta = {
    "n_races": len(races), "dates": data["dates"],
    "box_meta": meta_by_box, "box_return_summary": box_return_summary,
    "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
}

(SP / "data2_races.json").write_text(json.dumps(out_races, ensure_ascii=False, default=str), encoding="utf-8")
(SP / "data2_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print("wrote data2_races.json", (SP / "data2_races.json").stat().st_size, "bytes")
print("wrote data2_meta.json")
print(json.dumps(meta["box_return_summary"], ensure_ascii=False, indent=2, default=str)[:2000])
