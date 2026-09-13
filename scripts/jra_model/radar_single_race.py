# -*- coding: utf-8 -*-
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
import pandas as pd
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_radar_categories as RC
import jra_signals as JS
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path

RACE_ID = "202604030306"
KAISAI_DATE = "20260829"
RACECOURSE = "新潟"
RACE_NAME = "新発田城特別"

print("既存データセットからpriors/lap33基盤をロード中...")
data = jra_dataset.load(rebuild=False)
races_all = data["races"]
priors_all = JS.make_priors([r["df"] for r in races_all])
lap33_lookup = L33.load_lap33_lookup()
race_meta = L33.load_race_surface_distance()
history_index = L33.build_history_index()

path = newspaper_csv_path(RACE_ID)
print("newspaper csv:", path, path.exists())
df = pd.read_csv(path, dtype=str, encoding="utf-8")
df = JS._drop_scratched(df)
ped_cache = Path(jra_dataset.PEDIGREE_FEATURES_CACHE)
if ped_cache.exists():
    ped = pd.read_csv(ped_cache, dtype=str, encoding="utf-8")
    race_ped = ped[ped["race_id"] == RACE_ID].drop(columns=["race_id"])
    df = df.merge(race_ped, on="horse_id", how="left")

race = {"race_id": RACE_ID, "kaisai_date": KAISAI_DATE, "racecourse": RACECOURSE,
        "race_name": RACE_NAME, "df": df}

records = RC.build_axis_matrix([race], priors_all, history_index, lap33_lookup, race_meta)
rec = records[0]
print("n_race(有効カテゴリ数):", rec["n_race"], "dropped:", rec["dropped"])
axis = rec["axis"]
print(axis)

# horse names / umaban for labeling
name_col = "horse_name" if "horse_name" in df.columns else None
uma_col = "umaban" if "umaban" in df.columns else None
labels = []
for i in df.index:
    nm = df.loc[i, name_col] if name_col else str(i)
    um = df.loc[i, uma_col] if uma_col else ""
    labels.append(f"{um} {nm}")

box5 = RC.linear_picks(records, 5)[0]
print("\nbox5(線形平均 上位5、行番号):", box5)
print("box5馬名:", [labels[i] for i in box5])

out = {
    "race_id": RACE_ID, "race_name": RACE_NAME, "kaisai_date": KAISAI_DATE, "racecourse": RACECOURSE,
    "categories": RC.CATEGORIES,
    "n_race": rec["n_race"], "dropped": rec["dropped"],
    "horses": [
        {"umaban": df.loc[i, uma_col] if uma_col else None,
         "name": df.loc[i, name_col] if name_col else None,
         "axis": {cat: (None if pd.isna(axis.loc[i, cat]) else float(axis.loc[i, cat])) for cat in RC.CATEGORIES}}
        for i in df.index
    ],
    "box5_idx": [int(x) for x in box5],
}
outp = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad\radar_single_race_202604030306.json")
outp.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print("wrote", outp)
