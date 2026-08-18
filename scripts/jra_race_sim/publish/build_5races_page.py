# -*- coding: utf-8 -*-
"""8/2 追加5レース(苗場特別・中京スポニチ賞・名鉄杯・ポプラステークス・苫小牧特別)の
「展開予想の答え合わせ」を1ページにまとめて生成する。

2026-08-18永続化時の注記: このスクリプトが読む個別入力ファイル(horse_info_{race_id}.txt・
race5_margins.csv・{race_id}の実測/simラップCSV)は、旧セッションのscratchpad上でのみ
作成された一回限りの中間生成物であり、この移行では複製していない(他の永続化対象
(video_positions・エンジン本体)と異なりgit管理下の入力データとして整備されていないため。
このページ自体もmodel_update_panel_html()を使わない別系統の一回限りのハブページで、
恒常的な再生成対象からは外れている)。このスクリプトを再実行する場合はBASE配下に
これらの入力ファイルを別途用意すること。"""
import io
import json
from pathlib import Path

BASE = Path(r"C:\Users\yuyou\Desktop\新しい作業場所\scripts\jra_race_sim\_workdir")

WAKU_COLORS = {1: '#FFFFFF', 2: '#1A1A1A', 3: '#D8332B', 4: '#1660B0', 5: '#EFC22A', 6: '#1F8A45', 7: '#E8791A', 8: '#E0538C'}


def load_horses(race_id):
    lines = io.open(BASE / f"horse_info_{race_id}.txt", encoding="utf-8").read().strip().splitlines()
    horses = []
    for line in lines:
        umaban, name, waku, style, est, tt = line.split("\t")
        horses.append({
            "umaban": int(umaban), "name": name, "waku": int(waku),
            "runningStyle": style, "isEstimated": est == "True",
            "simTotalTime": float(tt),
        })
    return horses


RACES = [
    {
        "slug": "naeba", "race_id": "202604020406", "race_name": "苗場特別(2勝)",
        "eyebrow": "2026年8月2日 新潟6R・苗場特別", "actual_csv": "naeba_actual_timeline.csv",
        "sim_csv": "naeba_sim_dense.csv", "goal_t": 112.40,
        "cp": [[19.48, "1角"], [43.07, "2角"], [65.63, "3角"], [89.98, "4角"], [112.40, "ゴール"]],
        "mae": 3.44, "n_horses": 15, "n_estimated": 6,
        "default_highlight": ["1", "5"],
        "story_intro": "優勝馬#1ブレトワルダは15頭中4位評価と大きくは外していませんが、シムが単独1位評価した#5ナイトスラッガーは3着まででした——",
        "story_cards": [
            ("1", "優勝 #1 ブレトワルダ", "4角6位から直線で抜け出し<b>1着(1:52.4)</b>",
             "シム順位<b>4位</b>(15頭中)、走破タイム112.65秒(優勝タイムより0.25秒遅い評価)"),
            ("5", "シム1位評価 #5 ナイトスラッガー", "4角8位から追い込むも<b>3着(1:52.8)</b>",
             "全頭中もっとも速いタイム(110.50秒)と評価、<b>単独1位級の評価</b>だった"),
        ],
        "story_footer": "優勝馬の評価自体は大きく外していない一方、シムが「最速」と見た馬は着外ではないものの優勝までは届かず、上位互換的なズレにとどまりました。",
    },
    {
        "slug": "chukyosponichi", "race_id": "202607020406", "race_name": "中京スポニチ賞",
        "eyebrow": "2026年8月2日 中京6R・中京スポニチ賞", "actual_csv": "chukyosponichi_actual_timeline.csv",
        "sim_csv": "chukyosponichi_sim_dense.csv", "goal_t": 78.70,
        "cp": [[30.71, "3角"], [55.30, "4角"], [78.70, "ゴール"]],
        "mae": 3.04, "n_horses": 9, "n_estimated": 3,
        "default_highlight": ["6", "4"],
        "story_intro": "シムが単独1位評価した#4アイニードユーは実際2着(クビ差)まで肉薄しており、5レース中もっとも予想が惜しかったケースです——",
        "story_cards": [
            ("6", "優勝 #6 トミーバローズ", "4角6位から差し切り<b>1着(1:18.7)</b>",
             "シム順位<b>4位</b>(9頭中)、中位評価にとどまっていた"),
            ("4", "シム1位評価 #4 アイニードユー", "4角2位のまま伸びて<b>2着(1:18.8、クビ差)</b>",
             "全頭中もっとも速いタイム(80.50秒)と評価、<b>単独1位級の評価</b>だった"),
        ],
        "story_footer": "シム1位評価馬が僅差の2着に食い込んだ点で、5レース中もっともシミュレーションの評価と実際の結果が近かったレースです。",
    },
    {
        "slug": "meitetsuhai", "race_id": "202607020407", "race_name": "名鉄杯(L)",
        "eyebrow": "2026年8月2日 中京7R・名鉄杯", "actual_csv": "meitetsuhai_actual_timeline.csv",
        "sim_csv": "meitetsuhai_sim_dense.csv", "goal_t": 112.50,
        "cp": [[17.26, "1角"], [39.71, "2角"], [64.94, "3角"], [86.26, "4角"], [112.50, "ゴール"]],
        "mae": 2.40, "n_horses": 9, "n_estimated": 2,
        "default_highlight": ["9", "7"],
        "story_intro": "優勝馬#9クールミラボーはシムも3位評価と好評価、シム単独1位評価の#7サイレンサーナドーも実際3着とほぼ一致しました——",
        "story_cards": [
            ("9", "優勝 #9 クールミラボー", "1角〜4角すべて1位を維持し、そのまま<b>逃げ切り1着(1:52.5)</b>",
             "シム順位<b>3位</b>(9頭中)、優勝評価に迫る高評価だった"),
            ("7", "シム1位評価 #7 サイレンサーナドー", "3角・4角とも3位を維持し<b>3着(1:53.3)</b>",
             "全頭中もっとも速いタイム(111.60秒)と評価、<b>単独1位級の評価</b>で今回は着順もほぼ一致した"),
        ],
        "story_footer": "順位誤差(MAE)は5レース中もっとも小さく、2026-08-15のパラメータ更新前はシムの単独1位評価が#5ソーニーイシュー(最下位に終わった馬)でしたが、更新後は#7サイレンサーナドー(3着)に入れ替わり、シムの評価と実際の結果がより一致するようになりました。",
    },
    {
        "slug": "poplar", "race_id": "202601010410", "race_name": "ポプラステークス(3勝)",
        "eyebrow": "2026年8月2日 札幌10R・ポプラステークス", "actual_csv": "poplar_actual_timeline.csv",
        "sim_csv": "poplar_sim_dense.csv", "goal_t": 104.60,
        "cp": [[7.74, "1角"], [36.52, "2角"], [53.06, "3角"], [82.38, "4角"], [104.60, "ゴール"]],
        "mae": 2.86, "n_horses": 10, "n_estimated": 4,
        "default_highlight": ["8", "10"],
        "story_intro": "シムが10頭中もっとも遅いと評価した#8ヘニーガイストが優勝、逆に単独1位評価だった#10フルールドールは9着に終わった、5レース中もっとも劇的な逆転です——",
        "story_cards": [
            ("8", "優勝 #8 ヘニーガイスト", "4角3位から抜け出し<b>1着(1:44.6)</b>",
             "シム順位<b>10位</b>(10頭中最下位評価)、全頭中もっとも遅いと評価していた"),
            ("10", "シム1位評価 #10 フルールドール", "4角9位から伸びず<b>9着</b>",
             "全頭中もっとも速いタイム(102.45秒)と評価、<b>単独1位級の評価</b>だった"),
        ],
        "story_footer": "評価が完全に逆転した組み合わせで、シムが基礎スピード・スタミナから見積もった「地力」評価と、実際のレース運び・位置取りが噛み合わなかった典型例です。",
    },
    {
        "slug": "tomakomai", "race_id": "202601010412", "race_name": "苫小牧特別(2勝)",
        "eyebrow": "2026年8月2日 札幌12R・苫小牧特別", "actual_csv": "tomakomai_actual_timeline.csv",
        "sim_csv": "tomakomai_sim_dense.csv", "goal_t": 69.30,
        "cp": [[21.68, "3角"], [53.80, "4角"], [69.30, "ゴール"]],
        "mae": 2.38, "n_horses": 14, "n_estimated": 8,
        "default_highlight": ["5", "10"],
        "story_intro": "優勝馬#5マテンロウサンはシムも2位評価と好評価だった一方、単独1位評価の#10ベルビースタローンは3角・4角とも2位につけながら6着に後退しました——",
        "story_cards": [
            ("5", "優勝 #5 マテンロウサン", "3角・4角とも4位から抜け出し<b>1着(1:09.3)</b>",
             "シム順位<b>2位</b>(14頭中)、優勝評価に迫る高評価だった"),
            ("10", "シム1位評価 #10 ベルビースタローン", "3角・4角とも2位につけるも<b>6着</b>に後退",
             "全頭中もっとも速いタイム(68.65秒)と評価、<b>単独1位級の評価</b>だった"),
        ],
        "story_footer": "優勝馬の評価自体は良好でしたが、出走14頭中8頭が推定フォールバック(実測の持続タイムデータなし)という構成上、評価の不確実性が他レースより高い点に留意が必要です。",
    },
]

import csv as _csv
_margins_by_race = {}
with io.open(BASE / "race5_margins.csv", encoding="utf-8-sig") as f:
    for row in _csv.DictReader(f):
        _margins_by_race.setdefault(row["race_id"], []).append(
            [int(row["finish_pos"]), int(row["umaban"]), row["time"], row["margin"]]
        )

for r in RACES:
    r["horses"] = load_horses(r["race_id"])
    r["actual_csv_text"] = io.open(BASE / r["actual_csv"], encoding="utf-8").read().strip()
    r["sim_csv_text"] = io.open(BASE / r["sim_csv"], encoding="utf-8").read().strip()
    r["margins"] = _margins_by_race[r["race_id"]]
    del r["actual_csv"], r["sim_csv"]

json.dump(RACES, io.open(BASE / "race5_page_data.json", "w", encoding="utf-8"), ensure_ascii=False)
print("wrote race5_page_data.json, races=", len(RACES))
