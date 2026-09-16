# -*- coding: utf-8 -*-
"""Stage 0-C(3列目改良プラン、2026-09-16): 3連複フォーメーション列別検証ツールの
基礎データ(RACES)を、キャッシュ済みHTMLや旧パッチチェーンに依存せず一次データから
独立に再構築する。

- 母集団: jra_dataset_wide.load()のrace_type=="normal"(障害戦・新馬・未勝利を除外、
  factor_database.json側のnormal 314レースと一致する設計、Stage 0-B確認済み)。
- BOX4/BOX5スコア生値・順位: jra_model_scoring.score_race()で独立に再計算する
  (priorsはStage 0-Bの決定によりwinner_v3.json/winner_box4.json内の固定priorsを使用、
  build_factor_dataset.pyのcompute_priors_for_populationとは意図的に不一致)。
- 290atom評価用の生データ: factor_database.jsonの該当horse行を(race_id, umaban)で
  結合する。結合に失敗した馬が1頭でもあれば、黙ってFalse等にせずその場でエラー停止する
  (現行ツールの既知バグ=障害戦45頭の未結合サイレントFalseを再発させないための方針)。

出力: data/jra_pipeline/jra_formation_base_2026_09_16.json
"""
import json
import pathlib
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
LIB_DIR = PROJECT_ROOT / "scripts" / "jra_model"
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import jra_dataset_wide as JDW  # noqa: E402
import jra_model_scoring as JMS  # noqa: E402

OUT_PATH = DATA_DIR / "jra_formation_base_2026_09_16.json"

# ------------------------------------------------------------ 1) 母集団読み込み(障害戦は
# jra_dataset_wide.race_type_of()で既に除外済み、Stage 0-B確認どおり)
print("loading jra_dataset_wide...")
data = JDW.load(rebuild=False)
all_races = data["races"]
races = [r for r in all_races if r["race_type"] == "normal"]
actual = data["actual"]
print(f"normal races: {len(races)} (全母集団{len(all_races)}中)")

# ------------------------------------------------------------ 2) factor_database.jsonの
# horse行を(race_id, umaban)キーで索引化(290atom評価用の生データ源)
print("loading factor_database.json...")
fd = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
fd_horse_index = {}
fd_race_type_by_id = {}
for r in fd["races"]:
    fd_race_type_by_id[r["race_id"]] = r["race_type"]
    for h in r["horses"]:
        fd_horse_index[(r["race_id"], int(h["umaban"]))] = h

fd_normal_race_ids = {rid for rid, rt in fd_race_type_by_id.items() if rt == "normal"}
tool_race_ids = {r["race_id"] for r in races}
if tool_race_ids != fd_normal_race_ids:
    missing_in_fd = tool_race_ids - fd_normal_race_ids
    extra_in_fd = fd_normal_race_ids - tool_race_ids
    raise SystemExit(
        f"FATAL: レース単位の母集団がfactor_database(normal)と一致しません。"
        f"tool側のみ={sorted(missing_in_fd)}  factor_database側のみ={sorted(extra_in_fd)}"
    )
print("race-level population check: OK (tool側314レース = factor_database normal 314レースと完全一致)")

# ------------------------------------------------------------ 3) priors/weights読み込み
# (Stage 0-B決定: winner_*.json内の固定priorsを使用、build_data2.pyと同一方式)
print("loading priors/weights (winner_v3.json / winner_box4.json)...")
box_cfg = {
    5: {"model_race_type": "normal", "winner_file": "winner_v3.json"},
    4: {"model_race_type": "normal_box4", "winner_file": "winner_box4.json"},
}
weights_by_box, priors_by_box = {}, {}
for n, cfg in box_cfg.items():
    w = json.loads((DATA_DIR / cfg["winner_file"]).read_text(encoding="utf-8"))
    weights_by_box[n] = JMS.load_weights(cfg["model_race_type"])
    priors_by_box[n] = w["priors"]

# ------------------------------------------------------------ 4) 着順・払戻・レース番号を
# race_results/payoutsから読み込む(build_data2.pyと同一方式、jra_dataset_wideのactualは
# 払戻のみで着順を持たないため別読み込みが必要)
print("loading race_results for finish_pos & race_number...")
results_dir = PROJECT_ROOT / "data" / "race_results" / "2026"
finish_by_race = {}
race_number_by_id = {}
used_dates = sorted({r["kaisai_date"] for r in races})
for d in used_dates:
    p = results_dir / f"{d}.csv"
    if not p.exists():
        raise SystemExit(f"FATAL: race_results not found for date {d}: {p}")
    df = pd.read_csv(p, dtype=str)
    for race_id, g in df.groupby("race_id"):
        m = {}
        for _, row in g.iterrows():
            try:
                m[int(row["umaban"])] = row["finish_pos"]
            except (ValueError, TypeError):
                continue
        finish_by_race[race_id] = m
        race_number_by_id[race_id] = g.iloc[0].get("race_number")

# ------------------------------------------------------------ 5) レースごとにスコア計算+
# factor_database結合(fail-fast)
print("scoring races & joining factor_database (fail-fast on join failure)...")
join_failures = []
out_races = []
n_horses_total = 0
for race in races:
    df = race["df"]
    race_id, race_name = race["race_id"], race["race_name"]
    for n in (5, 4):
        score = JMS.score_race(df, race_name, box_cfg[n]["model_race_type"], priors_by_box[n], weights_by_box[n])
        df[f"_score_box{n}"] = score
        df[f"_rank_box{n}"] = JMS.pred_rank_of(score)

    fuku_payout = actual.get(race_id, {}).get("複勝", {})
    finishes = finish_by_race.get(race_id, {})

    horses = []
    for _, row in df.iterrows():
        try:
            umaban = int(row["umaban"])
        except (ValueError, TypeError):
            continue
        n_horses_total += 1
        fd_h = fd_horse_index.get((race_id, umaban))
        if fd_h is None:
            join_failures.append((race_id, umaban))
            continue  # 後段でjoin_failuresが空でないことを検査してfail-fastする

        sc5 = row.get("_score_box5")
        sc4 = row.get("_score_box4")
        rk5 = row.get("_rank_box5")
        rk4 = row.get("_rank_box4")
        fp = finishes.get(umaban)
        try:
            fp_i = int(fp)
        except (TypeError, ValueError):
            fp_i = None

        # atoms: factor_database側のhorse行から、290atom評価に使う生データ全てを引き継ぐ
        # (umaban/waku/horse_name/race_typeは本ファイル側の値と重複するため除外)
        atoms = {k: v for k, v in fd_h.items() if k not in ("umaban", "waku", "horse_name", "race_type")}

        horses.append({
            "umaban": umaban,
            "horse_name": row.get("horse_name"),
            "waku": row.get("waku") if "waku" in df.columns else fd_h.get("waku"),
            "score5": (round(float(sc5) * 100, 1) if sc5 is not None and not pd.isna(sc5) else None),
            "rank5": (int(rk5) if rk5 is not None and not pd.isna(rk5) else None),
            "score4": (round(float(sc4) * 100, 1) if sc4 is not None and not pd.isna(sc4) else None),
            "rank4": (int(rk4) if rk4 is not None and not pd.isna(rk4) else None),
            "finish_pos": fp_i,
            "fuku_payout": int(fuku_payout.get(umaban, 0)),
            "fd_pred_rank_box4": fd_h.get("pred_rank_box4"),  # 参考: factor_database自身のランク(Stage0-B参照)
            "atoms": atoms,
        })

    sanrenpuku_table = actual.get(race_id, {}).get("3連複", {})
    pay_sanrenpuku = {
        "-".join(str(x) for x in sorted(c)): int(p) for c, p in sanrenpuku_table.items() if p > 0
    }

    out_races.append({
        "race_id": race_id,
        "kaisai_date": race["kaisai_date"],
        "racecourse": race["racecourse"],
        "race_name": race_name,
        "race_number": race_number_by_id.get(race_id),
        "block_id": f"{race['kaisai_date']}_{race['racecourse']}",
        "horses": horses,
        "pay_sanrenpuku": pay_sanrenpuku,
    })

if join_failures:
    raise SystemExit(
        f"FATAL: factor_databaseとの結合に失敗した馬が{len(join_failures)}頭あります"
        f"(サイレントFalse化は行わず停止します): {join_failures[:20]}"
        f"{' ...' if len(join_failures) > 20 else ''}"
    )
print(f"join check: OK (全{n_horses_total}頭がfactor_databaseと結合成功)")

meta = {
    "generated_at": pd.Timestamp.now(tz="Asia/Tokyo").isoformat(),
    "n_races": len(out_races),
    "n_horses": n_horses_total,
    "dates": used_dates,
    "population_definition": "jra_dataset_wide.load()のrace_type=='normal'(障害戦・新馬・未勝利を除外)",
    "hurdle_races_excluded_by_design": True,
    "priors_source_decision": (
        "Stage 0-B決定: winner_v3.json(box5)/winner_box4.json(box4)内の固定priorsを使用"
        "(build_factor_dataset.pyのcompute_priors_for_populationとは意図的に不一致、"
        "現行公開中ツールの再現・後継が目的のため)"
    ),
    "join_policy": "factor_databaseとの(race_id, umaban)結合に1件でも失敗すればfail-fastでエラー停止(サイレントFalse化はしない)",
}

payload = {"meta": meta, "races": out_races}
OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, default=str), encoding="utf-8")
print(f"wrote {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)")
print(json.dumps(meta, ensure_ascii=False, indent=2, default=str))
