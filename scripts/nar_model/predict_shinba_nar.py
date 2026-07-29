# -*- coding: utf-8 -*-
"""地方競馬(NAR)新馬戦モデル。predict_shinba.py(JRA)の6シグナルロジック
(jt/waku/train/sire/bms/comment、いずれも列名ベースでJRA/NAR非依存)をそのまま
動的importして再利用し、重みはJRAのwinner_shinba.json(jt 89.4%集中)をユーザー決定に
基づき暫定流用する。NAR専用の重み探索は行わない(後日データが増えたら差し替え)。

対象は data/race_results/nar 配下でrace_nameに「新馬」を含む全レース。2026-07-27時点で
高知の2レース(202654072601「2歳新馬 弐」・202654072602「2歳新馬 壱」、ともに7/26・
7頭立て)のみ。NARの馬柱データには厩舎コメント欄・調教評価欄が存在しないため、
train(1.8%)・comment(8.2%)の計約10%は常にNaN→自動的に他シグナルへ再配分され、
実質ほぼjt(騎手・調教師勝率)一本の予想になる。この点は必ずレポートに明記すること。

サンプル数が2レースしかないため、box_return的な回収率検証も行うが数値は参考程度。
"""
import importlib.util
import itertools
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
UNIT = 100

spec = importlib.util.spec_from_file_location(
    "predict_shinba_mod", Path(__file__).resolve().parent / "predict_shinba.py"
)
shinba_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shinba_mod)

import sys  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT))
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402


def load_nar_shinba_races() -> list:
    """data/race_results/nar配下の各開催日CSVから、race_nameに「新馬」を含むレースだけを
    抽出する(predict_shinba.load_shinba_racesのNAR版。パスと対象ディレクトリのみ変更)。"""
    entries = []
    results_dir = PROJECT_ROOT / "data" / "race_results" / "nar" / "2026"
    for date_csv in sorted(results_dir.glob("2026*.csv")):
        date = date_csv.stem
        results = pd.read_csv(date_csv, dtype=str)
        races = results[["race_id", "race_name", "racecourse"]].drop_duplicates("race_id")
        shinba = races[races["race_name"].str.contains("新馬", na=False, regex=True)]
        for _, row in shinba.iterrows():
            entries.append({
                "race_id": row["race_id"], "kaisai_date": date,
                "racecourse": row["racecourse"], "race_name": row["race_name"],
            })
    return entries


def parse_combo(bet_type: str, combo_text: str):
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
    if "→" in combo_text:
        return None
    return frozenset(int(x) for x in combo_text.split("-"))


BET_TYPES = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]


def main() -> None:
    targets = load_nar_shinba_races()
    print(f"target NAR shinba races: {len(targets)} -> {[t['race_id'] for t in targets]}")
    if len(targets) < 5:
        print(f"WARNING: サンプル数が{len(targets)}レースのみ。統計的な信頼性は極めて低く、参考値として扱うこと。")

    raw = []
    missing = []
    for e in targets:
        path = newspaper_csv_path(e["race_id"])
        if not path.exists():
            missing.append(e["race_id"])
            continue
        df = pd.read_csv(path, dtype=str, encoding="utf-8")
        if df.empty:
            missing.append(e["race_id"])
            continue
        df = shinba_mod._drop_scratched(df)
        if df.empty:
            missing.append(e["race_id"])
            continue
        raw.append({**e, "df": df})

    if not raw:
        print("no NAR shinba races with usable newspaper data")
        return

    # priorはこの2レース自身の母集団から計算する(JRAの14レースpriorとは別母集団 -
    # 競馬場・クラス体系が異なるNARに対しJRAのpriorを流用するのは不適切なため)。
    priors = shinba_mod.compute_priors([e["df"] for e in raw])
    print("shrinkage priors (NAR shinba, own population):", {k: round(v, 2) for k, v in priors.items()})

    # 重みはJRAのwinner_shinba.jsonをユーザー決定により暫定流用(NAR専用探索は未実施)。
    weights, weights_source = shinba_mod.load_weights()
    print(f"weights ({weights_source}, JRAから暫定流用):", weights)

    all_predictions = []
    errored = []
    for e in raw:
        try:
            scored = shinba_mod.score_shinba_race(e["df"], priors, weights)
            scored = scored.sort_values("_score", ascending=False, kind="stable").reset_index(drop=True)
            top5 = scored.head(5).copy()
            top5["race_id"] = e["race_id"]
            top5.insert(0, "pred_rank", range(1, len(top5) + 1))
            top5.insert(0, "kaisai_date", e["kaisai_date"])
            top5.insert(1, "racecourse", e["racecourse"])
            top5.insert(2, "race_name", e["race_name"])
            all_predictions.append(
                top5[["kaisai_date", "racecourse", "race_name", "race_id", "pred_rank", "waku", "umaban",
                      "horse_name", "bias_ninki", "bias_win_odds", "bias_horse_weight", "_score"]]
            )
        except Exception as exc:  # noqa: BLE001 - 1レースの異常で全体を止めない
            errored.append((e["race_id"], repr(exc)))
            continue

    if not all_predictions:
        print("no predictions produced")
        return

    result = pd.concat(all_predictions, ignore_index=True)
    out_path = DATA_DIR / "predictions_shinba_nar.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"predicted races: {result['race_id'].nunique()}")
    print(f"missing newspaper csv: {len(missing)} -> {missing}")
    print(f"errored races: {len(errored)} -> {errored}")
    print(f"wrote {out_path}")

    # --- 簡易BOX5回収率検証(サンプル2レースのみ、参考値) ---
    dates = sorted(result["kaisai_date"].unique())
    payout_frames = []
    for d in dates:
        p = PROJECT_ROOT / "data" / "payouts" / "nar" / "2026" / f"{d}.csv"
        if p.exists():
            payout_frames.append(pd.read_csv(p, dtype=str))
    if not payout_frames:
        print("payoutsデータが見つからないため回収率検証はスキップします。")
        return
    payouts = pd.concat(payout_frames, ignore_index=True)
    payouts["payout"] = payouts["payout"].astype(int)

    box_results = {bt: {"stake": 0, "return": 0, "hit_races": 0, "race_count": 0} for bt in BET_TYPES}
    for race_id, g in result.groupby("race_id", sort=False):
        umabans = g["umaban"].astype(int).tolist()
        wakus = sorted(set(pd.to_numeric(g["waku"], errors="coerce").dropna().astype(int).tolist()))
        race_payouts = payouts[payouts["race_id"] == race_id]

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
            box_results[bt]["stake"] += stake
            box_results[bt]["return"] += ret
            box_results[bt]["race_count"] += 1
            box_results[bt]["hit_races"] += 1 if ret > 0 else 0

        settle("単勝", umabans)
        settle("複勝", umabans)
        settle("枠連", [frozenset(c) for c in itertools.combinations(wakus, 2)])
        settle("馬連", [frozenset(c) for c in itertools.combinations(umabans, 2)])
        settle("ワイド", [frozenset(c) for c in itertools.combinations(umabans, 2)])
        settle("馬単", list(itertools.permutations(umabans, 2)))
        settle("3連複", [frozenset(c) for c in itertools.combinations(umabans, 3)])
        settle("3連単", list(itertools.permutations(umabans, 3)))

    summary = []
    for bt in BET_TYPES:
        r = box_results[bt]
        rate = (r["return"] / r["stake"] * 100) if r["stake"] else 0.0
        summary.append({
            "bet_type": bt, "races": r["race_count"], "hit_races": r["hit_races"],
            "hit_rate_pct": round(r["hit_races"] / r["race_count"] * 100, 1) if r["race_count"] else 0,
            "total_stake": r["stake"], "total_return": r["return"], "return_rate_pct": round(rate, 1),
        })
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(DATA_DIR / "box_return_summary_shinba_nar.csv", index=False, encoding="utf-8-sig")
    print(f"\n[参考値・{result['race_id'].nunique()}レースのみ] BOX5回収率:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
