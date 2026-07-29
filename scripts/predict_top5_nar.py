# -*- coding: utf-8 -*-
"""レースカードの「予想5頭」を、BOX4回収率検証で採用した等重み14シグナルモデル
(nar_signals.py + winner_box4_nar.json、predict_box4_nar.pyと同一ロジック)で
生成し直す。

旧版は scripts/predict_pattern29.py 独自の WEIGHTS_NAR(pattern24、7/25+7/26の
40レースのみで探索・以後レビュー対象外)を使っており、BOX4検証で採用した
モデルと一致していなかった。ユーザー指示によりレースカード表示もBOX4と
同じスコアリングに統一する(頭数は5のまま、BOX4のBOX_N=4とは独立)。

race_names_nar_{date}.csv は既存ファイルをそのまま使う(ヘッダ再取得の
ネットワークアクセスを避ける)。対象日はscratchpad上のrace_names_nar_*.csv
から自動検出するため、新しい日付を取得したら本スクリプトを再実行するだけで
反映される。
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
NAR_MODEL_DIR = PROJECT_ROOT / "scripts" / "nar_model"
sys.path.insert(0, str(NAR_MODEL_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import predict_box4_nar as P  # noqa: E402
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path  # noqa: E402

N_SHOW = 5

parser = argparse.ArgumentParser(
    description="レースカードの予想5頭をBOX4等重みモデルで生成する。"
    "--date省略時はrace_names_nar_*.csvから検出した全日付を処理する。"
)
parser.add_argument("--date", help="単一日付のみ処理する場合(YYYYMMDD)。事前に "
                     "scripts/fetch_newspaper.py --date {date} --circuit nar が必要")
args = parser.parse_args()

dates = sorted(p.stem.replace("race_names_nar_", "") for p in DATA_DIR.glob("race_names_nar_*.csv"))
if args.date:
    if args.date not in dates:
        raise SystemExit(
            f"race_names_nar_{args.date}.csv が見つかりません。先に "
            f"scripts/fetch_newspaper.py --date {args.date} --circuit nar を実行してください。"
        )
    dates = [args.date]
print(f"dates: {dates}")

all_predictions = []
summary = []
for date in dates:
    race_names = pd.read_csv(DATA_DIR / f"race_names_nar_{date}.csv", dtype=str)
    targets = race_names[~race_names["race_name"].str.contains("新馬|未勝利", regex=True, na=False)]

    predicted, missing, errored = 0, [], []
    for _, row in targets.iterrows():
        race_id = row["race_id"]
        path = newspaper_csv_path(race_id)
        if not path.exists():
            missing.append(race_id)
            continue
        try:
            df = pd.read_csv(path, dtype=str, encoding="utf-8")
            if df.empty:
                missing.append(race_id)
                continue
            df = P._drop_scratched(df)
            if df.empty:
                missing.append(race_id)
                continue

            current_class = P._class_ordinal(row["race_name"])
            scored = P.score_race(df, current_class)
            s = scored["_score"].to_numpy(dtype=float)
            order = np.argsort(-np.where(np.isnan(s), -1e18, s), kind="stable")
            top = scored.iloc[order[:N_SHOW]].copy()
            top["race_id"] = race_id
            top.insert(0, "pred_rank", range(1, len(top) + 1))
            top.insert(0, "kaisai_date", date)
            top.insert(1, "racecourse", row["racecourse"])
            top.insert(2, "race_name", row["race_name"])
            all_predictions.append(
                top[["kaisai_date", "racecourse", "race_name", "race_id", "pred_rank", "waku", "umaban",
                     "horse_name", "bias_ninki", "bias_win_odds", "bias_horse_weight", "_score"]]
            )
            predicted += 1
        except Exception as exc:  # noqa: BLE001 - 1レースの異常で全体を止めない
            errored.append((race_id, repr(exc)))
            continue

    summary.append((date, len(targets), predicted, missing, errored))

for date, n_targets, predicted, missing, errored in summary:
    print(f"{date}: targets={n_targets} predicted={predicted} missing={len(missing)}->{missing} "
          f"errored={len(errored)}->{errored}")

result = pd.concat(all_predictions, ignore_index=True)
for date, g in result.groupby("kaisai_date"):
    out = DATA_DIR / f"predictions_nar_{date}.csv"
    g.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"wrote {out} ({g['race_id'].nunique()} races, model={P.MODEL_LABEL})")
