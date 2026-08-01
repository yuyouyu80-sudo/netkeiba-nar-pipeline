# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-01)の新規候補シグナル検証: 直前連続2暦日・同競馬場の枠番別勝率。

「馬場は数日単位で偏ることがある」という着眼から、course_analysisの全期間waku勝率とは
別に、レース当日の直前2"連続暦日"(D-1・D-2)における同競馬場での枠番別勝率を追加候補
シグナル化し、project_nar_factor_v2_rejected_finding と同じLOBO OOF二値採否の考え方で
検証する(候補は1つのみのため、8通り選抜のselection_optimismは行わず、基準との
単純な差分+block_bootstrapで判断する)。

馬柱CSVの生waku列はNARでは常に空(nar_backtest.pyの既知の制約)なため、各馬の枠番は
course_analysisが解決済みの ca_waku_category_label("N枠")から取り出す。
"""
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nar_dataset
import nar_eval as NE
import nar_signals as NS

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "data" / "race_results" / "nar" / "2026"
WINNER_BOX4 = PROJECT_ROOT / "data" / "nar_pipeline" / "winner_box4_nar.json"


def load_results_history() -> pd.DataFrame:
    frames = []
    for f in sorted(RESULTS_DIR.glob("2026*.csv")):
        df = pd.read_csv(f, dtype=str, usecols=["race_id", "racecourse", "waku", "finish_pos"])
        df["kaisai_date"] = f.stem
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def _date_minus(d: str, n: int) -> str:
    return (datetime.strptime(d, "%Y%m%d") - timedelta(days=n)).strftime("%Y%m%d")


def build_recent2d_table(history: pd.DataFrame, target_keys: set) -> dict:
    """(racecourse, kaisai_date) -> {waku_num(int): (win_rate_pct, runs)}。
    D-1・D-2の両方に該当競馬場の開催実績が無ければキー自体を作らない(=欠損)。"""
    table = {}
    for course, date in sorted(target_keys):
        d1, d2 = _date_minus(date, 1), _date_minus(date, 2)
        window = history[(history["racecourse"] == course) & (history["kaisai_date"].isin([d1, d2]))]
        if not ((window["kaisai_date"] == d1).any() and (window["kaisai_date"] == d2).any()):
            continue
        per_waku = {}
        for waku_str, g in window.groupby("waku"):
            try:
                w = int(waku_str)
            except (ValueError, TypeError):
                continue
            runs = len(g)
            wins = int((g["finish_pos"] == "1").sum())
            per_waku[w] = (wins / runs * 100.0, runs)
        table[(course, date)] = per_waku
    return table


def inject_recent2d_columns(races: list, table: dict) -> list:
    out = []
    for r in races:
        df = r["df"].copy()
        per_waku = table.get((r["racecourse"], r["kaisai_date"]), {})
        wnum = df["ca_waku_category_label"].astype(str).str.extract(r"(\d+)")[0]
        wnum = pd.to_numeric(wnum, errors="coerce")
        rates, runs = [], []
        for w in wnum:
            if pd.notna(w) and int(w) in per_waku:
                rate, n = per_waku[int(w)]
            else:
                rate, n = np.nan, 0
            rates.append(rate)
            runs.append(n)
        df["recent2d_waku_win_rate"] = rates
        df["recent2d_waku_runs"] = runs
        out.append({**r, "df": df})
    return out


def fit_equal(names):
    w = np.ones(len(names))

    def fit_fn(_train_idx):
        return w
    return fit_fn


def main():
    data = nar_dataset.load()
    races, actual = data["races"], data["actual"]

    history = load_results_history()
    target_keys = {(r["racecourse"], r["kaisai_date"]) for r in races}
    table = build_recent2d_table(history, target_keys)

    covered_keys = sum(1 for k in target_keys if k in table)
    races2 = inject_recent2d_columns(races, table)
    n_with = sum(1 for r in races2 if (r["df"]["recent2d_waku_runs"] > 0).any())
    print(f"date x course keys with a valid consecutive-2-day window: {covered_keys} / {len(target_keys)}")
    print(f"races with >=1 horse having recent2d data: {n_with} / {len(races2)}")
    non_na = pd.concat([r["df"]["recent2d_waku_runs"] for r in races2])
    print(f"per-waku runs in window (n>0 only): "
          f"mean={non_na[non_na > 0].mean():.2f} median={non_na[non_na > 0].median():.1f} "
          f"(n_horses_with_data={int((non_na > 0).sum())}/{len(non_na)})")

    import json
    base_names = json.loads(WINNER_BOX4.read_text(encoding="utf-8"))["alive_signals"]
    cand_names = base_names + ["waku_recent2d"]
    print(f"\nbase_names ({len(base_names)}): {base_names}")

    priors_all = NS.make_priors(races2)
    dead = NS.detect_dead(races2, priors_all, names=["waku_recent2d"])
    print("dead check (全レース基準): ", dead if dead else "生存(死にシグナルではない)")

    for box_n in (4, 3):
        print(f"\n{'=' * 70}\nbox_n={box_n}\n{'=' * 70}")
        ev = NE.Evaluator(races2, actual, box_n=box_n)

        mats_base = NS.signal_matrices(races2, priors_all, base_names)
        mats_cand = NS.signal_matrices(races2, priors_all, cand_names)

        oof_base = ev.lobo_oof(fit_equal(base_names), mats_base)
        oof_cand = ev.lobo_oof(fit_equal(cand_names), mats_cand)

        ci_base = ev.block_bootstrap(oof_base["picks"])
        ci_cand = ev.block_bootstrap(oof_cand["picks"])

        print(f"基準(alive_signals等重み{len(base_names)}本): "
              f"model={oof_base['model']:.2f} market={oof_base['market']:.2f} "
              f"excess={oof_base['excess']:+.2f}pt  "
              f"[95%CI model {ci_base['lo']:.1f}-{ci_base['hi']:.1f}]")
        print(f"基準+waku_recent2d(等重み{len(cand_names)}本): "
              f"model={oof_cand['model']:.2f} market={oof_cand['market']:.2f} "
              f"excess={oof_cand['excess']:+.2f}pt  "
              f"[95%CI model {ci_cand['lo']:.1f}-{ci_cand['hi']:.1f}]")
        print(f"候補追加による差分: {oof_cand['excess'] - oof_base['excess']:+.2f}pt")


if __name__ == "__main__":
    main()
