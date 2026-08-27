"""差し・追込馬限定で、展開予想(AI展開)の予想corner4_gap_lengths(先頭からの馬身差)刻みごとの
勝率・複勝率と、ブートストラップ95%信頼区間、および「予想X馬身超なら消す」ルールを適用した際の
費用対効果(除外率 vs 取りこぼす勝ち馬・複勝馬の割合)を算出し、data/mark_analysis/配下に保存する。

data/mark_analysis/corner4_result_joined.csv (scripts/analyze_corner4_straight.pyが生成)を
入力とする。ネットワークアクセスなし、ローカルCSVの読み込みのみ。

使い方:
    python scripts/analyze_closer_gap_threshold.py
"""
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

EDGES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 999]
LABELS = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12+"]
N_BOOT = 3000
SEED = 42


def build_bootstrap_ci_table(closers: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for i in range(len(EDGES) - 1):
        lo, hi = EDGES[i], EDGES[i + 1]
        mask = (closers["corner4_gap_lengths"] > lo if lo > 0 else closers["corner4_gap_lengths"] >= lo) & (
            closers["corner4_gap_lengths"] <= hi
        )
        sub = closers[mask]
        n = len(sub)
        if n == 0:
            continue
        win_arr = sub["win"].values
        p3_arr = sub["place3"].values
        boot_win = np.array([rng.choice(win_arr, size=n, replace=True).mean() for _ in range(N_BOOT)])
        boot_p3 = np.array([rng.choice(p3_arr, size=n, replace=True).mean() for _ in range(N_BOOT)])
        w_lo, w_hi = np.percentile(boot_win, [2.5, 97.5])
        p_lo, p_hi = np.percentile(boot_p3, [2.5, 97.5])
        rows.append(
            {
                "gap_bin": f"{LABELS[i]}馬身",
                "n": n,
                "win_rate": win_arr.mean(),
                "win_ci_lo": w_lo,
                "win_ci_hi": w_hi,
                "place3_rate": p3_arr.mean(),
                "place3_ci_lo": p_lo,
                "place3_ci_hi": p_hi,
            }
        )
    return pd.DataFrame(rows)


def build_exclusion_cost_table(closers: pd.DataFrame) -> pd.DataFrame:
    total_horses = len(closers)
    total_winners = closers["win"].sum()
    total_placers3 = closers["place3"].sum()
    rows = []
    for th in (4, 5, 6, 7, 8, 10):
        excluded = closers[closers["corner4_gap_lengths"] > th]
        n_ex = len(excluded)
        win_lost = excluded["win"].sum()
        p3_lost = excluded["place3"].sum()
        rows.append(
            {
                "threshold_lengths": th,
                "n_excluded": n_ex,
                "exclusion_rate": n_ex / total_horses,
                "wins_lost": int(win_lost),
                "wins_lost_rate": win_lost / total_winners,
                "place3_lost": int(p3_lost),
                "place3_lost_rate": p3_lost / total_placers3,
            }
        )
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(REPO_ROOT / "data" / "mark_analysis" / "corner4_result_joined.csv")
    closers = df[df["running_style"].isin(["差", "追"])].dropna(subset=["corner4_gap_lengths"]).copy()

    ci_table = build_bootstrap_ci_table(closers)
    ci_table.to_csv(OUT_DIR / "closer_gap_bootstrap_ci.csv", index=False, encoding="utf-8")

    cost_table = build_exclusion_cost_table(closers)
    cost_table.to_csv(OUT_DIR / "closer_gap_exclusion_cost.csv", index=False, encoding="utf-8")

    print(f"closers n = {len(closers)}")
    print(f"saved: {OUT_DIR / 'closer_gap_bootstrap_ci.csv'}")
    print(f"saved: {OUT_DIR / 'closer_gap_exclusion_cost.csv'}")


if __name__ == "__main__":
    main()
