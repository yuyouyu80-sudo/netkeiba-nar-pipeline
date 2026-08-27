"""予想印(本紙・CP予想・その他)の記号別・組合せ別の勝率・連対率・複勝率を集計し、
data/mark_analysis/ 配下のCSVに保存する。

data/newspaper/*.csv, data/newspaper/nar/*.csv の mark_honshi/mark_cp/mark_other と、
data/race_results/**/*.csv の finish_pos を race_id+umaban で突き合わせて算出する
(出走取消・除外・中止等finish_posが数値でない行は分母から除外)。

このスクリプトはネットワークアクセスなし、ローカルCSVの読み込みのみ(データ収集セッションで
実行可能)。実際の予想モデルへの組み込み・重み付け判断は「予想用」セッションの領分。

使い方:
    python scripts/analyze_mark_hitrate.py
"""
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

SYMS = ["◎", "○", "▲", "△", "☆"]


def load_results() -> pd.Series:
    result_files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "**" / "*.csv"), recursive=True)
    frames = [pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "finish_pos"]) for f in result_files]
    results = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])
    results["finish_pos_num"] = pd.to_numeric(results["finish_pos"], errors="coerce")
    results = results.dropna(subset=["finish_pos_num"])
    results["finish_pos_num"] = results["finish_pos_num"].astype(int)
    return results.set_index(["race_id", "umaban"])["finish_pos_num"]


def load_marks() -> pd.DataFrame:
    mark_files = glob.glob(str(REPO_ROOT / "data" / "newspaper" / "*.csv")) + glob.glob(
        str(REPO_ROOT / "data" / "newspaper" / "nar" / "*.csv")
    )
    frames = []
    for f in mark_files:
        cols = pd.read_csv(f, dtype=str, nrows=0).columns
        needed = ["race_id", "umaban"]
        for c in ("mark_honshi", "mark_cp", "mark_other"):
            if c in cols:
                needed.append(c)
        if len(needed) <= 2:
            continue
        df = pd.read_csv(f, dtype=str, usecols=needed, keep_default_na=False)
        if "race_id" not in df.columns:
            df["race_id"] = Path(f).stem
        for c in ("mark_honshi", "mark_cp", "mark_other"):
            if c not in df.columns:
                df[c] = ""
        frames.append(df[["race_id", "umaban", "mark_honshi", "mark_cp", "mark_other"]])
    return pd.concat(frames, ignore_index=True)


def rates(sub: pd.DataFrame):
    n = len(sub)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    win = (sub["finish_pos"] == 1).mean()
    p2 = (sub["finish_pos"] <= 2).mean()
    p3 = (sub["finish_pos"] <= 3).mean()
    return n, win, p2, p3


def build_single_symbol_table(marks: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col_label, col in (("本紙", "mark_honshi"), ("CP予想", "mark_cp"), ("その他", "mark_other")):
        for sym in SYMS:
            sub = marks[marks[col] == sym]
            n, win, p2, p3 = rates(sub)
            if n == 0:
                continue
            rows.append({"column": col_label, "mark": sym, "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
        blank = marks[(marks[col] == "") | (marks[col].isna())]
        n, win, p2, p3 = rates(blank)
        rows.append({"column": col_label, "mark": "(無印)", "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    star = marks[marks["mark_other"] == "★"]
    n, win, p2, p3 = rates(star)
    rows.append({"column": "その他", "mark": "★", "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    n, win, p2, p3 = rates(marks)
    rows.append({"column": "(全体平均)", "mark": "-", "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    return pd.DataFrame(rows)


def build_combo_table(marks: pd.DataFrame, col_a: str, label_a: str, col_b: str, label_b: str) -> pd.DataFrame:
    both = marks[(marks[col_a] != "") & (marks[col_b] != "")]
    rows = []
    for sa in SYMS:
        for sb in SYMS:
            sub = both[(both[col_a] == sa) & (both[col_b] == sb)]
            n, win, p2, p3 = rates(sub)
            if n == 0:
                continue
            rows.append(
                {
                    "pair": f"{label_a}x{label_b}",
                    "mark_a": sa,
                    "mark_b": sb,
                    "n": n,
                    "win_rate": win,
                    "place2_rate": p2,
                    "place3_rate": p3,
                }
            )
    df = pd.DataFrame(rows).sort_values("win_rate", ascending=False).reset_index(drop=True)
    return df


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    results_lookup = load_results()
    marks = load_marks()
    marks = marks.set_index(["race_id", "umaban"])
    marks["finish_pos"] = results_lookup.reindex(marks.index)
    marks = marks.dropna(subset=["finish_pos"])
    marks["finish_pos"] = marks["finish_pos"].astype(int)
    marks = marks.reset_index()

    single = build_single_symbol_table(marks)
    single.to_csv(OUT_DIR / "mark_hitrate_by_symbol.csv", index=False, encoding="utf-8")

    combos = pd.concat(
        [
            build_combo_table(marks, "mark_honshi", "本紙", "mark_cp", "CP予想"),
            build_combo_table(marks, "mark_honshi", "本紙", "mark_other", "その他"),
            build_combo_table(marks, "mark_cp", "CP予想", "mark_other", "その他"),
        ],
        ignore_index=True,
    )
    combos.to_csv(OUT_DIR / "mark_combo_hitrate.csv", index=False, encoding="utf-8")

    print(f"対象頭数(結果判明済み): {len(marks)}")
    print(f"saved: {OUT_DIR / 'mark_hitrate_by_symbol.csv'}")
    print(f"saved: {OUT_DIR / 'mark_combo_hitrate.csv'}")


if __name__ == "__main__":
    main()
