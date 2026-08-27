"""予想印・展開データ分析レポート(session_report.html)内で既に提示したパターンについて、
実際の払戻データ(単勝・複勝)を突き合わせ、回収率を算出する。

新規の特徴量・モデリングは行わず、レポート②③④⑤⑦⑧で既に使っているフィルタ条件
(印の記号・組合せ、4コーナー先頭差、3角→4角の位置変化)のみを対象とする。

使い方:
    python scripts/analyze_mark_payout_sim.py
"""
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"


def load_results():
    files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "**" / "*.csv"), recursive=True)
    frames = [pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "finish_pos"]) for f in files]
    r = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])
    r["finish_pos_num"] = pd.to_numeric(r["finish_pos"], errors="coerce")
    r = r.dropna(subset=["finish_pos_num"])
    r["finish_pos_num"] = r["finish_pos_num"].astype(int)
    return r[["race_id", "umaban", "finish_pos_num"]]


def load_marks():
    files = glob.glob(str(REPO_ROOT / "data" / "newspaper" / "*.csv")) + glob.glob(
        str(REPO_ROOT / "data" / "newspaper" / "nar" / "*.csv")
    )
    frames = []
    for f in files:
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


def load_payouts():
    files = glob.glob(str(REPO_ROOT / "data" / "payouts" / "**" / "*.csv"), recursive=True)
    p = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    win = p[p["bet_type"] == "単勝"][["race_id", "combination", "payout"]].copy()
    win["payout"] = pd.to_numeric(win["payout"], errors="coerce")
    win = win.rename(columns={"combination": "umaban"})
    win_lookup = win.set_index(["race_id", "umaban"])["payout"]

    place = p[p["bet_type"] == "複勝"][["race_id", "combination", "payout"]].copy()
    place["payout"] = pd.to_numeric(place["payout"], errors="coerce")
    place = place.rename(columns={"combination": "umaban"})
    place_lookup = place.set_index(["race_id", "umaban"])["payout"]
    return win_lookup, place_lookup


def load_corner34():
    return pd.read_csv(OUT_DIR / "corner34_transition_joined.csv", dtype=str)


def load_corner4():
    return pd.read_csv(OUT_DIR / "corner4_result_joined.csv", dtype=str)


def stats(sub, win_lookup, place_lookup, label):
    n = len(sub)
    if n == 0:
        return None
    win = sub["win"].mean()
    p2 = sub["place2"].mean()
    p3 = sub["place3"].mean()
    idx = list(zip(sub["race_id"], sub["umaban"]))
    win_pay = sum(win_lookup.get(k, 0) or 0 for k in idx)
    place_pay = sum(place_lookup.get(k, 0) or 0 for k in idx)
    return {
        "pattern": label, "n": n, "win_rate": round(win, 4), "place2_rate": round(p2, 4),
        "place3_rate": round(p3, 4),
        "win_return": round(win_pay / (n * 100), 4), "place_return": round(place_pay / (n * 100), 4),
    }


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    marks = load_marks().merge(results, on=["race_id", "umaban"], how="inner")
    marks["win"] = (marks["finish_pos_num"] == 1).astype(int)
    marks["place2"] = (marks["finish_pos_num"] <= 2).astype(int)
    marks["place3"] = (marks["finish_pos_num"] <= 3).astype(int)

    win_lookup, place_lookup = load_payouts()

    c34 = load_corner34()
    for c in ("win", "place2", "place3", "rank_delta", "gap_delta"):
        c34[c] = pd.to_numeric(c34[c], errors="coerce")

    c4 = load_corner4()
    for c in ("win", "place2", "place3", "corner4_gap_lengths"):
        c4[c] = pd.to_numeric(c4[c], errors="coerce")

    rows = []
    rows.append(stats(marks[(marks["mark_cp"] == "◎") & (marks["mark_other"] == "◎")], win_lookup, place_lookup,
                       "CP予想◎ x その他◎ (レポート③トップ)"))
    rows.append(stats(marks[marks["mark_other"] == "◎"], win_lookup, place_lookup, "その他◎単独(レポート②)"))
    rows.append(stats(marks[(marks["mark_honshi"] == "◎") & (marks["mark_other"] == "◎")], win_lookup, place_lookup,
                       "本紙◎ x その他◎ (レポート③)"))
    rows.append(stats(marks[(marks["mark_honshi"] == "○") & (marks["mark_other"] == "◎")], win_lookup, place_lookup,
                       "本紙○ x その他◎ (レポート③)"))
    rows.append(stats(marks[(marks["mark_honshi"] == "○") & (marks["mark_cp"] == "◎")], win_lookup, place_lookup,
                       "本紙○ x CP予想◎ (レポート③)"))
    rows.append(stats(c34[c34["gap_delta_bin"] == "2馬身以上縮小"], win_lookup, place_lookup,
                       "3角→4角 先頭差2馬身以上縮小 (レポート⑧)"))
    rows.append(stats(c34[c34["rank_delta_bin"] == "3位以上アップ"], win_lookup, place_lookup,
                       "3角→4角 3位以上順位アップ (レポート⑧)"))
    closers = c4[c4["running_style"].isin(["差", "追"])]
    rows.append(stats(closers[closers["corner4_gap_lengths"] <= 1], win_lookup, place_lookup,
                       "差し追込 4角先頭差0-1馬身 (レポート⑤)"))

    marks_key = marks[marks["mark_other"] == "◎"][["race_id", "umaban"]]
    combo1 = c34.merge(marks_key, on=["race_id", "umaban"], how="inner")
    rows.append(stats(combo1[combo1["gap_delta_bin"] == "2馬身以上縮小"], win_lookup, place_lookup,
                       "その他◎ x 3角→4角先頭差2馬身以上縮小(組合せ)"))
    rows.append(stats(combo1[combo1["rank_delta_bin"].isin(["2位アップ", "3位以上アップ"])], win_lookup, place_lookup,
                       "その他◎ x 3角→4角2位以上順位アップ(組合せ)"))

    marks_key2 = marks[(marks["mark_cp"] == "◎") & (marks["mark_other"] == "◎")][["race_id", "umaban"]]
    combo2 = c34.merge(marks_key2, on=["race_id", "umaban"], how="inner")
    rows.append(stats(combo2[combo2["gap_delta_bin"].isin(["0.5〜2馬身縮小", "2馬身以上縮小"])], win_lookup, place_lookup,
                       "CP予想◎xその他◎ x 3角→4角先頭差縮小(組合せ)"))

    rows.append(stats(marks, win_lookup, place_lookup, "(参考)全馬ベースライン"))

    df = pd.DataFrame([r for r in rows if r is not None]).sort_values("win_return", ascending=False)
    df.to_csv(OUT_DIR / "mark_payout_sim.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False))
    print(f"saved: {OUT_DIR / 'mark_payout_sim.csv'}")


if __name__ == "__main__":
    main()
