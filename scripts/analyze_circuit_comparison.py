"""予想印・展開データ分析レポート(session_report.html)の②③⑧⑩を、JRA・NAR(地方競馬)で
分けて集計する。④〜⑦は直線距離・上り坂データがJRAの一部コースにしか無いためNAR版は作れない
(現状の制約、data/mark_analysis/course_straight_and_uphill.csv参照)。

data/newspaper/*.csv(JRA) と data/newspaper/nar/*.csv(NAR) の由来ディレクトリからrace_idごとに
競馬種別(circuit)を判定し、既存の分析ロジック(analyze_mark_hitrate.py / analyze_corner34_transition.py /
analyze_mark_payout_sim.py / analyze_mark_box_sim.py)と同じ集計をJRA/NARそれぞれに対して行う。

使い方:
    python scripts/analyze_circuit_comparison.py
"""
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

SYMS = ["◎", "○", "▲", "△", "☆"]
POINTS = {"◎": 6, "○": 4, "▲": 3, "△": 2, "☆": 0.5}


def build_race_circuit_map() -> dict:
    m = {}
    for f in glob.glob(str(REPO_ROOT / "data" / "newspaper" / "*.csv")):
        m[Path(f).stem] = "JRA"
    for f in glob.glob(str(REPO_ROOT / "data" / "newspaper" / "nar" / "*.csv")):
        m[Path(f).stem] = "NAR"
    return m


def load_marks(circuit_map: dict) -> pd.DataFrame:
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
    out = pd.concat(frames, ignore_index=True)
    out["circuit"] = out["race_id"].map(circuit_map)
    return out


def load_results():
    files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "**" / "*.csv"), recursive=True)
    frames = [pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "finish_pos"]) for f in files]
    r = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])
    r["finish_pos_num"] = pd.to_numeric(r["finish_pos"], errors="coerce")
    r = r.dropna(subset=["finish_pos_num"])
    r["finish_pos_num"] = r["finish_pos_num"].astype(int)
    return r[["race_id", "umaban", "finish_pos_num"]]


def load_payout_lookup_win_place():
    files = glob.glob(str(REPO_ROOT / "data" / "payouts" / "**" / "*.csv"), recursive=True)
    p = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    win = p[p["bet_type"] == "単勝"][["race_id", "combination", "payout"]].copy()
    win["payout"] = pd.to_numeric(win["payout"], errors="coerce")
    win_lookup = win.set_index(["race_id", win["combination"]])["payout"]
    place = p[p["bet_type"] == "複勝"][["race_id", "combination", "payout"]].copy()
    place["payout"] = pd.to_numeric(place["payout"], errors="coerce")
    place_lookup = place.set_index(["race_id", place["combination"]])["payout"]
    return win_lookup, place_lookup


def rates(sub):
    n = len(sub)
    if n == 0:
        return n, float("nan"), float("nan"), float("nan")
    win = (sub["finish_pos_num"] == 1).mean()
    p2 = (sub["finish_pos_num"] <= 2).mean()
    p3 = (sub["finish_pos_num"] <= 3).mean()
    return n, win, p2, p3


def build_single_symbol_table(marks: pd.DataFrame, circuit: str) -> pd.DataFrame:
    sub_all = marks[marks["circuit"] == circuit]
    rows = []
    for col_label, col in (("本紙", "mark_honshi"), ("CP予想", "mark_cp"), ("その他", "mark_other")):
        for sym in SYMS:
            sub = sub_all[sub_all[col] == sym]
            n, win, p2, p3 = rates(sub)
            if n == 0:
                continue
            rows.append({"circuit": circuit, "column": col_label, "mark": sym, "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
        blank = sub_all[(sub_all[col] == "") | (sub_all[col].isna())]
        n, win, p2, p3 = rates(blank)
        rows.append({"circuit": circuit, "column": col_label, "mark": "(無印)", "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    star = sub_all[sub_all["mark_other"] == "★"]
    n, win, p2, p3 = rates(star)
    rows.append({"circuit": circuit, "column": "その他", "mark": "★", "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    n, win, p2, p3 = rates(sub_all)
    rows.append({"circuit": circuit, "column": "(全体平均)", "mark": "-", "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    return pd.DataFrame(rows)


def build_combo_table(marks: pd.DataFrame, circuit: str, col_a, label_a, col_b, label_b) -> pd.DataFrame:
    sub_all = marks[marks["circuit"] == circuit]
    both = sub_all[(sub_all[col_a] != "") & (sub_all[col_b] != "")]
    rows = []
    for sa in SYMS:
        for sb in SYMS:
            sub = both[(both[col_a] == sa) & (both[col_b] == sb)]
            n, win, p2, p3 = rates(sub)
            if n == 0:
                continue
            rows.append({"circuit": circuit, "pair": f"{label_a}x{label_b}", "mark_a": sa, "mark_b": sb, "n": n, "win_rate": win, "place2_rate": p2, "place3_rate": p3})
    if not rows:
        return pd.DataFrame(columns=["circuit", "pair", "mark_a", "mark_b", "n", "win_rate", "place2_rate", "place3_rate"])
    return pd.DataFrame(rows).sort_values("win_rate", ascending=False).reset_index(drop=True)


def corner34_split():
    c34 = pd.read_csv(OUT_DIR / "corner34_transition_joined.csv", dtype=str)
    for c in ("win", "place2", "place3", "rank_delta", "gap_delta"):
        c34[c] = pd.to_numeric(c34[c], errors="coerce")
    circuit_map = build_race_circuit_map()
    c34["circuit"] = c34["race_id"].map(circuit_map)

    order_rank = ["3位以上ダウン", "2位ダウン", "1位ダウン", "変化なし", "1位アップ", "2位アップ", "3位以上アップ"]
    order_gap = ["2馬身以上拡大", "0.5〜2馬身拡大", "ほぼ変化なし", "0.5〜2馬身縮小", "2馬身以上縮小"]

    out_rank = []
    out_gap = []
    for circuit in ("JRA", "NAR"):
        sub = c34[c34["circuit"] == circuit]
        g = sub.groupby("rank_delta_bin", observed=True).agg(n=("win", "size"), win_rate=("win", "mean"), place2_rate=("place2", "mean"), place3_rate=("place3", "mean")).reindex(order_rank).dropna(subset=["n"])
        g["n"] = g["n"].astype(int)
        g = g.reset_index()
        g.insert(0, "circuit", circuit)
        out_rank.append(g)

        gg = sub.dropna(subset=["gap_delta_bin"]).groupby("gap_delta_bin", observed=True).agg(n=("win", "size"), win_rate=("win", "mean"), place2_rate=("place2", "mean"), place3_rate=("place3", "mean")).reindex(order_gap).dropna(subset=["n"])
        gg["n"] = gg["n"].astype(int)
        gg = gg.reset_index()
        gg.insert(0, "circuit", circuit)
        out_gap.append(gg)

    return pd.concat(out_rank, ignore_index=True), pd.concat(out_gap, ignore_index=True)


def load_payout_lookup_box():
    files = glob.glob(str(REPO_ROOT / "data" / "payouts" / "**" / "*.csv"), recursive=True)
    p = pd.concat([pd.read_csv(f, dtype=str) for f in files], ignore_index=True)
    p["payout"] = pd.to_numeric(p["payout"], errors="coerce")
    lookups = {}
    for bt, sep in [("馬連", " - "), ("馬単", " → "), ("3連複", " - "), ("3連単", " → ")]:
        sub = p[p["bet_type"] == bt].dropna(subset=["payout"])
        d = {}
        for _, row in sub.iterrows():
            parts = tuple(x.strip() for x in row["combination"].split(sep))
            d[(row["race_id"], parts)] = row["payout"]
        lookups[bt] = d
    return lookups


def box_sim_split(marks, circuit_map):
    for c in ("mark_honshi", "mark_cp", "mark_other"):
        marks[c + "_pt"] = marks[c].map(POINTS).fillna(0)
    marks["score"] = marks["mark_honshi_pt"] + marks["mark_cp_pt"] + marks["mark_other_pt"]

    results = load_results()
    finishers = results[results["finish_pos_num"] <= 3].copy()
    finish_map = {}
    for race_id, g in finishers.groupby("race_id"):
        d = dict(zip(g["finish_pos_num"], g["umaban"]))
        if 1 in d and 2 in d and 3 in d:
            finish_map[race_id] = d

    payout_lookups = load_payout_lookup_box()
    marks_sorted = marks.sort_values(["race_id", "score", "umaban"], ascending=[True, False, True])
    box_by_race = {race_id: g["umaban"].tolist() for race_id, g in marks_sorted.groupby("race_id")}

    box_sizes = [3, 4, 5]
    bet_types = [("馬連", "quinella"), ("馬単", "exacta"), ("3連複", "trio"), ("3連単", "trifecta")]

    rows = []
    for circuit in ("JRA", "NAR"):
        race_ids = [rid for rid, c in circuit_map.items() if c == circuit]
        for n in box_sizes:
            for bt_label, kind in bet_types:
                total_stake = 0
                total_return = 0
                n_races = 0
                n_hit = 0
                lut = payout_lookups[bt_label]
                for race_id in race_ids:
                    box = box_by_race.get(race_id)
                    if box is None or len(box) < n or race_id not in finish_map:
                        continue
                    selected = set(box[:n])
                    fin = finish_map[race_id]

                    if kind == "quinella":
                        stake_combos = n * (n - 1) // 2
                        key_sets = [tuple(sorted([fin[1], fin[2]], key=int))] if fin[1] in selected and fin[2] in selected else []
                    elif kind == "exacta":
                        stake_combos = n * (n - 1)
                        key_sets = [(fin[1], fin[2])] if fin[1] in selected and fin[2] in selected else []
                    elif kind == "trio":
                        stake_combos = n * (n - 1) * (n - 2) // 6
                        key_sets = [tuple(sorted([fin[1], fin[2], fin[3]], key=int))] if all(fin[k] in selected for k in (1, 2, 3)) else []
                    else:
                        stake_combos = n * (n - 1) * (n - 2)
                        key_sets = [(fin[1], fin[2], fin[3])] if all(fin[k] in selected for k in (1, 2, 3)) else []

                    n_races += 1
                    total_stake += stake_combos * 100
                    for k in key_sets:
                        pay = lut.get((race_id, k))
                        if pay is not None:
                            total_return += pay
                            n_hit += 1

                rows.append({
                    "circuit": circuit, "box_size": n, "bet_type": bt_label, "n_races": n_races, "n_hit": n_hit,
                    "hit_rate": round(n_hit / n_races, 4) if n_races else float("nan"),
                    "return_rate": round(total_return / total_stake, 4) if total_stake else float("nan"),
                })
    return pd.DataFrame(rows)


def payout_sim_split(marks, win_lookup, place_lookup):
    marks_r = marks.merge(load_results(), on=["race_id", "umaban"], how="inner")
    marks_r["win"] = (marks_r["finish_pos_num"] == 1).astype(int)
    marks_r["place2"] = (marks_r["finish_pos_num"] <= 2).astype(int)
    marks_r["place3"] = (marks_r["finish_pos_num"] <= 3).astype(int)

    def stat(sub, label, circuit):
        n = len(sub)
        if n == 0:
            return None
        idx = list(zip(sub["race_id"], sub["umaban"]))
        win_pay = sum(win_lookup.get(k, 0) or 0 for k in idx)
        place_pay = sum(place_lookup.get(k, 0) or 0 for k in idx)
        return {
            "circuit": circuit, "pattern": label, "n": n,
            "win_rate": round(sub["win"].mean(), 4), "place2_rate": round(sub["place2"].mean(), 4), "place3_rate": round(sub["place3"].mean(), 4),
            "win_return": round(win_pay / (n * 100), 4), "place_return": round(place_pay / (n * 100), 4),
        }

    rows = []
    for circuit in ("JRA", "NAR"):
        sub_all = marks_r[marks_r["circuit"] == circuit]
        rows.append(stat(sub_all[(sub_all["mark_cp"] == "◎") & (sub_all["mark_other"] == "◎")], "CP予想◎ x その他◎", circuit))
        rows.append(stat(sub_all[sub_all["mark_other"] == "◎"], "その他◎単独", circuit))
        rows.append(stat(sub_all, "(参考)全馬ベースライン", circuit))
    return pd.DataFrame([r for r in rows if r is not None])


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    circuit_map = build_race_circuit_map()
    marks = load_marks(circuit_map)
    results = load_results()
    marks_with_result = marks.merge(results, on=["race_id", "umaban"], how="inner")

    single_rows = pd.concat([build_single_symbol_table(marks_with_result, c) for c in ("JRA", "NAR")], ignore_index=True)
    single_rows.to_csv(OUT_DIR / "mark_hitrate_by_symbol_by_circuit.csv", index=False, encoding="utf-8")

    combo_rows = []
    for circuit in ("JRA", "NAR"):
        combo_rows.append(build_combo_table(marks_with_result, circuit, "mark_honshi", "本紙", "mark_cp", "CP予想"))
        combo_rows.append(build_combo_table(marks_with_result, circuit, "mark_honshi", "本紙", "mark_other", "その他"))
        combo_rows.append(build_combo_table(marks_with_result, circuit, "mark_cp", "CP予想", "mark_other", "その他"))
    combos = pd.concat(combo_rows, ignore_index=True)
    combos.to_csv(OUT_DIR / "mark_combo_hitrate_by_circuit.csv", index=False, encoding="utf-8")

    rank_split, gap_split = corner34_split()
    rank_split.to_csv(OUT_DIR / "corner34_rank_delta_hitrate_by_circuit.csv", index=False, encoding="utf-8")
    gap_split.to_csv(OUT_DIR / "corner34_gap_delta_hitrate_by_circuit.csv", index=False, encoding="utf-8")

    win_lookup, place_lookup = load_payout_lookup_win_place()
    pay_split = payout_sim_split(marks, win_lookup, place_lookup)
    pay_split.to_csv(OUT_DIR / "mark_payout_sim_by_circuit.csv", index=False, encoding="utf-8")

    box_split = box_sim_split(marks.copy(), circuit_map)
    box_split.to_csv(OUT_DIR / "mark_box_sim_by_circuit.csv", index=False, encoding="utf-8")

    print("=== 印別成績(JRA/NAR) n件数のみ抜粋 ===")
    print(single_rows[["circuit", "column", "mark", "n"]].to_string(index=False))
    print("\n=== payout sim by circuit ===")
    print(pay_split.to_string(index=False))
    print("\n=== box sim by circuit ===")
    print(box_split.to_string(index=False))
    print(f"\nsaved: {OUT_DIR / 'mark_hitrate_by_symbol_by_circuit.csv'}")
    print(f"saved: {OUT_DIR / 'mark_combo_hitrate_by_circuit.csv'}")
    print(f"saved: {OUT_DIR / 'corner34_rank_delta_hitrate_by_circuit.csv'}")
    print(f"saved: {OUT_DIR / 'corner34_gap_delta_hitrate_by_circuit.csv'}")
    print(f"saved: {OUT_DIR / 'mark_payout_sim_by_circuit.csv'}")
    print(f"saved: {OUT_DIR / 'mark_box_sim_by_circuit.csv'}")


if __name__ == "__main__":
    main()
