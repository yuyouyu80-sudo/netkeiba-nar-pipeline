"""予想印(本紙・CP予想・その他)を合算した複合スコアで各レースの馬をランク付けし、
上位N頭(BOX3/4/5)で馬連・馬単・3連複・3連単を総流しした場合の回収率をシミュレーションする。

新規のモデリングは行わず、レポート②③で既に使っている印のスコアリング(◎6/○4/▲3/△2/☆0.5)を
そのまま複合スコアとして本紙・CP予想・その他の3列で合算するのみ(同点はumaban昇順でタイブレーク)。

使い方:
    python scripts/analyze_mark_box_sim.py
"""
import glob
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = REPO_ROOT / "data" / "mark_analysis"

POINTS = {"◎": 6, "○": 4, "▲": 3, "△": 2, "☆": 0.5}


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


def load_results():
    files = glob.glob(str(REPO_ROOT / "data" / "race_results" / "**" / "*.csv"), recursive=True)
    frames = [pd.read_csv(f, dtype=str, usecols=["race_id", "umaban", "finish_pos"]) for f in files]
    r = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "umaban"])
    r["finish_pos_num"] = pd.to_numeric(r["finish_pos"], errors="coerce")
    r = r.dropna(subset=["finish_pos_num"])
    r["finish_pos_num"] = r["finish_pos_num"].astype(int)
    return r


def load_payout_lookup():
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


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    marks = load_marks()
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

    payout_lookups = load_payout_lookup()

    marks_sorted = marks.sort_values(["race_id", "score", "umaban"], ascending=[True, False, True])
    box_by_race = {race_id: g["umaban"].tolist() for race_id, g in marks_sorted.groupby("race_id")}

    box_sizes = [3, 4, 5]
    bet_types = [("馬連", "quinella"), ("馬単", "exacta"), ("3連複", "trio"), ("3連単", "trifecta")]

    rows = []
    for n in box_sizes:
        for bt_label, kind in bet_types:
            total_stake = 0
            total_return = 0
            n_races = 0
            n_hit = 0
            lut = payout_lookups[bt_label]
            for race_id, box in box_by_race.items():
                if len(box) < n or race_id not in finish_map:
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
                    key_sets = [tuple(sorted([fin[1], fin[2], fin[3]], key=int))] if all(
                        fin[k] in selected for k in (1, 2, 3)) else []
                else:  # trifecta
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
                "box_size": n, "bet_type": bt_label, "n_races": n_races, "n_hit": n_hit,
                "hit_rate": round(n_hit / n_races, 4) if n_races else float("nan"),
                "total_stake_yen": total_stake, "total_return_yen": total_return,
                "return_rate": round(total_return / total_stake, 4) if total_stake else float("nan"),
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "mark_box_sim.csv", index=False, encoding="utf-8")
    print(df.to_string(index=False))
    print(f"saved: {OUT_DIR / 'mark_box_sim.csv'}")


if __name__ == "__main__":
    main()
