# -*- coding: utf-8 -*-
"""T3 のコメント掲載レースがどのクラスに偏っているか(2026-10-12 2回目レビュー対応、記述のみ)。

クラスはレース名の文字から決める(新馬・未勝利・1勝・2勝・3勝、それ以外は OP以上)。
入力: t3_rows.pkl(t3_comments.py が書く)、race_results / intraday_results のレース名
出力: data/jra_pipeline/market_pools_2026_10_12/t3_class_mix.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_common as M  # noqa: E402

ORDER = ["新馬", "未勝利", "1勝", "2勝", "3勝", "OP以上"]


def klass(name):
    s = str(name)
    for k in ORDER[:-1]:
        if k in s:
            return k
    return "OP以上"


def race_names(dates):
    out = {}
    for d in dates:
        for p in (M.ROOT / "data" / "race_results" / d[:4] / f"{d}.csv", M.ROOT / "data" / "intraday_results" / f"{d}.csv"):
            if p.exists():
                x = pd.read_csv(p, dtype=str, usecols=["race_id", "race_name"]).drop_duplicates("race_id")
                for r, n in zip(x["race_id"], x["race_name"]):
                    out.setdefault(r, n)
    return out


def main():
    df = pd.read_pickle(M.OUT / "t3_rows.pkl")
    names = race_names(sorted(df["date"].unique()))
    df["klass"] = df["race_id"].map(lambda r: klass(names.get(r, "")))
    has = df.groupby("race_id")["hasc"].max()
    rk = df.drop_duplicates("race_id").set_index("race_id")["klass"]
    tab = pd.DataFrame({"klass": rk, "has": has.reindex(rk.index)})
    rows = []
    for k in ORDER:
        t = tab[tab["klass"] == k]
        c = df[(df["klass"] == k) & (df["hasc"] == 1)]
        r = {"klass": k, "n_races": int(len(t)), "n_comment_races": int(t["has"].sum())}
        if len(c):
            r.update({"n_horses": int(len(c)), "win_by_E": {int(e): [int(len(g)), float(g["win"].mean()), float(g["p"].mean())]
                                                             for e, g in c.groupby("E")}})
        rows.append(r)
    (M.OUT / "t3_class_mix.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r["klass"], r["n_comment_races"], "/", r["n_races"])


if __name__ == "__main__":
    main()
