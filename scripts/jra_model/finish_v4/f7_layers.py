# -*- coding: utf-8 -*-
"""finish_v4: 確認(2021〜2025年)の結果を、レースの種類(新馬戦・未勝利戦・1〜3勝クラス・OP/L・重賞)に分けて記述する(2026-10-12)。

確認の後の記述で、合否は変えない。入力は確認の1レースずつの記録(final_races.parquet: 人気どおり・M1・M5・M6・FINAL、
f7_final_races.parquet: FINAL・F7)と、レースの種類(final_tables.parquet の class_ord・is_debut_race)。
区間は日付×競馬場のブロックのブートストラップ(的中の差は block_boot、回収率・上積みは ratio_boot)。
出力: data/jra_pipeline/finish_v4/f7_layers.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402

P, OUT = F0.P, F0.OUT_DIR
LAYERS = ["新馬戦", "未勝利戦", "1〜3勝クラス", "OP・L", "重賞"]


def race_type(class_ord: float, debut: bool) -> str:
    if class_ord == 0:
        return "新馬戦" if debut else "未勝利戦"
    if class_ord <= 3:
        return "1〜3勝クラス"
    return "OP・L" if class_ord == 4 else "重賞"


def main():
    T = pd.read_parquet(OUT / "final_tables.parquet", columns=["eval_year", "year", "race_id", "class_ord", "is_debut_race"])
    T = T[T["year"] == T["eval_year"]].drop_duplicates("race_id").set_index("race_id")
    A = pd.read_parquet(OUT / "final_races.parquet").set_index("race_id")
    B = pd.read_parquet(OUT / "f7_final_races.parquet").set_index("race_id")
    R = A.join(B[[c for c in B.columns if c.startswith("F7_")]], how="inner")
    R["type"] = [race_type(c, d) for c, d in zip(T.loc[R.index, "class_ord"], T.loc[R.index, "is_debut_race"].astype(bool))]
    out = {"n_races": int(len(R)), "layers": {}}
    for lab in LAYERS + ["全体"]:
        S = R if lab == "全体" else R[R["type"] == lab]
        b = S["block"].to_numpy()
        row = {"n": int(len(S)), "field_mean": round(float(S["field"].mean()), 1)}
        for rule in ["N10", "N20"]:
            for who in ["mkt", "M1", "FINAL", "F7"]:
                a, e = S[f"{who}_act_{rule}"].to_numpy(float), S[f"{who}_exp_{rule}"].to_numpy(float)
                st = 100.0 * S[f"{who}_k_{rule}"].to_numpy(float)
                u = P.ratio_boot(a - e, st, b, n=1000)
                row[f"{who}_{rule}"] = {"hit": round(float(S[f"{who}_hit_{rule}"].mean()), 4), "roi": round(float(a.sum() / st.sum()), 4),
                                        "uplift": round(float(u["ratio"]), 4), "uplift_ci": [round(float(x), 4) for x in u["ci_bca"]]}
            d = (S[f"F7_hit_{rule}"] - S[f"FINAL_hit_{rule}"]).to_numpy(float)
            bb = P.block_boot(d, b, n=1000)
            row[f"hit_F7_minus_FINAL_{rule}"] = {"mean": round(float(bb["mean"]), 4), "ci": [round(float(x), 4) for x in bb["ci_bca"]]}
            d = (S[f"F7_hit_{rule}"] - S[f"M1_hit_{rule}"]).to_numpy(float)
            bb = P.block_boot(d, b, n=1000)
            row[f"hit_F7_minus_M1_{rule}"] = {"mean": round(float(bb["mean"]), 4), "ci": [round(float(x), 4) for x in bb["ci_bca"]]}
        out["layers"][lab] = row
        F0.log(lab, row["n"], "20点 人気", row["mkt_N20"]["hit"], "M1", row["M1_N20"]["hit"], "FINAL", row["FINAL_N20"]["hit"], "F7", row["F7_N20"]["hit"],
               "F7回収", row["F7_N20"]["roi"], "上積み", row["F7_N20"]["uplift"], row["F7_N20"]["uplift_ci"])
    (OUT / "f7_layers.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
