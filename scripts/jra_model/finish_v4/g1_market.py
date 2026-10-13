# -*- coding: utf-8 -*-
"""finish_v4 G1 の記述(2026-10-13): 開発の期間(2018〜2020年)で、レースの種類別の20点の的中と市場の物差し(評価だけ)を出す。

予想家の指摘「生産者は新馬戦の的中を上げるが、回収率を下げる(人気の牧場に寄る)」を確かめる。比べる版:
FINAL、F7、F7 から初出走×生産者(新×breeder_m)を外した版。確認期間は使わない。合否は付けない。
出力: data/jra_pipeline/finish_v4/g1_market.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import age_breeder as AGB  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import f7_layers as FL  # noqa: E402
import finish_v4_eval as EV  # noqa: E402
import g1_decompose as G1  # noqa: E402

OUT = F0.OUT_DIR
COMPARE = {"FINAL": AGB.FINAL, "F7": G1.F7, "F7−新×生産者": [c for c in G1.F7 if c != "新×breeder_m"]}


def main():
    LL, CM, R, _ = EV.collect("g1_dev_market", F0.YEARS, 2020, G1.TABLES, COMPARE, inc_model="F7", prep=G1.add_g1, l2_of=lambda k: 0.0)
    T = pd.read_parquet(G1.TABLES, columns=["eval_year", "year", "race_id", "class_ord", "is_debut_race"])
    T = T[T["year"] == T["eval_year"]].drop_duplicates("race_id").set_index("race_id")
    R = R.set_index("race_id")
    R["type"] = [FL.race_type(c, bool(d)) for c, d in zip(T.loc[R.index, "class_ord"], T.loc[R.index, "is_debut_race"])]
    out = {"years": F0.YEARS, "n_races": int(len(R)), "layers": {}}
    for lab in FL.LAYERS + ["全体"]:
        S = R if lab == "全体" else R[R["type"] == lab]
        b = S["block"].to_numpy()
        row = {"n": int(len(S))}
        for who in ["mkt"] + list(COMPARE):
            a, e = S[f"{who}_act_N20"].to_numpy(float), S[f"{who}_exp_N20"].to_numpy(float)
            st = 100.0 * S[f"{who}_k_N20"].to_numpy(float)
            u = F0.P.ratio_boot(a - e, st, b, n=1000)
            row[who] = {"hit": round(float(S[f"{who}_hit_N20"].mean()), 4), "roi": round(float(a.sum() / st.sum()), 4),
                        "uplift": round(float(u["ratio"]), 4), "uplift_ci": [round(float(x), 4) for x in u["ci_bca"]],
                        "median_market_rank": float(S[f"{who}_mrank_N20"].median()) if who != "mkt" else None}
        out["layers"][lab] = row
        F0.log(lab, row["n"], {k: (v["hit"], v["roi"], v["uplift"]) for k, v in row.items() if isinstance(v, dict)})
    (OUT / "g1_market.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
