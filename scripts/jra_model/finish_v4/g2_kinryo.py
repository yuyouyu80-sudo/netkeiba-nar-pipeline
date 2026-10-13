# -*- coding: utf-8 -*-
"""finish_v4 G2(プラン第4版、2026-10-13): 斤量の扱いを開発の期間(2018〜2019年だけ)で比べる。

背景: FINAL・F7 の斤量(kin=斤量−55、d_kin=前走からの差、斤量×距離)には、見習い騎手の減量(=騎手の情報)と、
ハンデ戦・別定戦の「重い=強い」が混ざる。結果の表・馬柱には、減量の印もハンデ・別定の区別も無い(2026-10-13 に確かめた)ため、
騎手の情報を使わずに減量だけを取り除くことはできない。
比べる版(土台は F7):
  - そのまま(F7)
  - 斤量を外す(kin・d_kin・斤量×距離 を外す): 騎手を使わない決定に完全に沿う
  - 同じレースの同じ性(牝/牡・セ)・年齢の馬の中央値との差(kin_rel)に置き換え、d_kin は外す: 定量・馬齢の基準の違いは消えるが、減量は残る
2020年はここでは見ない(G* の凍結のときに1回だけ確かめる)。
出力: data/jra_pipeline/finish_v4/g2_kinryo.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import g1_decompose as G1  # noqa: E402

OUT = F0.OUT_DIR
YEARS = G1.SEL_YEARS
KIN = ["kin", "d_kin", "斤量×距離"]


def sex_map(max_year: int = 2019) -> pd.Series:
    rows = []
    for y in range(2011, max_year + 1):
        for p in sorted((F0.C.PROJECT_ROOT / "data" / "race_results" / str(y)).glob("*.csv")):
            d = pd.read_csv(p, dtype=str, usecols=["race_id", "horse_id", "sex_age"])
            rows.append(d)
    d = pd.concat(rows, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    d["female"] = d["sex_age"].str[:1].eq("牝").astype(float)
    return d.set_index(["race_id", "horse_id"])["female"]


def add_rel(D: pd.DataFrame, fem: pd.Series) -> pd.DataFrame:
    D = D.copy()
    D["female"] = pd.Series(list(zip(D["race_id"], D["horse_id"]))).map(fem).to_numpy()
    age_y = D["year"] - pd.to_numeric(D["horse_id"].str[:4], errors="coerce")
    grp = [D["race_id"], D["female"].fillna(-1), age_y]
    D["kin_rel"] = (D["kin"] - D.groupby(grp)["kin"].transform("median")).fillna(0.0)
    D["kin_rel×距離"] = D["kin_rel"] * D["logd_c"]
    return D


def main():
    fem = sex_map(max(YEARS))
    tabs = {Y: add_rel(t, fem) for Y, t in G1.load_tables().items() if Y in YEARS}
    blk = pd.concat([t.drop_duplicates("race_id").set_index("race_id")["block"] for t in tabs.values()])
    blk = blk[~blk.index.duplicated()]
    F7 = G1.F7
    specs = {"F7(そのまま)": F7, "斤量を外す": [c for c in F7 if c not in KIN],
             "同じ性・年齢との差": [c for c in F7 if c not in KIN] + ["kin_rel", "kin_rel×距離"]}
    LL, CM = G1.eval_specs(tabs, specs, YEARS)
    out = {"years": YEARS, "specs": {k: [c for c in v if c not in F7] or ("−" + "・".join(c for c in KIN if c not in v)) for k, v in specs.items()},
           "vs_F7": {k: G1.diff(CM, LL, k, "F7(そのまま)", blk) for k in specs if k != "F7(そのまま)"}}
    T = pd.concat([t[t["year"] == Y] for Y, t in tabs.items()])
    out["kin_rel_share_nonzero"] = round(float((T["kin_rel"].abs() > 1e-9).mean()), 4)
    out["kin_rel_values"] = T["kin_rel"].round(1).value_counts().head(10).to_dict()
    (OUT / "g2_kinryo.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    G1.ledger({"stage": "G2 kinryo", "n_specs_tried": len(specs)})
    for k, v in out["vs_F7"].items():
        F0.log(k, round(v["combo"]["mean"], 5), [round(x, 5) for x in v["combo"]["ci_bca"]], v["by_year"], "hit20", v["hit20"], "ll", v["ll"])


if __name__ == "__main__":
    main()
