# -*- coding: utf-8 -*-
"""T3 の採点の一致度(2026-10-12 レビュー対応)。

無作為の200件を、結果を見ていない別の評価者(サブエージェント、同じ基準)が盲検で採点し、
本採点(この会話の Claude)と比べる。入力: inputs/rater2/key200.csv(rid→cid)・rater2.csv(rid,E,U)
出力: rater_agreement.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_common as M  # noqa: E402

D = M.OUT / "inputs" / "rater2"


def weighted_kappa(a, b, cats):
    k = len(cats)
    idx = {c: i for i, c in enumerate(cats)}
    O = np.zeros((k, k))
    for x, y in zip(a, b):
        O[idx[x], idx[y]] += 1
    W = np.array([[(i - j) ** 2 for j in range(k)] for i in range(k)]) / (k - 1) ** 2
    Ex = np.outer(O.sum(1), O.sum(0)) / O.sum()
    return float(1 - (W * O).sum() / (W * Ex).sum())


def kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = (a == b).mean()
    cats = np.union1d(a, b)
    pe = sum((a == c).mean() * (b == c).mean() for c in cats)
    return float((po - pe) / (1 - pe))


def main():
    key = pd.read_csv(D / "key200.csv")
    r2 = pd.read_csv(D / "rater2.csv")
    assert len(r2) == 200 and r2["rid"].is_unique and set(r2["rid"]) == set(range(1, 201))
    assert r2["E"].between(-2, 2).all() and r2["U"].isin([0, 1]).all()
    lab = pd.read_csv(M.OUT / "comment_labels.csv")[["cid", "E", "U"]]
    m = key.merge(r2, on="rid").merge(lab, on="cid", suffixes=("_2", "_1"))
    res = {"n": len(m),
           "E_exact": float((m["E_1"] == m["E_2"]).mean()),
           "E_within1": float(((m["E_1"] - m["E_2"]).abs() <= 1).mean()),
           "E_kappa_quadratic": weighted_kappa(m["E_1"], m["E_2"], [-2, -1, 0, 1, 2]),
           "E_spearman": float(spearmanr(m["E_1"], m["E_2"]).correlation),
           "E_mean_1": float(m["E_1"].mean()), "E_mean_2": float(m["E_2"].mean()),
           "U_kappa": kappa(m["U_1"], m["U_2"]), "U_rate_1": float(m["U_1"].mean()), "U_rate_2": float(m["U_2"].mean()),
           "crosstab_E": {str(k): {str(kk): int(vv) for kk, vv in v.items()}
                          for k, v in pd.crosstab(m["E_1"], m["E_2"]).to_dict().items()}}
    (M.OUT / "rater_agreement.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in res.items() if k != "crosstab_E"}, ensure_ascii=False, indent=1))
    print(pd.crosstab(m["E_1"], m["E_2"], rownames=["本採点"], colnames=["別評価者"]))


if __name__ == "__main__":
    main()
