# -*- coding: utf-8 -*-
"""finish_v4 F0-3 の診断: 確定単勝オッズから作る市場の組確率(市場PL)と、発走前の3連複オッズの対応(2026-10-11)。

評価の物差し(market_yardstick)は「単勝オッズ → PL」の近似で3連複の市場を表している。その近似のずれを、実際の発走前の
3連複オッズ(data/odds_history_sanrenpuku、2026-10-03〜、各レースの発走にいちばん近い時点)で確かめる。
モデルの予想・着順は使わない(この診断はオッズ同士の比較だけ。2026年の着順も見ない)。γ・λ は 2014〜2019年で推定した値
(dev_f0_market.json の 2020年の行)を使う。
- 組ごと: 3連複オッズから作る確率 q = (1/odds)/Σ(1/odds)、市場PLの確率 p。
- 指標: レースごとの順位相関(スピアマン)、上位10組の重なり、log q を log p に回帰した傾き。
- 期待払戻の構造: p で見た 100円あたりの期待払戻 p·odds·100 を、p の分位で平均(人気薄ほど低いか)。
出力: data/jra_pipeline/finish_v4/f0_odds_proxy_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import market_yardstick as MY  # noqa: E402

ODDS_DIR = F0.C.PROJECT_ROOT / "data" / "odds_history_sanrenpuku" / "2026"


def main():
    fit = json.loads((F0.OUT_DIR / "dev_f0_market.json").read_text(encoding="utf-8"))["market_fit"]["2020"]
    gamma, lam = fit["gamma"], tuple(fit["lam"])
    win = MY.load_odds([2026])
    per, struct = [], []
    for p in sorted(ODDS_DIR.glob("*.csv.gz")):
        d = pd.read_csv(p, dtype=str)
        d["off"] = pd.to_numeric(d["checkpoint_offset_sec"], errors="coerce")
        d["o"] = pd.to_numeric(d["sanrenpuku_odds"], errors="coerce")
        d = d[d["o"] > 0]
        for rid, g in d.groupby("race_id"):
            g = g[g["off"] == g["off"].min()]
            w = win[win["race_id"] == rid]
            if len(w) < 3:
                continue
            c, pm, um = MY.market_combos(w, gamma, lam)
            kp = MY.combo_keys(c)
            ko = MY.combo_keys(g[["umaban_a", "umaban_b", "umaban_c"]].astype(int).to_numpy())
            o_of = dict(zip(ko.tolist(), g["o"].tolist()))
            oo = np.array([o_of.get(int(k), np.nan) for k in kp])
            m = np.isfinite(oo)
            if m.mean() < 0.95:
                continue
            q = (1 / oo[m]) / np.sum(1 / oo[m])
            pp = pm[m] / pm[m].sum()
            top_p, top_q = set(np.argsort(-pp)[:10]), set(np.argsort(-q)[:10])
            b = np.polyfit(np.log(pp), np.log(q), 1)[0]
            per.append({"race_id": rid, "n": len(um), "spearman": spearmanr(pp, q)[0], "top10_overlap": len(top_p & top_q) / 10,
                        "slope": b, "takeout_implied": 1 - 1 / np.sum(1 / oo[m])})
            struct.append(pd.DataFrame({"p": pp, "ev": pp * oo[m] * 100}))
    R = pd.DataFrame(per)
    S = pd.concat(struct)
    S["bin"] = pd.qcut(np.log(S["p"]), 10, labels=False)
    ev = S.groupby("bin").agg(p_med=("p", "median"), ev_mean=("ev", "mean")).reset_index()
    out = {"n_races": int(len(R)), "dates": sorted({x.name[:8] for x in ODDS_DIR.glob("*.csv.gz")}),
           "gamma": gamma, "lam": list(lam),
           "spearman_median": round(float(R["spearman"].median()), 4), "top10_overlap_mean": round(float(R["top10_overlap"].mean()), 4),
           "slope_median": round(float(R["slope"].median()), 4), "takeout_implied_median": round(float(R["takeout_implied"].median()), 4),
           "ev_by_p_decile": [{"p_median": float(r.p_med), "ev_per_100yen": round(float(r.ev_mean), 1)} for r in ev.itertuples()]}
    (F0.OUT_DIR / "f0_odds_proxy_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
