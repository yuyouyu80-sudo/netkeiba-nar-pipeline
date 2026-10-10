# -*- coding: utf-8 -*-
"""finish_v4 F0-0: 開発期間(2018〜2020年)の M1・M5 の効用を保存し、3連複の指標を出す(2026-10-10、プラン第3版 F0)。

- M1 = 血統+実績(S_own・F_own)、M5 = M1 + 距離の変更・能力値・斤量・休み明け・前走の能力値(radar_v2_ability の開発評価と同じ特徴量)。
- 毎年 2014〜Y−1年で学習し Y年で評価する(Y=2018・2019・2020)。2021年以降は読まない(assert)。
- 組の確率は2通り: PL(2・3着段階の縮め係数 λ2・λ3 を推定)/ 3頭の集合の条件付きロジット。
- 指標: 組の対数スコア(実際の3頭の組に付けた確率の log、同着は当たり組の確率の平均)、確率順 N 点買いの的中率と回収率(N=1〜50)。
- 対象はこの段階では完走馬だけ(中止馬・取消の扱いは F0-1 で入れる)。市場(オッズ)は一切使わない。
  特徴量の作成は radar_v2_ability(内部で radar_v2_eval の Tables・add_own を使う。期間に依存する build_D は呼ばない)。
出力: data/jra_pipeline/finish_v4/dev_utils.parquet(Git管理外)、dev_f0_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
import pl_ext as P  # noqa: E402
import radar_v2_ability as AB  # noqa: E402
import radar_v2_common as C  # noqa: E402

OUT_DIR = C.PROJECT_ROOT / "data" / "jra_pipeline" / "finish_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS = (2018, 2019, 2020)
TRAIN_START = 2014
SPECS = {"M1": ["S_own", "F_own"],
         "M5": ["S_own", "F_own", "ext", "short", "ext_amt", "ab_mu", "ab_sd", "ab_new", "kin", "d_kin", "rest_long", "last_fig"]}
AB_COLS = ["ab_mu", "ab_sd", "ab_new", "ew_mu", "last_fig", "kin", "d_kin", "rest_long"]
NS = list(range(1, 51))
log = AB.log


def load_sanrenpuku(years) -> dict:
    """{race_id: [(組の馬番 frozenset, 払戻円)]}。同着で複数の当たり組があれば全部。"""
    out = {}
    for y in years:
        for p in sorted((C.PROJECT_ROOT / "data" / "payouts" / str(y)).glob("*.csv")):
            d = pd.read_csv(p, dtype=str)
            d = d[d["bet_type"] == "3連複"]
            for rid, comb, pay in zip(d["race_id"], d["combination"], d["payout"]):
                try:
                    s = frozenset(int(x) for x in comb.split("-"))
                    out.setdefault(rid, []).append((s, int(float(pay))))
                except (ValueError, AttributeError):
                    pass
    return out


def build() -> tuple[pd.DataFrame, dict]:
    tp = json.loads(AB.TUNE_OUT.read_text(encoding="utf-8"))
    runs, D = AB.build_eval_table(tp)
    assert D["year"].max() <= 2020 and runs["year"].max() <= 2020
    rows, params = [], {}
    for Y in YEARS:
        F = AB.ability_features(runs[runs["year"] <= Y], Y, tp).set_index(["race_id", "horse_id"])[AB_COLS]
        Dy = D[D["year"] <= Y].join(F, on=["race_id", "horse_id"])
        tr, te = Dy[Dy["year"].between(TRAIN_START, Y - 1)], Dy[Dy["year"] == Y]
        out = te[["race_id", "horse_id", "year", "pos", "umaban_num", "block", "is_debut_race", "class_ord"]].copy()
        for name, feats in SPECS.items():
            btr = P.RaceBatch(tr, feats)
            for kind in ("pl", "set"):
                t0 = time.time()
                beta, lam = P.fit(btr, kind=kind, fit_lam=(kind == "pl"))
                out[f"u_{name}_{kind}"] = te[feats].fillna(0).to_numpy(float) @ beta
                params[f"{Y}_{name}_{kind}"] = {"beta": beta.round(5).tolist(), "lam": lam}
                log("fit", Y, name, kind, "lam", np.round(lam, 3), round(time.time() - t0), "秒")
        rows.append(out)
    return pd.concat(rows, ignore_index=True), params


def metrics(U: pd.DataFrame, pay: dict, lam: dict) -> dict:
    res = {}
    cols = [c for c in U.columns if c.startswith("u_")]
    per = {c: [] for c in cols}
    for rid, g in U.groupby("race_id"):
        if rid not in pay:
            continue
        um = g["umaban_num"].to_numpy(int)
        Y = int(g["year"].iloc[0])
        wins = [(s, v) for s, v in pay[rid] if s <= set(um)]
        if not wins:
            continue
        for c in cols:
            _, name, kind = c.split("_")
            lm = tuple(lam[f"{Y}_{name}_{kind}"]["lam"])
            combos, p = P.triple_probs(g[c].to_numpy(float), kind, lm)
            sets = [frozenset(um[x]) for x in combos]
            idx = {s: i for i, s in enumerate(sets)}
            wi = [idx[s] for s, _ in wins]
            order = np.argsort(-p)
            rank = np.empty(len(p), int)
            rank[order] = np.arange(1, len(p) + 1)
            best = min(rank[i] for i in wi)
            # N 点買いの払戻(当たった当たり組の払戻の合計)
            payN = np.zeros(len(NS))
            for i, (s, v) in zip(wi, wins):
                payN += np.where(rank[i] <= np.array(NS), v, 0)
            per[c].append({"race_id": rid, "logscore": float(np.log(np.mean(p[wi]))), "best_rank": int(best),
                           "n_combos": len(p), "pay": payN, "block": g["block"].iloc[0],
                           "debut_or_maiden": bool(g["is_debut_race"].iloc[0] or g["class_ord"].iloc[0] == 0),
                           "field": len(um)})
    for c, lst in per.items():
        d = pd.DataFrame(lst)
        hit = np.array([[r <= n for n in NS] for r in d["best_rank"]])
        payN = np.vstack(d["pay"].to_numpy())
        res[c] = {"n_races": int(len(d)), "logscore": round(float(d["logscore"].mean()), 5),
                  "uniform_logscore": round(float(-np.log(d["n_combos"]).mean()), 5),
                  "hit_rate": {n: round(float(hit[:, i].mean()), 4) for i, n in enumerate(NS) if n in (1, 5, 10, 15, 20, 30, 50)},
                  "roi": {n: round(float(payN[:, i].sum() / (100 * n * len(d))), 4) for i, n in enumerate(NS) if n in (1, 5, 10, 15, 20, 30, 50)}}
        res[c]["_d"] = d
    return res


def main():
    t0 = time.time()
    U, params = build()
    U.to_parquet(OUT_DIR / "dev_utils.parquet", index=False)
    pay = load_sanrenpuku(YEARS)
    M = metrics(U, pay, params)
    out = {"years": YEARS, "train_start": TRAIN_START, "specs": SPECS, "params": params, "models": {}}
    base = M["u_M1_pl"]["_d"].set_index("race_id")
    for c, m in M.items():
        d = m.pop("_d").set_index("race_id")
        j = d.join(base[["logscore"]], rsuffix="_M1pl", how="inner")
        m["logscore_vs_M1pl"] = P.block_boot((j["logscore"] - j["logscore_M1pl"]).to_numpy(), j["block"].to_numpy(), n=1000)
        out["models"][c] = m
        log(c, m["logscore"], "uniform", m["uniform_logscore"], "hit", m["hit_rate"], "roi", m["roi"])
    out["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "dev_f0_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("wrote dev_f0_metrics.json")


if __name__ == "__main__":
    main()
