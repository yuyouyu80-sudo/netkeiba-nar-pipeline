# -*- coding: utf-8 -*-
"""finish_v4 F3: 統合モデル(2026-10-11、プラン第3版 F3)。開発は2018〜2020年、学習は2014年〜前年。2021年以降は読まない。

- 主のモデル: PL(2・3着段階の縮め係数 λ2・λ3、弱い L2)。特徴量は学習の行の SD で割ってから当てはめる(L2 の効き方をそろえる)。
- 群(外して比べる単位):
  - 基礎: 血統+実績(S_own・F_own)、距離の変更(ext・short・ext_amt)。
  - 芝ダ別の状態(F2-2)。外した版では1次元の能力値(ab_mu・ab_sd・ab_new)に戻す。
  - 父の月次(F2-1)・母のきょうだい(F2-5)・主戦距離との差(F2-3): それぞれ「初出走の馬」との交差も同じ群に入れる。
  - 血統×条件: 血統の値10本(父の総合力・脚質・キレ・芝・ダ・距離帯4つ・新馬)と、レースの条件(ダート・log距離・クラス・新馬戦)との交差。
    radar_v2 の C3(条件に線形な要求)と同じ考え方。DL(v3)はローリングの学習し直しが要るため、この版では入れない。
  - 能力の交差: 初出走×血統+実績、斤量×距離、芝ダの差×初芝・初ダ、休み明け×経験(5走以上)。
  - 斤量・休み・前走: kin・d_kin・rest_long・last_fig。
- L2 は FULL(全部入り)で 2018・2019年の評価だけを見て選ぶ。
- 凍結の規則(プランどおり、事前登録に書く): 全部入りから群を1つ外したとき、組の対数スコアの悪化が95%区間(BCa)で0を超えた群だけを残す。
- 不確かさを使う版(MC): 能力値を N(平均, s·SD) から引いて組の確率を平均する(M=32、s は 2018・2019年で選ぶ)。

使い方:
  python model.py --tables     # F2 の特徴量を足した表 f3_tables.parquet(Git管理外)
  python model.py --eval       # → f3_eval_dev.json・f3_utils.parquet・f3_params.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ability_v2 as V  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import pl_ext as P  # noqa: E402

AB = F0.AB
CACHE = F0.OUT_DIR / "f3_tables.parquet"
EVAL_OUT = F0.OUT_DIR / "f3_eval_dev.json"
UTILS_OUT = F0.OUT_DIR / "f3_utils.parquet"
PARAMS_OUT = F0.OUT_DIR / "f3_params.json"
V2_COLS = V.V2 + ["k_n"]
CONDS = ["is_dirt", "logd_c", "class_n", "debut"]
L2_GRID = [0.0, 1e-4, 1e-3, 1e-2]
SELECT_YEARS = (2018, 2019)
MC_M, MC_S = 32, [0.5, 1.0]
log = F0.log

BASE = ["S_own", "F_own", "ext", "short", "ext_amt"]
AB1 = ["ab_mu", "ab_sd", "ab_new"]
GROUPS = {
    "芝ダ別の状態": ["k_mu", "k_sd", "k_new_s", "k_mu_gap"],
    "父の月次": ["sire_m", "新×sire_m"],
    "母のきょうだい": ["dam_m", "新×dam_m"],
    "主戦距離との差": ["dist_dev", "dist_adev"],
    "血統×条件": F0.PED_Z + [f"{z}×{c}" for z in F0.PED_Z for c in CONDS],
    "能力の交差": ["新×S_own", "斤量×距離", "芝ダ差×初芝ダ", "休み明け×経験"],
    "斤量・休み・前走": ["kin", "d_kin", "rest_long", "last_fig"],
}


def spec(groups) -> list:
    f = list(BASE)
    if "芝ダ別の状態" not in groups:
        f += AB1
    for g in GROUPS:
        if g in groups:
            f += GROUPS[g]
    return f


SPECS = {"M5": F0.SPECS["M5"],
         "M6": V.SPECS["M6(全部)"],
         "FULL": spec(list(GROUPS))}
for g in GROUPS:
    SPECS[f"FULL−{g}"] = spec([x for x in GROUPS if x != g])


# ------------------------------------------------------------------ 表
def v2_par() -> dict:
    tr = json.loads(V.TUNE_OUT.read_text(encoding="utf-8"))
    return tr["base_kalman"] | {"rho0": tr["rho0"], "rhoq": tr["rhoq"], "k_s": tr["k_s"], "k_d": tr["k_d"]}


def build_tables(years=F0.YEARS, max_year: int = 2020, cache_in=None, cache_out=None):
    par = v2_par()
    tabs = F0.load_tables(cache_in, max_year)
    ra = V.load_runs(max_year)
    S_all = AB.sire_S(ra)
    out = []
    for Y in years:
        Fv = V.features(ra[ra["year"] <= Y], Y, par, S_all).set_index(["race_id", "horse_id"])[V2_COLS]
        Dy = tabs[Y].join(Fv, on=["race_id", "horse_id"])
        log("tables", Y, len(Dy), "v2 欠け", round(float(Dy[V2_COLS].isna().any(axis=1).mean()), 5))
        out.append(Dy.assign(eval_year=Y))
    T = pd.concat(out, ignore_index=True)
    assert T["year"].max() <= max_year
    T.to_parquet(cache_out or CACHE, index=False)


def derived(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    D["logd_c"] = D["logd"] - np.log(1600.0)
    new = (D["k_n"] == 0).astype(float)
    D["新×sire_m"] = new * D["sire_m"]
    D["新×dam_m"] = new * D["dam_m"]
    D["新×S_own"] = new * D["S_own"]
    D["斤量×距離"] = D["kin"] * D["logd_c"]
    D["芝ダ差×初芝ダ"] = D["k_mu_gap"] * D["k_new_s"]
    D["休み明け×経験"] = D["rest_long"] * (D["k_n"] >= 5).astype(float)
    for z in F0.PED_Z:
        for c in CONDS:
            D[f"{z}×{c}"] = D[z] * D[c]
    return D


def load_tables(cache=None, max_year: int = 2020) -> dict:
    T = pd.read_parquet(cache or CACHE)
    assert T["year"].max() <= max_year
    return {int(Y): derived(g.drop(columns="eval_year")) for Y, g in T.groupby("eval_year")}


# ------------------------------------------------------------------ 当てはめ
def fit_std(tr: pd.DataFrame, feats: list, l2: float):
    sd = tr[feats].std().replace(0, 1).fillna(1).to_numpy(float)
    b = P.RaceBatch(tr.assign(**{f: tr[f] / s for f, s in zip(feats, sd)}), feats)
    beta, lam = P.fit(b, kind="pl", fit_lam=True, l2=l2)
    return beta / sd, lam        # 生の特徴量に掛ける係数


def test_eval(te: pd.DataFrame, feats: list, beta, lam, pay) -> tuple[pd.Series, pd.DataFrame, np.ndarray]:
    bt = P.RaceBatch(te, feats)
    ll = pd.Series(P.race_ll(bt, beta, lam, "pl"), index=bt.race_ids)
    u = te[feats].fillna(0).to_numpy(float) @ beta
    cm = V.combo_metrics(te.assign(_u=u), pay, lam)
    return ll, cm, u


def mc_combo(te: pd.DataFrame, u: np.ndarray, beta: dict, lam, pay, s: float, M: int = MC_M, seed: int = 20261011) -> pd.DataFrame:
    """能力値の不確かさ: u_m = u + c·s·SD·ε。c は k_mu を1増やしたときの効用の変化(芝ダ差とその交差を通る分も含む)。"""
    c = beta.get("k_mu", 0.0) - beta.get("k_mu_gap", 0.0) - beta.get("芝ダ差×初芝ダ", 0.0) * te["k_new_s"].to_numpy(float)
    sdv = te["k_sd"].to_numpy(float) * c * s
    rng = np.random.default_rng(seed)
    rows = []
    d = te.assign(_u=u, _s=sdv)
    for rid, g in d.groupby("race_id"):
        if rid not in pay:
            continue
        um = g["umaban_num"].to_numpy(int)
        wins = [x for x, _ in pay[rid] if x <= set(um)]
        if not wins:
            continue
        uu, ss = g["_u"].to_numpy(float), g["_s"].to_numpy(float)
        acc = None
        for _ in range(M):
            combos, p = P.triple_probs(uu + ss * rng.standard_normal(len(uu)), "pl", tuple(lam))
            acc = p if acc is None else acc + p
        p = acc / M
        idx = {frozenset(um[x]): i for i, x in enumerate(combos)}
        wi = [idx[x] for x in wins]
        rank = np.empty(len(p), int)
        rank[np.argsort(-p)] = np.arange(1, len(p) + 1)
        rows.append({"race_id": rid, "logscore": float(np.log(np.mean(p[wi]))), "best_rank": int(min(rank[i] for i in wi))})
    return pd.DataFrame(rows).set_index("race_id")


def summarize(LL, CM, blk, base: str) -> dict:
    out = {}
    for k in LL:
        d = (LL[k] - LL[base]).dropna()
        c = CM[k].join(CM[base], rsuffix="_b", how="inner")
        hit = lambda n, x: (x <= n).astype(float)  # noqa: E731
        out[k] = {"ll": round(float(LL[k].mean()), 5), "combo_logscore": round(float(CM[k]["logscore"].mean()), 5),
                  "hit": {f"N{n}": round(float((CM[k]["best_rank"] <= n).mean()), 4) for n in (1, 5, 10, 20, 30)},
                  f"dll_vs_{base}": P.block_boot(d.to_numpy(), blk.loc[d.index].to_numpy(), n=1000),
                  f"combo_vs_{base}": P.block_boot((c["logscore"] - c["logscore_b"]).to_numpy(), blk.loc[c.index].to_numpy(), n=1000),
                  f"hit10_vs_{base}": P.block_boot((hit(10, c["best_rank"]) - hit(10, c["best_rank_b"])).to_numpy(), blk.loc[c.index].to_numpy(), n=1000),
                  "by_year_combo": (c["logscore"] - c["logscore_b"]).groupby(c.index.str[:4]).mean().round(5).to_dict()}
    return out


def evaluate() -> dict:
    t0 = time.time()
    tabs = load_tables()
    pay = F0.load_sanrenpuku(F0.YEARS)
    blk = tabs[F0.YEARS[-1]].drop_duplicates("race_id").set_index("race_id")["block"]
    res = {"eval_years": F0.YEARS, "select_years": SELECT_YEARS, "groups": GROUPS, "specs": SPECS, "l2_grid": {}}
    # 1. L2 の選択(FULL、2018・2019年だけ)
    for l2 in L2_GRID:
        lls = []
        for Y in SELECT_YEARS:
            D = tabs[Y]
            tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
            beta, lam = fit_std(tr, SPECS["FULL"], l2)
            lls.append(P.race_ll(P.RaceBatch(te, SPECS["FULL"]), beta, lam, "pl"))
        res["l2_grid"][str(l2)] = round(float(np.mean(np.concatenate(lls))), 5)
        log("l2", l2, res["l2_grid"][str(l2)])
    l2 = float(max(res["l2_grid"], key=lambda k: res["l2_grid"][k]))
    res["l2"] = l2
    # 2. 全モデル(2018〜2020年)
    LL = {k: [] for k in SPECS}
    CM = {k: [] for k in SPECS}
    U = []
    params = {}
    betas = {}
    for Y in F0.YEARS:
        D = tabs[Y]
        tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
        out = te[["race_id", "horse_id", "year", "pos", "umaban_num", "block", "is_debut_race", "class_ord", "nonfinish", "k_sd", "k_new_s"]].copy()
        for k, feats in SPECS.items():
            beta, lam = fit_std(tr, feats, l2 if k not in ("M5", "M6") else 0.0)
            ll, cm, u = test_eval(te, feats, beta, lam, pay)
            LL[k].append(ll)
            CM[k].append(cm)
            betas[(Y, k)] = dict(zip(feats, beta))
            if k in ("M5", "M6", "FULL"):
                out[f"u_{k}_pl"] = u
                params[f"{Y}_{k}_pl"] = {"lam": list(lam), "beta": {f: round(float(b), 6) for f, b in zip(feats, beta)}}
            log(Y, k, "ll", round(float(ll.mean()), 4), "combo", round(float(cm["logscore"].mean()), 4), round(time.time() - t0), "秒")
        U.append(out)
    LL = {k: pd.concat(v) for k, v in LL.items()}
    CM = {k: pd.concat(v) for k, v in CM.items()}
    res["vs_M6"] = summarize(LL, CM, blk, "M6")
    res["vs_FULL"] = {k: v for k, v in summarize(LL, CM, blk, "FULL").items() if k.startswith("FULL−")}
    # 3. 凍結の規則: 外すと組の対数スコアがはっきり悪くなる群だけを残す
    keep = []
    for g in GROUPS:
        lo, hi = res["vs_FULL"][f"FULL−{g}"]["combo_vs_FULL"]["ci_bca"]
        if hi < 0:                       # 外した版 − 全部入り の区間が0より下 = 外すと悪化
            keep.append(g)
    res["kept_groups"] = keep
    res["final_spec"] = spec(keep)
    # 最終の版(残した群だけ)を当てはめる
    LLf, CMf, Uf = [], [], []
    for Y in F0.YEARS:
        D = tabs[Y]
        tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
        beta, lam = fit_std(tr, res["final_spec"], l2)
        ll, cm, u = test_eval(te, res["final_spec"], beta, lam, pay)
        LLf.append(ll)
        CMf.append(cm)
        Uf.append(u)
        betas[(Y, "FINAL")] = dict(zip(res["final_spec"], beta))
        params[f"{Y}_FINAL_pl"] = {"lam": list(lam), "beta": {f: round(float(b), 6) for f, b in zip(res["final_spec"], beta)}}
    U = pd.concat(U, ignore_index=True)
    U["u_FINAL_pl"] = np.concatenate(Uf)
    LL["FINAL"], CM["FINAL"] = pd.concat(LLf), pd.concat(CMf)
    # 4. 不確かさを使う版(FINAL に当てる。s は 2018・2019年の組の対数スコアで選ぶ)
    if "芝ダ別の状態" in keep:
        mc = {}
        for s in MC_S:
            parts = []
            for Y in F0.YEARS:
                te = tabs[Y][tabs[Y]["year"] == Y]
                parts.append(mc_combo(te, U.loc[U["year"] == Y, "u_FINAL_pl"].to_numpy(), betas[(Y, "FINAL")],
                                      params[f"{Y}_FINAL_pl"]["lam"], pay, s))
            mc[s] = pd.concat(parts)
            sel = mc[s][mc[s].index.str[:4].astype(int).isin(SELECT_YEARS)]["logscore"].mean()
            log("mc", s, round(float(sel), 5))
        base_sel = CM["FINAL"][CM["FINAL"].index.str[:4].astype(int).isin(SELECT_YEARS)]["logscore"].mean()
        best = max(MC_S, key=lambda s: mc[s][mc[s].index.str[:4].astype(int).isin(SELECT_YEARS)]["logscore"].mean())
        best_sel = mc[best][mc[best].index.str[:4].astype(int).isin(SELECT_YEARS)]["logscore"].mean()
        res["mc"] = {"s_grid": {str(s): round(float(mc[s][mc[s].index.str[:4].astype(int).isin(SELECT_YEARS)]["logscore"].mean()), 5) for s in MC_S},
                     "deterministic_select": round(float(base_sel), 5), "s": best, "adopted": bool(best_sel > base_sel)}
        LL["FINAL+MC"] = LL["FINAL"]
        CM["FINAL+MC"] = mc[best]
    res["final_vs_M6"] = summarize({k: LL[k] for k in ("M6", "FINAL") + (("FINAL+MC",) if "FINAL+MC" in LL else ())},
                                   {k: CM[k] for k in ("M6", "FINAL") + (("FINAL+MC",) if "FINAL+MC" in CM else ())}, blk, "M6")
    res["final_vs_M5"] = summarize({k: LL[k] for k in ("M5", "FINAL")}, {k: CM[k] for k in ("M5", "FINAL")}, blk, "M5")
    U.to_parquet(UTILS_OUT, index=False)
    PARAMS_OUT.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.tables:
        build_tables()
    if a.eval:
        res = evaluate()
        EVAL_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log("final", res["kept_groups"], res.get("mc"))


if __name__ == "__main__":
    main()
