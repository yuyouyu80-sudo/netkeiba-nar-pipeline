# -*- coding: utf-8 -*-
"""finish_v4 F2: 能力値の改良(2026-10-11、プラン第3版 F2)。生年月日・生産者は使わない(F1 のゲート未達のため、F2-4・F2-6 は保留)。

土台は radar_v2_ability(走ごとの能力値 fig と1次元のカルマンフィルタ)。走の表は中止の行を足したもの(nonfinish.py)。
すべての係数は評価年 Y の前年末まで、状態のパラメータ・縮約の強さは2017年末までで選ぶ。2021年以降は読まない(assert)。

1. 芝・ダ別の2次元の状態(kalman2): 状態 (芝, ダ) の平均と共分散。初出走前の相関 ρ0、走の間の分散の加算の相関 ρq。
   観測はその走の芝ダの成分だけ。ρ0=ρq=1・芝ダ同じ事前で radar_v2_ability.kalman と一致する(テスト)。
   中止の行は「その日の発走前の値を読むだけ」で、状態を進めない(完走馬の値が中止の行の有無で変わらない)。
   特徴量: その走の芝ダの平均・SD・その芝ダの走数0(初芝・初ダ)・もう一方の平均。
2. 父の産駒の能力(月次、sire_m): 前月末までの産駒の走の fig の残差(年齢×半期×芝ダの平均を引いたもの)を、馬ごとに平均し、
   重み min(走数,5) で父ごとに加重平均、本人を除き k_s で0へ縮める。事前の平均 a + b·S + c·sire_m にも入れる。
3. 母のきょうだいの能力(dam_m): 同じ母の他の馬の、前月末までの「残差 − その時点の父の値」を同じ形で縮める(k_d)。
4. 主戦距離との差: その走の log距離 − それまでの完走の走の log距離の平均(dist_dev)と、その絶対値。

使い方:
  python ability_v2.py --tune     # 2017年末まで: ρ0・ρq・k_s・k_d → ability_v2_tune.json
  python ability_v2.py --eval     # 2018〜2020年: M5 に対する ΔLL・組の対数スコア・的中率(外して比べる)→ ability_v2_eval_dev.json
"""
import argparse
import csv
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import nonfinish as NF  # noqa: E402
import pl_ext as P  # noqa: E402

AB, A, C = F0.AB, F0.A, F0.C
TUNE_OUT = F0.OUT_DIR / "ability_v2_tune.json"
EVAL_OUT = F0.OUT_DIR / "ability_v2_eval_dev.json"
DAM_CACHE = F0.OUT_DIR / "dam_ids.parquet"
TUNE_END = 2017
W_CAP = 5
K_GRID = [10, 20, 40, 80, 160, 320]
log = F0.log


# ------------------------------------------------------------------ 入力
def dam_ids(horse_ids) -> pd.Series:
    cache = pd.read_parquet(DAM_CACHE) if DAM_CACHE.exists() else pd.DataFrame(columns=["horse_id", "dam_id"])
    have = set(cache["horse_id"])
    rows = []
    for h in horse_ids:
        if h in have:
            continue
        p = C.PED_DIR / f"{h}.csv"
        dam = None
        if p.exists():
            with open(p, encoding="utf-8", newline="") as f:
                r = csv.reader(f)
                hdr = next(r, None)
                row = next(r, None)
            if hdr and row:
                dam = dict(zip(hdr, row)).get("ped_D_horse_id") or None
        rows.append({"horse_id": h, "dam_id": dam})
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_parquet(DAM_CACHE, index=False)
    return cache.drop_duplicates("horse_id").set_index("horse_id")["dam_id"]


def load_runs(max_year: int = 2020) -> pd.DataFrame:
    base = pd.read_parquet(A.RUNS)
    aug, _ = NF.augment(base, max_year)
    ra = F0.load_aug_runs(aug)
    assert ra["year"].max() <= max_year
    ra["dam_id"] = ra["horse_id"].map(dam_ids(ra["horse_id"].unique()))
    ra["is_turf"] = (ra["surface"] == "芝").astype(int)
    ra["month0"] = ra["date"].dt.to_period("M").dt.start_time
    return ra


# ------------------------------------------------------------------ 1. 芝ダ別の2次元の状態
def kalman2(runs: pd.DataFrame, fig: pd.Series, pm_t: np.ndarray, pm_d: np.ndarray, par: dict, growth: dict) -> pd.DataFrame:
    """runs は horse_id・date 順。nonfinish の行は発走前の値だけ読み、状態を進めない。"""
    h = runs["horse_id"].to_numpy()
    f = fig.to_numpy(float)
    sidx = 1 - runs["is_turf"].to_numpy()          # 0=芝, 1=ダ
    nf = runs["nonfinish"].to_numpy(bool) if "nonfinish" in runs else np.zeros(len(h), bool)
    rest = runs["rest_days"].fillna(0).to_numpy(float)
    gs = AB.growth_seg(runs).map(growth).fillna(0).to_numpy(float)
    P0, q0, q1, Rv, r0, rq = par["P0"], par["q0"], par["q1"], par["R"], par["rho0"], par["rhoq"]
    n = len(h)
    out = np.full((n, 6), np.nan)                  # mu_s, sd_s, n_s, mu_o, n_all, ll
    last_h, has = None, False
    m0 = m1 = p00 = p01 = p11 = 0.0
    c0 = c1 = 0
    for i in range(n):
        if h[i] != last_h:
            last_h, has = h[i], False
        if not has:
            a0, a1 = pm_t[i], pm_d[i]
            b00, b11, b01 = P0, P0, P0 * r0
            k0 = k1 = 0
        else:
            g = gs[i]
            a0, a1 = m0 + g, m1 + g
            dq = q0 + q1 * rest[i] / 30.0
            b00, b11, b01 = p00 + dq, p11 + dq, p01 + dq * rq
            k0, k1 = c0, c1
        s = sidx[i]
        mu_s, var_s, mu_o = (a0, b00, a1) if s == 0 else (a1, b11, a0)
        out[i, 0], out[i, 1], out[i, 2], out[i, 3], out[i, 4] = mu_s, np.sqrt(var_s), (k0 if s == 0 else k1), mu_o, k0 + k1
        if nf[i]:
            continue
        # 完走の行: 予測を確定し、fig があれば更新する
        if not np.isnan(f[i]):
            S_ = var_s + Rv
            out[i, 5] = -0.5 * (np.log(2 * np.pi * S_) + (f[i] - mu_s) ** 2 / S_)
            e = f[i] - mu_s
            if s == 0:
                g0, g1 = b00 / S_, b01 / S_
                a0, a1 = a0 + g0 * e, a1 + g1 * e
                b00, b01, b11 = b00 - g0 * b00, b01 - g0 * b01, b11 - g1 * b01
                k0 += 1
            else:
                g0, g1 = b01 / S_, b11 / S_
                a0, a1 = a0 + g0 * e, a1 + g1 * e
                b00, b01, b11 = b00 - g0 * b01, b01 - g1 * b01, b11 - g1 * b11
                k1 += 1
        m0, m1, p00, p01, p11, c0, c1, has = a0, a1, b00, b01, b11, k0, k1, True
    return pd.DataFrame(out, columns=["k_mu", "k_sd", "k_n_s", "k_mu_o", "k_n", "k_ll"], index=runs.index)


# ------------------------------------------------------------------ 2・3. 月次の父・母の値
def resid_table(runs: pd.DataFrame, fig: pd.Series, end_year: int) -> pd.Series:
    key = AB.growth_seg(runs) + "|" + runs["surface"]
    m = (runs["year"] <= end_year) & fig.notna()
    mu = fig[m].groupby(key[m]).mean()
    return fig - key.map(mu)


def horse_events(runs: pd.DataFrame, r: pd.Series) -> pd.DataFrame:
    """完走で r がある走ごとの、馬の累積の平均 x と重み w(その走の後)。"""
    e = runs.loc[r.notna() & ~runs["nonfinish"], ["horse_id", "date", "month0", "ped_S_horse_id", "dam_id"]].assign(r=r)
    e = e.sort_values(["horse_id", "date"])
    g = e.groupby("horse_id")
    e["n"] = g.cumcount() + 1
    e["x"] = g["r"].cumsum() / e["n"]
    e["w"] = np.minimum(e["n"], W_CAP).astype(float)
    e["w_prev"] = g["w"].shift(1).fillna(0.0)
    e["x_prev"] = g["x"].shift(1).fillna(0.0)
    return e


def group_month_value(ev: pd.DataFrame, key: str, val: str, q: pd.DataFrame, k: float) -> pd.Series:
    """q(列 horse_id・month0・key)の各行: 前月末までの group(key)の Σw·val の縮約平均(本人を除く)。

    ev は horse_events に val 列(その走の後の馬の値)を足したもの。group の和は「馬の値の入れ替え」の差分の累積。"""
    ev = ev.dropna(subset=[key]).copy()
    ev["vprev"] = ev.groupby("horse_id")[val].shift(1).fillna(0.0)
    ev["dswx"] = ev["w"] * ev[val] - ev["w_prev"] * ev["vprev"]
    ev["dsw"] = ev["w"] - ev["w_prev"]
    ev = ev.sort_values([key, "date"])
    ev["Sx"] = ev.groupby(key)["dswx"].cumsum()
    ev["Sw"] = ev.groupby(key)["dsw"].cumsum()
    qq = q[["horse_id", "month0", key]].reset_index().rename(columns={"index": "_i"})
    qq = qq.dropna(subset=[key]).sort_values("month0")
    grp = ev[[key, "date", "Sx", "Sw"]].sort_values("date")
    a = pd.merge_asof(qq, grp, left_on="month0", right_on="date", by=key, allow_exact_matches=False)
    own = ev[["horse_id", "date", "w", val]].sort_values("date").rename(columns={"date": "d_own"})
    a = pd.merge_asof(a.sort_values("month0"), own, left_on="month0", right_on="d_own", by="horse_id", allow_exact_matches=False)
    Sx, Sw = a["Sx"].fillna(0) - (a["w"] * a[val]).fillna(0), a["Sw"].fillna(0) - a["w"].fillna(0)
    out = pd.Series(np.nan, index=q.index)
    out[a["_i"].to_numpy()] = (Sx / (Sw + k)).to_numpy()
    return out.fillna(0.0)


def monthly_values(runs: pd.DataFrame, fig: pd.Series, end_year: int, k_s: float, k_d: float) -> pd.DataFrame:
    r = resid_table(runs, fig, end_year)
    ev = horse_events(runs, r)
    sire_m = group_month_value(ev, "ped_S_horse_id", "x", runs, k_s)
    # 母: 各走の後の馬の値 −(その走の月の父の値)。父の値は同じ表の sire_m(本人除外)
    ev["sire_at"] = sire_m.reindex(ev.index).to_numpy()
    ev["z"] = ev["x"] - ev["sire_at"]
    dam_m = group_month_value(ev, "dam_id", "z", runs, k_d)
    return pd.DataFrame({"sire_m": sire_m, "dam_m": dam_m}, index=runs.index)


# ------------------------------------------------------------------ 4. 主戦距離との差
def dist_features(runs: pd.DataFrame) -> pd.DataFrame:
    x = runs["logd"].where(~runs["nonfinish"])
    g = runs.assign(_x=x.fillna(0), _n=x.notna().astype(float)).groupby("horse_id")
    s = g["_x"].cumsum() - x.fillna(0)
    c = g["_n"].cumsum() - x.notna().astype(float)
    dev = (runs["logd"] - s / c.replace(0, np.nan)).fillna(0.0)
    return pd.DataFrame({"dist_dev": dev, "dist_adev": dev.abs()}, index=runs.index)


# ------------------------------------------------------------------ 年ごとの特徴量
def priors(runs, fig, S, sm, end_year):
    """初出走の fig ~ a + b·S + c·sire_m を芝・ダ別に(end_year 年末まで)。返り値: 事前の平均 (芝, ダ) の配列。"""
    m0 = (runs["career"] == 0) & (runs["year"] <= end_year) & fig.notna() & S.notna() & ~runs["nonfinish"]
    coef = {}
    for s in ["芝", "ダ"]:
        m = m0 & (runs["surface"] == s)
        X = np.column_stack([np.ones(m.sum()), S[m], sm[m]])
        coef[s] = np.linalg.lstsq(X, fig[m], rcond=None)[0]
    Xa = np.column_stack([np.ones(len(runs)), S.fillna(0), sm])
    return Xa @ coef["芝"], Xa @ coef["ダ"], {s: [float(v) for v in c] for s, c in coef.items()}


def features(ra: pd.DataFrame, Y: int, par: dict, S_all: pd.Series) -> pd.DataFrame:
    """Y年の評価用(係数は Y−1年末まで)。ra は Y年末まで。"""
    fin = ra[~ra["nonfinish"]]
    Pf = AB.fit_fig(fin, Y - 1)
    fig = AB.apply_fig(ra, Pf)
    gr = AB.growth_table(fin, fig[fin.index], Y - 1)
    mv = monthly_values(ra, fig, Y - 1, par["k_s"], par["k_d"])
    S = S_all[ra.index]
    pt, pdirt, coef = priors(ra, fig, S, mv["sire_m"], Y - 1)
    st = kalman2(ra, fig, pt, pdirt, par, gr)
    out = pd.concat([ra[["race_id", "horse_id", "year"]], st, mv, dist_features(ra)], axis=1)
    out["k_new_s"] = (out["k_n_s"] == 0).astype(float)
    out["k_mu_gap"] = out["k_mu_o"] - out["k_mu"]
    return out


# ------------------------------------------------------------------ 2017年末までの選択
def tune() -> dict:
    t0 = time.time()
    ra = load_runs(TUNE_END)
    tp = json.loads(AB.TUNE_OUT.read_text(encoding="utf-8"))
    fin = ra[~ra["nonfinish"]]
    Pf = AB.fit_fig(fin, TUNE_END)
    fig = AB.apply_fig(ra, Pf)
    gr = AB.growth_table(fin, fig[fin.index], TUNE_END)
    S = AB.sire_S(ra)
    ev = ra["year"].between(2015, 2017).to_numpy() & ~ra["nonfinish"].to_numpy()
    res = {"end": TUNE_END, "base_kalman": tp["kalman"], "k_grid": [], "rho_grid": []}
    # k_s・k_d: 初出走の fig の予測の平均二乗誤差(2015〜2017年の初出走)
    deb = (ra["career"] == 0) & fig.notna() & S.notna() & ~ra["nonfinish"]
    best_k = None
    # 1回目の格子 {5,10,20,40}² は端(40・40)が最良だったため広げた(2017年末まで、2026-10-11)
    for k_s, k_d in itertools.product(K_GRID, K_GRID):
        mv = monthly_values(ra, fig, TUNE_END, k_s, k_d)
        fit_m = deb & (ra["year"] <= 2014)
        tst = deb & ra["year"].between(2015, 2017)
        X = lambda m: np.column_stack([np.ones(m.sum()), S[m], mv.loc[m, "sire_m"], mv.loc[m, "dam_m"]])  # noqa: E731
        b = np.linalg.lstsq(X(fit_m), fig[fit_m], rcond=None)[0]
        mse = float(np.mean((fig[tst] - X(tst) @ b) ** 2))
        X0 = lambda m: np.column_stack([np.ones(m.sum()), S[m]])  # noqa: E731
        b0 = np.linalg.lstsq(X0(fit_m), fig[fit_m], rcond=None)[0]
        mse0 = float(np.mean((fig[tst] - X0(tst) @ b0) ** 2))
        res["k_grid"].append({"k_s": k_s, "k_d": k_d, "mse": mse, "mse_S_only": mse0, "b": [float(x) for x in b]})
        if best_k is None or mse < best_k[1]:
            best_k = ((k_s, k_d), mse)
        log("k", k_s, k_d, round(mse, 5), "S only", round(mse0, 5))
    res["k_s"], res["k_d"] = best_k[0]
    mv = monthly_values(ra, fig, TUNE_END, res["k_s"], res["k_d"])
    pt, pdirt, coef = priors(ra, fig, S, mv["sire_m"], TUNE_END)
    res["prior_coef"] = coef
    best = None
    for r0, rq in itertools.product([0.5, 0.7, 0.85, 1.0], [0.5, 0.8, 1.0]):
        par = tp["kalman"] | {"rho0": r0, "rhoq": rq}
        st = kalman2(ra, fig, pt, pdirt, par, gr)
        ll = float(np.nanmean(st["k_ll"].to_numpy()[ev]))
        res["rho_grid"].append({"rho0": r0, "rhoq": rq, "ll": ll})
        log("rho", r0, rq, round(ll, 5))
        if best is None or ll > best[1]:
            best = ((r0, rq), ll)
    res["rho0"], res["rhoq"] = best[0]
    res["ll_best"] = best[1]
    # 1次元(元の事前)の同じ走での尤度(比較用)
    a, b = AB.prior_fit(fin, fig[fin.index], S[fin.index], TUNE_END)
    st1 = AB.kalman(fin, fig[fin.index], a + b * S[fin.index].fillna(0), tp["kalman"], gr)
    ev1 = fin["year"].between(2015, 2017).to_numpy()
    res["ll_1d"] = float(np.nanmean(st1["ab_ll"].to_numpy()[ev1]))
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------ 2018〜2020年の開発評価
V2 = ["k_mu", "k_sd", "k_new_s", "k_mu_gap", "sire_m", "dam_m", "dist_dev", "dist_adev"]
M5 = F0.SPECS["M5"]
SPECS = {
    "M5": M5,
    "M5+芝ダ別の状態": [c for c in M5 if c not in ("ab_mu", "ab_sd", "ab_new")] + ["k_mu", "k_sd", "k_new_s", "k_mu_gap"],
    "M5+父の月次": M5 + ["sire_m"],
    "M5+母のきょうだい": M5 + ["dam_m"],
    "M5+主戦距離との差": M5 + ["dist_dev", "dist_adev"],
    "M6(全部)": [c for c in M5 if c not in ("ab_mu", "ab_sd", "ab_new")] + V2,
}


def evaluate(par: dict) -> dict:
    t0 = time.time()
    tabs = F0.load_tables()
    ra = load_runs(2020)
    S_all = AB.sire_S(ra)
    pay = F0.load_sanrenpuku(F0.YEARS)
    per = {k: [] for k in SPECS}
    combo = {k: [] for k in SPECS}
    for Y in F0.YEARS:
        r = ra[ra["year"] <= Y]
        Fv = features(r, Y, par, S_all).set_index(["race_id", "horse_id"])[V2]
        Dy = tabs[Y].join(Fv, on=["race_id", "horse_id"])
        miss = Dy[V2].isna().any(axis=1).mean()
        log("eval", Y, "rows", len(Dy), "v2 欠け", round(float(miss), 5))
        tr, te = Dy[Dy["year"].between(F0.TRAIN_START, Y - 1)], Dy[Dy["year"] == Y]
        for name, feats in SPECS.items():
            beta, lam = P.fit(P.RaceBatch(tr, feats), kind="pl", fit_lam=True)
            bt = P.RaceBatch(te, feats)
            ll = P.race_ll(bt, beta, lam, "pl")
            per[name].append(pd.Series(ll, index=bt.race_ids))
            u = te[feats].fillna(0).to_numpy(float) @ beta
            combo[name].append(combo_metrics(te.assign(_u=u), pay, lam))
            log(Y, name, "ll", round(float(np.mean(ll)), 4), "lam", np.round(lam, 3))
    PL_ = {k: pd.concat(v) for k, v in per.items()}
    CM = {k: pd.concat(v) for k, v in combo.items()}
    blk = tabs[F0.YEARS[-1]].drop_duplicates("race_id").set_index("race_id")["block"]
    res = {"eval_years": F0.YEARS, "par": par, "n_races": int(len(PL_["M5"])), "models": {}}
    for k in SPECS:
        d = (PL_[k] - PL_["M5"]).dropna()
        c = CM[k].join(CM["M5"], rsuffix="_b", how="inner")
        res["models"][k] = {
            "features": SPECS[k], "ll": round(float(PL_[k].mean()), 5),
            "dll_vs_M5": P.block_boot(d.to_numpy(), blk.loc[d.index].to_numpy(), n=1000),
            "by_year_dll": d.groupby(d.index.str[:4]).mean().round(5).to_dict(),
            "combo_logscore": round(float(CM[k]["logscore"].mean()), 5),
            "combo_logscore_vs_M5": P.block_boot((c["logscore"] - c["logscore_b"]).to_numpy(), blk.loc[c.index].to_numpy(), n=1000),
            "hit": {f"N{n}": round(float((CM[k]["best_rank"] <= n).mean()), 4) for n in (1, 5, 10, 20, 30)},
            "hit_N10_vs_M5": P.block_boot(((c["best_rank"] <= 10).astype(float) - (c["best_rank_b"] <= 10).astype(float)).to_numpy(),
                                          blk.loc[c.index].to_numpy(), n=1000),
            "by_layer_dll": {lab: round(float(d[d.index.isin(ids)].mean()), 5) for lab, ids in layer_ids(tabs[F0.YEARS[-1]]).items()}}
        log(k, res["models"][k]["dll_vs_M5"]["mean"], res["models"][k]["dll_vs_M5"]["ci_bca"], "combo",
            res["models"][k]["combo_logscore"], "hit10", res["models"][k]["hit"]["N10"])
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def layer_ids(D: pd.DataFrame) -> dict:
    f = D.drop_duplicates("race_id").set_index("race_id")
    dm = f["is_debut_race"].astype(bool) | (f["class_ord"] == 0)
    n = D.groupby("race_id").size()
    return {"新馬・未勝利": set(f.index[dm]), "それ以外": set(f.index[~dm]),
            "新馬戦": set(f.index[f["is_debut_race"].astype(bool)]), "15頭以上": set(n.index[n >= 15])}


def combo_metrics(te: pd.DataFrame, pay: dict, lam) -> pd.DataFrame:
    rows = []
    for rid, g in te.groupby("race_id"):
        if rid not in pay:
            continue
        um = g["umaban_num"].to_numpy(int)
        wins = [s for s, _ in pay[rid] if s <= set(um)]
        if not wins:
            continue
        combos, p = P.triple_probs(g["_u"].to_numpy(float), "pl", tuple(lam))
        idx = {frozenset(um[x]): i for i, x in enumerate(combos)}
        wi = [idx[s] for s in wins]
        rank = np.empty(len(p), int)
        rank[np.argsort(-p)] = np.arange(1, len(p) + 1)
        rows.append({"race_id": rid, "logscore": float(np.log(np.mean(p[wi]))), "best_rank": int(min(rank[i] for i in wi))})
    return pd.DataFrame(rows).set_index("race_id")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.tune:
        res = tune()
        TUNE_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        log("tune", res["k_s"], res["k_d"], res["rho0"], res["rhoq"], "ll", round(res["ll_best"], 5), "1d", round(res["ll_1d"], 5))
    if a.eval:
        tr = json.loads(TUNE_OUT.read_text(encoding="utf-8"))
        par = tr["base_kalman"] | {"rho0": tr["rho0"], "rhoq": tr["rhoq"], "k_s": tr["k_s"], "k_d": tr["k_d"]}
        res = evaluate(par)
        EVAL_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
