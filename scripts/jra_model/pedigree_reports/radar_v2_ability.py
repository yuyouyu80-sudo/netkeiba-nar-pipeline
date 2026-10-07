# -*- coding: utf-8 -*-
"""着順予想の改善 Part 2 / G1: 本人の能力値(2026-10-08 新設、プラン wiggly-sniffing-metcalfe.md の Part 2)。

**開発はすべて 2020年までのデータ**(評価の行は 2018〜2020年、assert あり)。2021〜2026年は確認に使い回しが重なっているため使わない。
使う情報は 血統+本人の実績+斤量(オッズ・騎手・馬体重は使わない。市場は比較の物差しとしてだけ並べる)。

## 走ごとの能力値(fig、秒/km、大きいほど速い)
- 1kmあたりのタイム tpk = タイム(秒)/(距離/1000)。
- レースの水準 y_r = 上位3頭の tpk の中央値 を、コース(内外回り・京都の改修前後込み)+ クラス×芝ダ + 日(開催日×競馬場×芝ダ)+ γ_s·pace_hi に分解
  (交互最小二乗、日の効果は縮約 λ=1)。コース・クラス・γ は「Y−1年末まで」のレースで推定し、日の効果はその日のレースだけから作る(その日が終わった後に分かる量)。
- fig = −(tpk − コース − 日 − γ·pace_hi)。クラスは引かない(上のクラスで速く走るのは能力)。大敗はレースの勝ち馬 −2.0秒/km で頭打ち。
- pace_hi は radar_v2_stamina(2011〜2020年で固定した標準化)。

## 馬の状態(カルマンフィルタ、1次元)
- 初出走の前: 平均 = a + b·S(父の総合力、時点表)、分散 P0。
- 走と走の間: 平均 += 成長(年齢帯×月の区分ごとの平均の変化、2017年末まで)、分散 += q0 + q1·(間隔の日数/30)。
- 観測: fig = 状態 + 雑音(分散 R)。各走の「発走前」の値(その走で更新する前の平均・SD・走数)を特徴量にする。
- パラメータ(P0・q0・q1・R)は 2017年末までの走で、1走先の予測の対数尤度が最大になる格子から選ぶ。
- 対照: 過去の fig の指数加重平均(α)を事前の平均に k で縮めたもの。

使い方:
  python radar_v2_ability.py --build      # runs_ext.parquet(タイム・斤量、Git管理外)
  python radar_v2_ability.py --tune       # 2017年末までで格子探索 → radar_v2/ability_tune.json
  python radar_v2_ability.py --eval       # 2018〜2020年のローリング評価 → radar_v2/ability_eval_dev.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402

RUNS_EXT = C.OUT_DIR / "runs_ext.parquet"
TUNE_OUT = C.OUT_DIR / "ability_tune.json"
DEV_END = 2020
CAP = 2.0          # 大敗の頭打ち(秒/km)
log = A.log


# ------------------------------------------------------------------ 入力
def to_sec(t: pd.Series) -> pd.Series:
    m = t.fillna("").str.extract(r"^(?:(\d+):)?(\d+(?:\.\d+)?)$")
    return pd.to_numeric(m[0], errors="coerce").fillna(0) * 60 + pd.to_numeric(m[1], errors="coerce")


def build_ext() -> pd.DataFrame:
    frames = []
    for p in sorted(C.RESULTS_DIR.glob("20*/*.csv")):
        frames.append(pd.read_csv(p, dtype=str, usecols=["race_id", "horse_id", "time", "kinryo"]))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    d["time_sec"] = to_sec(d["time"])
    d["kinryo"] = pd.to_numeric(d["kinryo"], errors="coerce")
    d = d[["race_id", "horse_id", "time_sec", "kinryo"]]
    d.to_parquet(RUNS_EXT, index=False)
    return d


def load(max_year: int = DEV_END) -> pd.DataFrame:
    """runs_all(+京都の改修前後のキー・クラスのキー)に タイム・斤量・pace_hi を付ける。max_year より後の行は読まない。"""
    import radar_v2_stamina as S
    runs = S.load_base()
    runs = runs[runs["year"] <= max_year].copy()
    assert runs["year"].max() <= max_year
    ext = pd.read_parquet(RUNS_EXT) if RUNS_EXT.exists() else build_ext()
    runs = runs.merge(ext, on=["race_id", "horse_id"], how="left")
    sr = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")[["pace_hi"]]
    runs = runs.join(sr, on="race_id")
    runs["km"] = runs["distance"] / 1000.0
    runs["tpk"] = runs["time_sec"] / runs["km"]
    runs["day_key"] = runs["date"].dt.strftime("%Y%m%d") + "|" + runs["racecourse"] + "|" + runs["surface"]
    runs["cls_key"] = runs["surface"] + "|" + runs["class_key"].astype(str)
    return runs.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


# ------------------------------------------------------------------ 走ごとの能力値
def race_level(runs: pd.DataFrame) -> pd.DataFrame:
    top = runs[runs["pos"] <= 3]
    R = top.groupby("race_id").agg(y=("tpk", "median"), course=("course_key_io", "first"), cls=("cls_key", "first"),
                                   day=("day_key", "first"), surface=("surface", "first"), pace=("pace_hi", "first"),
                                   year=("year", "first"))
    return R.dropna(subset=["y"])


def fit_fig(runs: pd.DataFrame, end_year: int, n_iter: int = 15) -> dict:
    """コース・クラス・γ を end_year 年末までのレースで推定(日の効果は推定の途中で一緒に解く)。"""
    R = race_level(runs[runs["year"] <= end_year])
    R["pace"] = R["pace"].fillna(0)
    c = pd.Series(0.0, index=R["course"].unique())
    k = pd.Series(0.0, index=R["cls"].unique())
    gam = {"芝": 0.0, "ダ": 0.0}
    d = pd.Series(0.0, index=R["day"].unique())
    c = R.groupby("course")["y"].mean()
    for _ in range(n_iter):
        g = R["surface"].map(gam)
        r = R["y"] - R["course"].map(c) - R["cls"].map(k) - g * R["pace"]
        d = r.groupby(R["day"]).sum() / (r.groupby(R["day"]).count() + 1.0)
        r = R["y"] - R["day"].map(d) - R["cls"].map(k) - g * R["pace"]
        c = r.groupby(R["course"]).mean()
        r = R["y"] - R["day"].map(d) - R["course"].map(c) - g * R["pace"]
        k = r.groupby(R["cls"]).mean()
        k = k - k.mean()
        r = R["y"] - R["day"].map(d) - R["course"].map(c) - R["cls"].map(k)
        for s in gam:
            m = R["surface"] == s
            gam[s] = float(np.polyfit(R.loc[m, "pace"], r[m], 1)[0])
    resid = R["y"] - R["day"].map(d) - R["course"].map(c) - R["cls"].map(k) - R["surface"].map(gam) * R["pace"]
    return {"course": c.to_dict(), "cls": k.to_dict(), "gamma": gam, "end_year": end_year, "resid_sd": float(resid.std())}


def apply_fig(runs: pd.DataFrame, P: dict) -> pd.Series:
    """各走の fig。日の効果は、その日のレースだけから(コース・クラス・γ は P で固定)作る。"""
    R = race_level(runs)
    R["pace"] = R["pace"].fillna(0)
    c = R["course"].map(P["course"])
    # コースが P に無い(新しいコース): 芝ダ×距離の近いコースが無いので、その日の推定に含めず、その走の fig は NaN
    k = R["cls"].map(P["cls"]).fillna(0)
    g = R["surface"].map(P["gamma"]) * R["pace"]
    r = R["y"] - c - k - g
    d = r.groupby(R["day"]).sum() / (r.groupby(R["day"]).count() + 1.0)
    rr = runs[["race_id", "tpk", "pos", "course_key_io", "day_key", "surface", "pace_hi"]]
    base = rr["course_key_io"].map(P["course"]) + rr["day_key"].map(d) + rr["surface"].map(P["gamma"]) * rr["pace_hi"].fillna(0)
    fig = -(rr["tpk"] - base)
    win = fig.groupby(rr["race_id"]).transform("max")
    return fig.clip(lower=win - CAP)


# ------------------------------------------------------------------ 馬の状態
def growth_table(runs: pd.DataFrame, fig: pd.Series, end_year: int) -> dict:
    """連続する2走の fig の変化の平均(後の走の 年齢帯×半年 の区分ごと)。end_year 年末までの走。"""
    d = runs[["horse_id", "year", "age", "date"]].assign(fig=fig)
    d = d[d["year"] <= end_year]
    d["prev"] = d.groupby("horse_id")["fig"].shift(1)
    d["seg"] = growth_seg(d)
    g = (d["fig"] - d["prev"]).groupby(d["seg"]).mean()
    return {k: float(v) for k, v in g.items()}


def growth_seg(d: pd.DataFrame) -> pd.Series:
    age = d["age"].fillna(4).clip(2, 5).astype(int).astype(str)
    half = np.where(d["date"].dt.month <= 6, "前", "後")
    return age + half


def kalman(runs: pd.DataFrame, fig: pd.Series, prior_mu: pd.Series, par: dict, growth: dict) -> pd.DataFrame:
    """各走の発走前の状態(平均・SD・走数)と、1走先の予測の対数尤度(fig がある走)。runs は horse_id・date 順。"""
    h = runs["horse_id"].to_numpy()
    f = fig.to_numpy(float)
    pm = prior_mu.to_numpy(float)
    rest = runs["rest_days"].fillna(0).to_numpy(float)
    gs = growth_seg(runs).map(growth).fillna(0).to_numpy(float)
    P0, q0, q1, Rv = par["P0"], par["q0"], par["q1"], par["R"]
    n = len(h)
    mu_pre, sd_pre, n_pre, ll = np.empty(n), np.empty(n), np.zeros(n, int), np.full(n, np.nan)
    mu, P, cnt, last_h = 0.0, P0, 0, None
    for i in range(n):
        if h[i] != last_h:
            mu, P, cnt, last_h = pm[i], P0, 0, h[i]
        else:
            mu += gs[i]
            P += q0 + q1 * rest[i] / 30.0
        mu_pre[i], sd_pre[i], n_pre[i] = mu, np.sqrt(P), cnt
        if not np.isnan(f[i]):
            S_ = P + Rv
            ll[i] = -0.5 * (np.log(2 * np.pi * S_) + (f[i] - mu) ** 2 / S_)
            K = P / S_
            mu += K * (f[i] - mu)
            P *= (1 - K)
            cnt += 1
    return pd.DataFrame({"ab_mu": mu_pre, "ab_sd": sd_pre, "ab_n": n_pre, "ab_ll": ll}, index=runs.index)


def ewma(runs: pd.DataFrame, fig: pd.Series, prior_mu: pd.Series, alpha: float, k: float) -> pd.Series:
    """対照: 過去の fig の指数加重平均を、事前の平均に k(走数換算)で縮めた発走前の値。"""
    h = runs["horse_id"].to_numpy()
    f = fig.to_numpy(float)
    pm = prior_mu.to_numpy(float)
    out = np.empty(len(h))
    m, w, last_h = 0.0, 0.0, None
    for i in range(len(h)):
        if h[i] != last_h:
            m, w, last_h = 0.0, 0.0, h[i]
        out[i] = (w * m + k * pm[i]) / (w + k) if w > 0 else pm[i]
        if not np.isnan(f[i]):
            m = f[i] if w == 0 else alpha * f[i] + (1 - alpha) * m
            w = min(w + 1, 1 / alpha)
    return pd.Series(out, index=runs.index)


def prior_fit(runs: pd.DataFrame, fig: pd.Series, S: pd.Series, end_year: int) -> tuple[float, float]:
    """初出走の fig ~ a + b·S(end_year 年末までの初出走)。"""
    m = (runs["career"] == 0) & (runs["year"] <= end_year) & fig.notna() & S.notna()
    b, a = np.polyfit(S[m], fig[m], 1)
    return float(a), float(b)


def sire_S(runs: pd.DataFrame) -> pd.Series:
    """父の総合力 S(z、時点表 Y−1年末まで・本人除外)。2013年より前の走は NaN。"""
    import radar_v2_eval as E
    out = pd.Series(np.nan, index=runs.index)
    for Y in range(2013, int(runs["year"].max()) + 1):
        m = runs["year"] == Y
        out[m] = E.Tables(Y).z(runs[m], "perf_adj", "-")
    return out


# ------------------------------------------------------------------ 2017年末までの格子探索
# 1回目の格子 {P0 0.4/0.8/1.6, q0 0.01/0.03/0.08, q1 0/0.01/0.03, R 0.3/0.6/1.0} は端(0.4・0.01・0.01・0.3)が最良だったため広げた(2017年末まで)
GRID = {"P0": [0.2, 0.3, 0.4, 0.6], "q0": [0.0, 0.004, 0.01, 0.02], "q1": [0.003, 0.006, 0.01, 0.02], "R": [0.12, 0.18, 0.24, 0.3]}


def tune(runs: pd.DataFrame) -> dict:
    end = 2017
    r = runs[runs["year"] <= end]
    assert r["year"].max() <= end
    P = fit_fig(r, end)
    fig = apply_fig(r, P)
    S = sire_S(r)
    a, b = prior_fit(r, fig, S, end)
    pmu = a + b * S.fillna(0)
    gr = growth_table(r, fig, end)
    best, rows = None, []
    ev = r["year"].between(2015, 2017).to_numpy()   # 尤度を測る走(状態は2011年から積み上げる)
    import itertools
    for P0, q0, q1, Rv in itertools.product(*GRID.values()):
        par = {"P0": P0, "q0": q0, "q1": q1, "R": Rv}
        st = kalman(r, fig, pmu, par, gr)
        ll = float(np.nanmean(st["ab_ll"].to_numpy()[ev]))
        rows.append(par | {"ll": ll})
        if best is None or ll > best[1]:
            best = (par, ll)
    # 対照(指数加重平均)の α・k: 1走先の予測の平均二乗誤差が最小
    ew = []
    for alpha in (0.2, 0.35, 0.5, 0.7):
        for k in (0.5, 1, 2, 4):
            e = ewma(r, fig, pmu, alpha, k)
            m = ev & fig.notna().to_numpy() & (r["career"] > 0).to_numpy()
            ew.append({"alpha": alpha, "k": k, "mse": float(np.mean((fig.to_numpy()[m] - e.to_numpy()[m]) ** 2))})
    eb = min(ew, key=lambda x: x["mse"])
    out = {"end": end, "fig_resid_sd": P["resid_sd"], "gamma": P["gamma"], "prior": {"a": a, "b": b}, "growth": gr,
           "kalman": best[0], "kalman_ll": best[1], "grid": rows, "ewma": {"alpha": eb["alpha"], "k": eb["k"]}, "ewma_grid": ew,
           "fig_coverage": float(fig.notna().mean())}
    return out


# ------------------------------------------------------------------ 2018〜2020年のローリング評価(開発)
EVAL_YEARS = (2018, 2019, 2020)
TRAIN_START = 2014
DEV_OUT = C.OUT_DIR / "ability_eval_dev.json"


def ability_features(runs: pd.DataFrame, Y: int, tp: dict) -> pd.DataFrame:
    """Y年の評価用: 係数(コース・クラス・γ・事前・成長)は Y−1年末まで、状態のパラメータは tune(2017年末まで)。runs は Y年末まで。"""
    P = fit_fig(runs, Y - 1)
    fig = apply_fig(runs, P)
    S = sire_S(runs)
    a, b = prior_fit(runs, fig, S, Y - 1)
    pmu = a + b * S.fillna(0)
    gr = growth_table(runs, fig, Y - 1)
    st = kalman(runs, fig, pmu, tp["kalman"], gr)
    st["ew_mu"] = ewma(runs, fig, pmu, tp["ewma"]["alpha"], tp["ewma"]["k"])
    g = runs.groupby("horse_id")
    st["last_fig"] = fig.groupby(runs["horse_id"]).shift(1)
    st["kin"] = runs["kinryo"] - 55.0
    st["d_kin"] = (runs["kinryo"] - g["kinryo"].shift(1)).fillna(0)
    st["rest_long"] = (runs["rest_days"] > 90).astype(float)
    st["ab_new"] = (st["ab_n"] == 0).astype(float)
    st["fig"] = fig
    return pd.concat([runs[["race_id", "horse_id", "year"]], st], axis=1)


def build_eval_table(tp: dict) -> pd.DataFrame:
    import radar_v2_eval as E
    import radar_v2_stamina as S
    runs = load(DEV_END)
    dl = pd.read_parquet(C.OUT_DIR / "dl_table.parquet")
    dl = dl[dl["year"].between(TRAIN_START, DEV_END)]
    assert dl["year"].max() <= DEV_END
    base_runs = pd.read_parquet(A.RUNS)
    base_runs = A.add_tiers(base_runs[base_runs["year"] <= DEV_END], A.fixed_thresholds(base_runs))
    base_runs["umaban_num"] = pd.to_numeric(base_runs["umaban"], errors="coerce")
    assert base_runs["year"].max() <= DEV_END
    req = pd.read_parquet(C.OUT_DIR / "race_req.parquet")
    kh = E.estimate_kh(base_runs)
    R = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")
    flags = S.add_run_cols(runs[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan), R)
    flags = flags.set_index(["race_id", "horse_id"])[["ext", "short", "ext_amt"]]
    own = []
    for Y in range(TRAIN_START, DEV_END + 1):
        f = E.add_own(base_runs, E.features_for_year(base_runs, req, Y), Y, kh)
        own.append(f[["race_id", "horse_id", "S_own", "F_own"]])
        log("own", Y)
    own = pd.concat(own).set_index(["race_id", "horse_id"])
    D = dl.join(own, on=["race_id", "horse_id"]).join(flags, on=["race_id", "horse_id"])
    for c in ["ext", "short"]:
        D[c] = D[c].fillna(False).astype(float)
    D["ext_amt"] = D["ext_amt"].fillna(0.0)
    return runs, D


def evaluate(tp: dict) -> dict:
    import radar_v2_eval as E
    t0 = time.time()
    runs, D = build_eval_table(tp)
    AB = ["ab_mu", "ab_sd", "ab_new", "ew_mu", "last_fig", "kin", "d_kin", "rest_long"]
    specs = {"M0 父の総合力+v2の5成分(C2)": ["S", "F_front", "F_kire", "F_surf", "F_db", "F_deb"],
             "M1 血統+実績(R4 の (4))": ["S_own", "F_own"],
             "M2 M1+距離の変更": ["S_own", "F_own", "ext", "short", "ext_amt"],
             "M3 M2+能力値(カルマン)": ["S_own", "F_own", "ext", "short", "ext_amt", "ab_mu", "ab_sd", "ab_new"],
             "M3c M2+能力値(対照: 指数加重平均)": ["S_own", "F_own", "ext", "short", "ext_amt", "ew_mu", "ab_new"],
             "M4 M3+斤量・休み明け": ["S_own", "F_own", "ext", "short", "ext_amt", "ab_mu", "ab_sd", "ab_new", "kin", "d_kin", "rest_long"],
             "M5 M4+前走の能力値": ["S_own", "F_own", "ext", "short", "ext_amt", "ab_mu", "ab_sd", "ab_new", "kin", "d_kin", "rest_long", "last_fig"],
             "参考: 能力値だけ(カルマン)": ["ab_mu", "ab_sd", "ab_new"],
             "物差し: 市場(確定オッズ)": ["logp"],
             "物差し: 市場+M4": ["logp", "S_own", "F_own", "ext", "short", "ext_amt", "ab_mu", "ab_sd", "ab_new", "kin", "d_kin", "rest_long"]}
    per = {k: [] for k in specs}
    met = {k: [] for k in specs}
    uni = []
    for Y in EVAL_YEARS:
        r = runs[runs["year"] <= Y]
        F = ability_features(r, Y, tp).set_index(["race_id", "horse_id"])[AB + ["ab_n"]]
        Dy = D[D["year"] <= Y].join(F, on=["race_id", "horse_id"])
        tr, te = Dy[Dy["year"].between(TRAIN_START, Y - 1)], Dy[Dy["year"] == Y]
        ok = lambda X: X.groupby("race_id")["logp"].transform(lambda s: bool(np.isfinite(s).all()))  # noqa: E731
        trm, tem = tr[ok(tr)], te[ok(te)]
        keep = set(tem["race_id"])   # 全モデルで同じレース(確定オッズが全馬にあるレース)で比べる
        tr2, te2 = trm, tem
        for k, feats in specs.items():
            beta = E.PL(tr2, feats).fit()
            pl = E.PL(te2, feats)
            ll = pl.race_ll(beta)
            per[k].append(pd.Series(ll, index=pl.race_ids))
            u = np.where(pl.mask, pl.X @ beta, -np.inf)
            # 行は着順で並んでいる: 0番目が1着。1位予想の的中、予想上位3頭に入った1着、レース内の順位相関
            top1 = (np.argmax(u, axis=1) == 0)
            rank_u = np.argsort(np.argsort(-u, axis=1), axis=1)
            top3 = rank_u[:, 0] < 3
            rho = []
            for i in range(len(u)):
                m = pl.mask[i]
                x = rank_u[i, m]
                y = np.arange(m.sum())
                rho.append(np.corrcoef(x, y)[0, 1] if m.sum() > 2 else np.nan)
            met[k].append(pd.DataFrame({"top1": top1, "win_in_top3": top3, "rho": rho}, index=pl.race_ids))
        nn = pl.mask.sum(axis=1)
        uni.append(pd.Series(-(np.log(nn) + np.log(nn - 1) + np.log(nn - 2)), index=pl.race_ids))
        log("eval", Y, {k[:3]: round(float(per[k][-1].mean()), 4) for k in specs})
    P_ = {k: pd.concat(v) for k, v in per.items()}
    M_ = {k: pd.concat(v) for k, v in met.items()}
    U = pd.concat(uni)
    blk = D.drop_duplicates("race_id").set_index("race_id")["block"]
    res = {"eval_years": EVAL_YEARS, "train_start": TRAIN_START, "n_races": int(len(U)), "uniform_ll": round(float(U.mean()), 5),
           "tune": {"kalman": tp["kalman"], "ewma": tp["ewma"]}, "models": {}}
    base = "M1 血統+実績(R4 の (4))"
    mk = "物差し: 市場(確定オッズ)"
    for k in specs:
        d = P_[k] - P_[base]
        ci, p, se = E.block_boot(d.to_numpy(), blk.loc[d.index].to_numpy(), with_p=True)
        res["models"][k] = {"ll": round(float(P_[k].mean()), 5), "dll_vs_uniform": round(float((P_[k] - U).mean()), 5),
                            "dll_vs_M1": round(float(d.mean()), 5), "ci95_vs_M1": ci,
                            "by_year_vs_M1": d.groupby(d.index.str[:4]).mean().round(5).to_dict(),
                            "top1": round(float(M_[k]["top1"].mean()), 4), "win_in_top3": round(float(M_[k]["win_in_top3"].mean()), 4),
                            "rank_corr": round(float(M_[k]["rho"].mean()), 4),
                            "share_of_market_gain": round(float((P_[k] - U).mean() / (P_[mk] - U).mean()), 3)}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.eval:
        tp = json.loads(TUNE_OUT.read_text(encoding="utf-8"))
        res = evaluate(tp)
        DEV_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        for k, v in res["models"].items():
            log(k, v["ll"], v["dll_vs_M1"], v["ci95_vs_M1"], "top1", v["top1"], "市場比", v["share_of_market_gain"])
    if a.build:
        d = build_ext()
        log("runs_ext", len(d), "rows; time_sec NaN", round(float(d["time_sec"].isna().mean()), 4))
    if a.tune:
        t0 = time.time()
        runs = load(2017)
        res = tune(runs)
        res["elapsed_sec"] = round(time.time() - t0, 1)
        TUNE_OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
        log("tune", res["kalman"], round(res["kalman_ll"], 4), res["ewma"], "coverage", round(res["fig_coverage"], 3))


if __name__ == "__main__":
    main()
