# -*- coding: utf-8 -*-
"""市場の中の別情報(T1 単勝×複勝、T2 単勝オッズの動き)の共通部品(2026-10-12)。事前登録: PREREG_market_pools_2026_10_12.md"""
import glob
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
OUT = ROOT / "data" / "jra_pipeline" / "market_pools_2026_10_12"
OUT.mkdir(parents=True, exist_ok=True)
# 単勝オッズ推移(予想パイプラインの odds_history_{date}.csv)の置き場所。環境変数で上書き可
# 2026-10-12 レビュー対応: セッション固有の一時フォルダから repo 内へ複製(gzip、ハッシュは inputs/odds_tansho_sha256.txt)
TANSHO_HIST_DIR = Path(os.environ.get("MP_TANSHO_HIST_DIR", str(OUT / "inputs" / "odds_tansho")))
LAMBDA_JSON = OUT / "lambda_2017_2025.json"
# 2026-10-12 2回目レビュー対応: 対象期間の終わりを固定(再実行で標本が増えないように)
END_DATE = "20261012"


def in_period(date):
    return str(date) <= END_DATE


def tansho_hist_path(date):
    for ext in (".csv.gz", ".csv"):
        p = TANSHO_HIST_DIR / f"odds_history_{date}{ext}"
        if p.exists():
            return p
    return None


def tansho_hist_dates():
    return sorted({p.name.split("_")[-1].split(".")[0] for p in TANSHO_HIST_DIR.glob("odds_history_2026*")})


def _results_one(date):
    """1開催日の着順・確定単勝(race_results、無ければ intraday_results)。障害戦は除外。"""
    p = ROOT / "data" / "race_results" / date[:4] / f"{date}.csv"
    src = "rr"
    if not p.exists():
        p = ROOT / "data" / "intraday_results" / f"{date}.csv"
        src = "intraday"
        if not p.exists():
            return None
    d = pd.read_csv(p, dtype=str)
    d = d[~d["surface"].astype(str).str.contains("障")]
    if src == "intraday":
        # 2026-10-12 2回目レビュー対応: 速報は1〜5着だけの部分取得があるため、馬柱の頭数より行が少ないレースを除く
        n = d.groupby("race_id")["umaban"].transform("size")
        need = d["race_id"].map(lambda r: _newspaper_n(r))
        d = d[need.isna() | (n >= need)]
    out = pd.DataFrame({"race_id": d["race_id"], "date": date, "venue": d["race_id"].str[4:6],
                        "um": pd.to_numeric(d["umaban"], errors="coerce"),
                        "fin": pd.to_numeric(d["finish_pos"], errors="coerce"),
                        "odds": pd.to_numeric(d["odds_final"], errors="coerce"), "src": src})
    return out.dropna(subset=["um"]).drop_duplicates(["race_id", "um"], keep="last")


def _newspaper_n(rid):
    p = ROOT / "data" / "newspaper" / f"{rid}.csv"
    if not p.exists():
        return np.nan
    return len(pd.read_csv(p, dtype=str, usecols=["umaban"]).drop_duplicates())


def results(dates):
    fr = [r for r in (_results_one(d) for d in dates) if r is not None]
    return pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()


def norm_p(odds):
    inv = np.where(odds > 0, 1.0 / odds, np.nan)
    return inv / np.nansum(inv)


# ---------------------------------------------------------------- Benter 型の縮めた PL
def top3_prob(p, l2, l3):
    """各馬の3着内確率(1着 ∝ p、2着 ∝ p^l2、3着 ∝ p^l3)。p は合計1の配列。"""
    n = len(p)
    a2, a3 = p ** l2, p ** l3
    P = p.copy()  # 1着
    S2 = a2.sum()
    # 2着: sum_i p_i * a2_j / (S2 - a2_i), i != j
    w2 = p / (S2 - a2)  # i ごと
    P2 = a2 * (w2.sum() - w2)
    # 3着: sum_{i!=j!=k} p_i * a2_j/(S2-a2_i) * a3_k/(S3-a3_i-a3_j)
    S3 = a3.sum()
    pij = p[:, None] * a2[None, :] / (S2 - a2)[:, None]
    np.fill_diagonal(pij, 0.0)
    den = S3 - a3[:, None] - a3[None, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        g = np.where(den > 0, pij / den, 0.0)
    np.fill_diagonal(g, 0.0)
    tot = g.sum()
    # k を除く: 行 k・列 k の寄与を引く(対角は0)
    P3 = a3 * (tot - g.sum(axis=0) - g.sum(axis=1))
    return np.clip(P + P2 + P3, 1e-6, 1 - 1e-6) if n > 3 else np.full(n, 1 - 1e-6)


def fit_lambda(years=range(2017, 2026)):
    """2017〜2025年の確定オッズ・着順で λ2・λ3 を最尤推定(1着を条件に2着・3着の条件付き尤度)。"""
    if LAMBDA_JSON.exists():
        cached = json.loads(LAMBDA_JSON.read_text(encoding="utf-8"))
        if cached.get("years") == f"{min(years)}-{max(years)}":
            return cached
    recs = []
    for y in years:
        for f in sorted(glob.glob(str(ROOT / "data" / "race_results" / str(y) / f"{y}*.csv"))):
            d = pd.read_csv(f, dtype=str, usecols=["race_id", "surface", "umaban", "finish_pos", "odds_final"])
            d = d[~d["surface"].astype(str).str.contains("障")]
            d["fin"] = pd.to_numeric(d["finish_pos"], errors="coerce")
            d["odds"] = pd.to_numeric(d["odds_final"], errors="coerce")
            for rid, g in d.groupby("race_id"):
                g = g[g["odds"] > 0]
                if len(g) < 6 or not {1, 2, 3} <= set(g["fin"].dropna().astype(int)):
                    continue
                if (g["fin"] == 1).sum() != 1 or (g["fin"] == 2).sum() != 1:
                    continue  # 同着は除く
                lp = np.log(norm_p(g["odds"].to_numpy()))
                f = g["fin"].to_numpy()
                recs.append((lp, int(np.argmax(f == 1)), int(np.argmax(f == 2)), int(np.argmax(f == 3))))

    R, N = len(recs), max(len(r[0]) for r in recs)
    LP = np.full((R, N), np.nan)
    for k, (lp, *_r) in enumerate(recs):
        LP[k, :len(lp)] = lp
    I1, I2, I3 = (np.array([r[j] for r in recs]) for j in (1, 2, 3))
    ar = np.arange(R)
    m2 = ~np.isnan(LP)
    m2[ar, I1] = False
    m3 = m2.copy()
    m3[ar, I2] = False
    LP0 = np.nan_to_num(LP, nan=0.0)

    def lse(z, m):
        z = np.where(m, z, -np.inf)
        mx = z.max(axis=1, keepdims=True)
        return (mx[:, 0] + np.log(np.exp(z - mx).sum(axis=1)))

    def nll(th):
        l2, l3 = th
        s2 = l2 * LP0[ar, I2] - lse(l2 * LP0, m2)
        s3 = l3 * LP0[ar, I3] - lse(l3 * LP0, m3)
        return -(s2 + s3).mean()

    r = minimize(nll, x0=[0.8, 0.7], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-7})
    res = {"lambda2": float(r.x[0]), "lambda3": float(r.x[1]), "n_races": len(recs), "nll": float(r.fun),
           "years": f"{min(years)}-{max(years)}"}
    LAMBDA_JSON.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    return res


# ---------------------------------------------------------------- 評価
def logistic_ll(X, y, w):
    z = X @ w
    return y * z - np.logaddexp(0, z)


def fit_logistic(X, y, lam=1.0):
    """切片と先頭の特徴(市場)は罰則なし、残りに L2(条件付きロジットとそろえる。2026-10-12 レビュー対応)。"""
    pen = np.r_[0.0, 0.0, np.ones(X.shape[1] - 2)] * lam

    def f(w):
        z = X @ w
        v = -(y * z - np.logaddexp(0, z)).sum() + 0.5 * (pen * w * w).sum()
        g = -X.T @ (y - 1 / (1 + np.exp(-z))) + pen * w
        return v, g
    return minimize(f, np.zeros(X.shape[1]), jac=True, method="L-BFGS-B").x


def _codes(gid):
    return np.unique(gid, return_inverse=True)[1]


def clogit_ll(X, gid, y, w):
    """条件付きロジット: レース(gid)ごとの勝ち馬の対数確率を行に置く(勝ち馬以外は0)。"""
    c = _codes(gid)
    u = X @ w
    mx = np.full(c.max() + 1, -np.inf)
    np.maximum.at(mx, c, u)
    lse = np.log(np.bincount(c, np.exp(u - mx[c]))) + mx
    return np.where(y == 1, u - lse[c], 0.0)


def fit_clogit(X, gid, y, lam=1.0):
    """先頭列(市場)は罰則なし、残りに L2。"""
    c = _codes(gid)
    pen = np.r_[0.0, np.ones(X.shape[1] - 1)] * lam

    def f(w):
        u = X @ w
        mx = np.full(c.max() + 1, -np.inf)
        np.maximum.at(mx, c, u)
        e = np.exp(u - mx[c])
        Z = np.bincount(c, e)
        pr = e / Z[c]
        v = -(y * (u - (np.log(Z) + mx)[c])).sum() + 0.5 * (pen * w * w).sum()
        yr = np.bincount(c, y)  # レースごとの勝ち馬数
        g = -(X.T @ (y - pr * yr[c])) + pen * w
        return v, g
    return minimize(f, np.r_[1.0, np.zeros(X.shape[1] - 1)], jac=True, method="L-BFGS-B").x


def lodo_dll(df, base_cols, new_cols, ycol, kind, lam=1.0):
    """開催日を1日ずつ抜く交差検証。行ごとの ΔLL(新特徴あり − なし)を返す。"""
    dll = np.zeros(len(df))
    for d in df["date"].unique():
        tr, te = (df["date"] != d).to_numpy(), (df["date"] == d).to_numpy()
        lls = []
        for cols in (base_cols, base_cols + new_cols):
            if kind == "logit":
                X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in cols])
                w = fit_logistic(X[tr], df[ycol].to_numpy()[tr], lam)
                lls.append(logistic_ll(X[te], df[ycol].to_numpy()[te], w))
            else:
                # 条件付きロジットは切片不要。先頭(市場)は罰則なし、新特徴だけに L2
                X = np.column_stack([df[c].to_numpy() for c in cols])
                w = fit_clogit(X[tr], df["race_id"].to_numpy()[tr], df[ycol].to_numpy()[tr], lam)
                lls.append(clogit_ll(X[te], df["race_id"].to_numpy()[te], df[ycol].to_numpy()[te], w))
        dll[te] = lls[1] - lls[0]
    return dll


def block_ci(df, dll, n_boot=4000, seed=0):
    """開催日×競馬場ブロックのブートストラップ。合計 ΔLL の95%区間。"""
    blk = (df["date"] + "_" + df["venue"]).to_numpy()
    s = pd.Series(dll).groupby(blk).sum()
    rng = np.random.default_rng(seed)
    v = s.to_numpy()
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].sum(axis=1)
    return float(v.sum()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)), len(v)


def placebo(df, base_cols, new_cols, ycol, kind, n=20, seed=1):
    """新特徴の列をまとめて(同じ並べ替えで)レース内シャッフルし、LODO の ΔLL 合計を n 回求める。
    2026-10-12 レビュー対応: 複数列を別々に並べ替えると評価コード01と03が両方1の馬ができたため、行ごとに並べ替える。"""
    rng = np.random.default_rng(seed)
    gid = df["race_id"].to_numpy()
    order = np.argsort(gid, kind="stable")
    starts = np.flatnonzero(np.r_[True, gid[order][1:] != gid[order][:-1]])
    ends = np.r_[starts[1:], len(order)]
    out = []
    for _ in range(n):
        src = np.empty(len(df), dtype=int)
        for a, b in zip(starts, ends):
            src[order[a:b]] = order[a:b][rng.permutation(b - a)]
        d2 = df.copy()
        for c in new_cols:
            d2[c] = df[c].to_numpy()[src]
        out.append(float(lodo_dll(d2, base_cols, new_cols, ycol, kind).sum()))
    return out


def coef_boot(df, base_cols, new_cols, ycol, kind, n_boot=1000, seed=2):
    """係数のブートストラップ。復元抽出して全体を当て直す。
    2026-10-12 2回目レビュー対応: ΔLL の区間は係数を固定した区間で、効果の大きさの範囲にはならないため追加。
    Fable 5.1 レビュー対応: 単位は2通り(開催日×競馬場ブロック、レース)。ブロックは30前後しかなく区間が狭く出るため、
    レース単位も併記する。回数は300→1000。
    戻り値: {列: [ブロック下限, ブロック上限, 特徴のSD, レース下限, レース上限]}"""
    df = df.reset_index(drop=True)
    cols = base_cols + new_cols
    Xall = np.column_stack([df[c].to_numpy() for c in cols])
    y = df[ycol].to_numpy()
    rid = df["race_id"].to_numpy()
    rng = np.random.default_rng(seed)
    res = {}
    for unit in ("block", "race"):
        key = (df["date"] + "_" + df["venue"]).to_numpy() if unit == "block" else rid
        ub, inv = np.unique(key, return_inverse=True)
        order = np.argsort(inv, kind="stable")
        starts = np.searchsorted(inv[order], np.arange(len(ub)))
        ends = np.r_[starts[1:], len(order)]
        rows = [order[a:b] for a, b in zip(starts, ends)]
        out = []
        for _ in range(n_boot):
            pick = rng.integers(0, len(ub), len(ub))
            idx = np.concatenate([rows[k] for k in pick])
            if kind == "logit":
                X = np.column_stack([np.ones(len(idx)), Xall[idx]])
                w = fit_logistic(X, y[idx])[1:]
            else:
                # 同じブロック・レースを2回引いたときに別レースとして扱う
                tag = np.concatenate([np.full(len(rows[k]), j) for j, k in enumerate(pick)])
                g = pd.factorize(pd.Series(rid[idx]).astype(str) + "_" + pd.Series(tag).astype(str))[0]
                w = fit_clogit(Xall[idx], g, y[idx])
            out.append(w[len(base_cols):])
        res[unit] = np.array(out)
    return {c: [float(np.percentile(res["block"][:, j], 2.5)), float(np.percentile(res["block"][:, j], 97.5)), float(df[c].std()),
                float(np.percentile(res["race"][:, j], 2.5)), float(np.percentile(res["race"][:, j], 97.5))]
            for j, c in enumerate(new_cols)}


def evaluate(df, base_cols, new_cols, ycol, kind, name, n_placebo=20):
    """合否は事前登録どおり(区間下限>0 かつ プラセボ先頭20回の最大超え)。
    2026-10-12 レビュー対応: n_placebo を増やせるようにし、プラセボ分布での順位 p 値(perm_p)を参考として追加。
    2回目レビュー対応: hi_per_race(区間上限/レース数)は上限として意味が無いので削除。係数の区間 coef_ci([下限, 上限, 特徴のSD])を追加。
    罰則: 条件付きロジットは先頭の市場列だけ、ロジスティックは切片と先頭の市場列だけが罰則なし(評価コード等の基準側の列は L2 で縮む)。"""
    df = df.reset_index(drop=True)
    dll = lodo_dll(df, base_cols, new_cols, ycol, kind)
    tot, lo, hi, nb = block_ci(df, dll)
    pl = placebo(df, base_cols, new_cols, ycol, kind, n=n_placebo)
    # 全期間フィットの係数(説明用)
    if kind == "logit":
        X = np.column_stack([np.ones(len(df))] + [df[c].to_numpy() for c in base_cols + new_cols])
        w = fit_logistic(X, df[ycol].to_numpy())[1:]
    else:
        X = np.column_stack([df[c].to_numpy() for c in base_cols + new_cols])
        w = fit_clogit(X, df["race_id"].to_numpy(), df[ycol].to_numpy())
    res = {"name": name, "n_races": int(df["race_id"].nunique()), "n_rows": len(df), "n_blocks": nb,
           "n_days": int(df["date"].nunique()), "dll": tot, "lo": lo, "hi": hi,
           "placebo_max": max(pl[:20]), "placebo_mean": float(np.mean(pl)), "n_placebo": len(pl),
           "placebo_max_all": max(pl), "perm_p": float((1 + sum(v >= tot for v in pl)) / (len(pl) + 1)),
           "placebo_values": pl, "dll_minus_placebo_mean": float(tot - np.mean(pl)),
           "coef": dict(zip(base_cols + new_cols, map(float, w))),
           "coef_ci": coef_boot(df, base_cols, new_cols, ycol, kind),
           "pass": bool(lo > 0 and tot > max(pl[:20]))}
    return res
