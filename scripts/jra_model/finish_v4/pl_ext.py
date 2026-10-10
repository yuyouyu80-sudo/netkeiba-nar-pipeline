# -*- coding: utf-8 -*-
"""着順予想 finish_v4 の確率モデルの部品(2026-10-10、プラン wiggly-sniffing-metcalfe.md 第3版 F0-2)。

radar_v2_eval.PL・block_boot をここへ写して広げたもの。radar_v2_eval は import しない(import すると期間の既定値が
2021〜2026年になり、build_D が2021年以降を読むため。プランのエンジニアリングの約束)。

- PL(上位3頭を順に選ぶ Plackett–Luce)に、2着・3着の段階の縮め係数 λ2・λ3 を入れた版(Henery・Stern 型)。
  効用 u = X·β、k着目の段階は λ_k·u(λ1=1)。λ=1 で radar_v2_eval.PL と同じ尤度。
- 3頭の集合を直接選ぶ条件付きロジット: P(S) = exp(Σ_{i∈S} u_i) / e3(exp u)(e3 は3次の基本対称式)。
- 行の並び: 着順(完走のみ)→ 馬番。着順の無い馬(競走中止など)は最後に置き、どの段階でも選ばれない候補として分母に入る。
- 同着: 上位3着までに同じ着順の馬がいるとき、ありうる並び(PL)・集合(集合ロジット)の尤度の平均を取る。
- 組の確率: 各レースの全3頭組の確率(PL は6通りの順序の和、集合ロジットは定義どおり)。合計は1。
- 学習は torch の自動微分 + scipy L-BFGS(倍精度)。弱い L2 は β だけに掛ける。
- block_boot: 日付×競馬場のブロックの再抽出(パーセンタイルと BCa)。ratio_boot: 回収率のような比の推定量。
"""
import itertools

import numpy as np
import pandas as pd
import torch
from scipy.optimize import minimize

torch.set_default_dtype(torch.float64)


# ------------------------------------------------------------------ レースの詰め物テンソル
class RaceBatch:
    def __init__(self, df: pd.DataFrame, feats: list, pos_col: str = "pos", um_col: str = "umaban_num"):
        d = df.assign(_p=pd.to_numeric(df[pos_col], errors="coerce"))
        d = d.assign(_pk=d["_p"].fillna(1e9)).sort_values(["race_id", "_pk", um_col])
        n_fin = d.groupby("race_id")["_p"].transform("count")
        d = d[n_fin >= 3]
        self.race_ids, idx = np.unique(d["race_id"].to_numpy(), return_index=True)
        counts = np.diff(np.append(idx, len(d)))
        R, N = len(counts), int(counts.max())
        pos = np.arange(len(d)) - np.repeat(idx, counts)
        ri = np.repeat(np.arange(R), counts)
        X = np.zeros((R, N, len(feats)))
        X[ri, pos] = d[feats].fillna(0).to_numpy(float)
        M = np.zeros((R, N), bool)
        M[ri, pos] = True
        P = np.full((R, N), np.nan)
        P[ri, pos] = d["_p"].to_numpy(float)
        self.X, self.M, self.P, self.feats, self.N = X, M, P, feats, N
        self.horse_ids = np.full((R, N), None, dtype=object)
        if "horse_id" in d.columns:
            self.horse_ids[ri, pos] = d["horse_id"].to_numpy()
        self._variants()

    def _variants(self):
        """上位3着の並びの候補(同着のレースは複数)。orders[v] = 3頭の行番号、vrace[v] = レース番号。"""
        orders, vrace = [], []
        for r in range(len(self.race_ids)):
            p = self.P[r]
            fin = np.where(~np.isnan(p))[0]
            top = [i for i in fin if p[i] <= p[fin[2]]]   # 3着の着順以下の馬(同着で4頭以上になりうる)
            if len(top) == 3 and len(set(p[top])) == 3:
                orders.append(top)
                vrace.append(r)
                continue
            cands = set()
            for perm in itertools.permutations(top, 3):
                if all(p[perm[i]] <= p[perm[i + 1]] for i in range(2)) and \
                        all(sum(p[j] < p[perm[i]] for j in top) <= i for i in range(3)):
                    cands.add(perm)
            for c in sorted(cands):
                orders.append(list(c))
                vrace.append(r)
        self.orders = np.array(orders, int)
        self.vrace = np.array(vrace, int)
        self.n_var = np.bincount(self.vrace, minlength=len(self.race_ids))


def _stage_ll(u, M, orders, vrace, lam):
    """PL(λつき)の各並びの対数尤度。u: [R,N] torch。"""
    uu = u[vrace]
    mm = M[vrace]
    ll = torch.zeros(len(vrace))
    removed = torch.zeros_like(mm)
    ar = torch.arange(len(vrace))
    for k in range(3):
        s = (lam[k] * uu).masked_fill(~mm | removed, float("-inf"))
        ll = ll + s[ar, orders[:, k]] - torch.logsumexp(s, dim=1)
        removed = removed.clone()
        removed[ar, orders[:, k]] = True
    return ll


def _log_e3(u, M):
    """log e3(exp u)(3次の基本対称式)を、馬の列について漸化式で。"""
    R, N = u.shape
    # −∞ 同士の logaddexp は勾配が NaN になるため、有限の十分小さい値(exp すると0)を使う
    NEG = -1e4
    neg = torch.full((R,), NEG)
    e = [torch.zeros(R), neg.clone(), neg.clone(), neg.clone()]   # log e0..e3
    for j in range(N):
        w = u[:, j].masked_fill(~M[:, j], NEG)
        e3 = torch.logaddexp(e[3], w + e[2])
        e2 = torch.logaddexp(e[2], w + e[1])
        e1 = torch.logaddexp(e[1], w + e[0])
        e = [e[0], e1, e2, e3]
    return e[3]


def _set_ll(u, M, orders, vrace):
    uu, mm = u[vrace], M[vrace]
    ar = torch.arange(len(vrace))
    num = uu[ar, orders[:, 0]] + uu[ar, orders[:, 1]] + uu[ar, orders[:, 2]]
    return num - _log_e3(uu, mm)


def race_ll(batch: RaceBatch, beta, lam=(1.0, 1.0, 1.0), kind: str = "pl") -> np.ndarray:
    """レースごとの対数尤度(同着は並び・集合の平均)。"""
    with torch.no_grad():
        u = torch.tensor(batch.X) @ torch.tensor(np.asarray(beta, float))
        M = torch.tensor(batch.M)
        o, v = torch.tensor(batch.orders), torch.tensor(batch.vrace)
        if kind == "pl":
            lv = _stage_ll(u, M, o, v, torch.tensor(lam, dtype=torch.float64))
        else:
            # 集合ロジット: 同じ集合の並びは同じ値なので、集合ごとに1つ数える
            lv = _set_ll(u, M, o, v)
        return _avg_variants(lv, batch, kind).numpy()


def _avg_variants(lv, batch, kind):
    v = torch.tensor(batch.vrace)
    if kind == "set":
        keys = [tuple(sorted(o)) + (r,) for o, r in zip(batch.orders.tolist(), batch.vrace.tolist())]
        _, first = np.unique(keys, axis=0, return_index=True)
        keep = torch.zeros(len(keys), dtype=torch.bool)
        keep[torch.tensor(first)] = True
        lv, v = lv[keep], v[keep]
    R = len(batch.race_ids)
    mx = torch.full((R,), float("-inf")).scatter_reduce(0, v, lv, reduce="amax")
    s = torch.zeros(R).index_add(0, v, torch.exp(lv - mx[v]))
    cnt = torch.zeros(R).index_add(0, v, torch.ones_like(lv))
    return mx + torch.log(s) - torch.log(cnt)


def fit(batch: RaceBatch, kind: str = "pl", fit_lam: bool = False, l2: float = 0.0, beta0=None):
    """β(と λ2・λ3)を最尤で推定。戻り値 (beta, lam)。lam は kind='pl' のとき (1, λ2, λ3)。"""
    F = batch.X.shape[2]
    X, M = torch.tensor(batch.X), torch.tensor(batch.M)
    o, v = torch.tensor(batch.orders), torch.tensor(batch.vrace)
    n_par = F + (2 if (kind == "pl" and fit_lam) else 0)
    x0 = np.zeros(n_par)
    if beta0 is not None:
        x0[:F] = beta0

    def f(x):
        t = torch.tensor(x, requires_grad=True)
        beta = t[:F]
        u = X @ beta
        if kind == "pl":
            lam = torch.cat([torch.ones(1), torch.exp(t[F:F + 2])]) if fit_lam else torch.ones(3)
            lv = _stage_ll(u, M, o, v, lam)
        else:
            lv = _set_ll(u, M, o, v)
        ll = _avg_variants(lv, batch, kind)
        loss = -ll.sum() + l2 * (beta ** 2).sum() * len(batch.race_ids)
        loss.backward()
        return float(loss.item()), t.grad.numpy().copy()
    r = minimize(f, x0, jac=True, method="L-BFGS-B")
    beta = r.x[:F]
    lam = (1.0, float(np.exp(r.x[F])), float(np.exp(r.x[F + 1]))) if (kind == "pl" and fit_lam) else (1.0, 1.0, 1.0)
    return beta, lam


# ------------------------------------------------------------------ 組の確率
def triple_probs(u: np.ndarray, kind: str = "pl", lam=(1.0, 1.0, 1.0)):
    """1レース分の効用 u(長さ n)から、全3頭組 (i<j<k) とその確率(合計1)。"""
    n = len(u)
    combos = np.array(list(itertools.combinations(range(n), 3)), int)
    if kind == "set":
        s = u[combos].sum(axis=1)
        p = np.exp(s - s.max())
        return combos, p / p.sum()
    u = u - u.max()
    w1, w2, w3 = np.exp(lam[0] * u), np.exp(lam[1] * u), np.exp(lam[2] * u)
    S1, S2, S3 = w1.sum(), w2.sum(), w3.sum()
    p = np.zeros(len(combos))
    for perm in itertools.permutations(range(3)):
        a, b, c = combos[:, perm[0]], combos[:, perm[1]], combos[:, perm[2]]
        p += (w1[a] / S1) * (w2[b] / (S2 - w2[a])) * (w3[c] / (S3 - w3[a] - w3[b]))
    return combos, p


def top3_probs(combos: np.ndarray, p: np.ndarray, n: int) -> np.ndarray:
    """各馬の P(3着内)(組の確率の周辺和、合計3)。"""
    out = np.zeros(n)
    for c in range(3):
        np.add.at(out, combos[:, c], p)
    return out


# ------------------------------------------------------------------ ブートストラップ
def block_boot(delta: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 20261010, bca: bool = True) -> dict:
    """1レースあたりの平均の95%区間(ブロック再抽出)。パーセンタイルと BCa、片側 p(≤0 の割合、+1補正)。"""
    df = pd.DataFrame({"d": delta, "b": blocks})
    g = df.groupby("b")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    B = len(s)
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, B, B)
        out[i] = s[pick].sum() / c[pick].sum()
    est = s.sum() / c.sum()
    res = {"mean": float(est), "ci_pct": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
           "p_one_sided": float((np.sum(out <= 0) + 1) / (n + 1)), "se": float(np.std(out)), "n_blocks": int(B)}
    if bca:
        res["ci_bca"] = _bca(out, est, s, c)
    return res


def ratio_boot(num: np.ndarray, den: np.ndarray, blocks: np.ndarray, n: int = 2000, seed: int = 20261010) -> dict:
    """Σnum/Σden(回収率など)の区間。ブロック再抽出、BCa。"""
    df = pd.DataFrame({"a": num, "b": den, "k": blocks})
    g = df.groupby("k")[["a", "b"]].sum()
    a, b = g["a"].to_numpy(), g["b"].to_numpy()
    rng = np.random.default_rng(seed)
    B = len(a)
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, B, B)
        out[i] = a[pick].sum() / b[pick].sum()
    est = a.sum() / b.sum()
    return {"ratio": float(est), "ci_pct": [float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))],
            "ci_bca": _bca(out, est, a, b), "se": float(np.std(out)), "n_blocks": int(B)}


def _bca(boot, est, a, b):
    """BCa の95%区間。統計量はブロックの和の比 Σa/Σb(平均は b=件数)。加速度は1ブロック除外のジャックナイフ。"""
    from scipy.stats import norm
    z0 = norm.ppf(np.clip((boot < est).mean(), 1e-6, 1 - 1e-6))
    A, Bs = a.sum(), b.sum()
    jk = (A - a) / (Bs - b)
    m = jk.mean()
    acc = np.sum((m - jk) ** 3) / (6 * np.sum((m - jk) ** 2) ** 1.5 + 1e-300)
    out = []
    for q in (0.025, 0.975):
        z = norm.ppf(q)
        adj = norm.cdf(z0 + (z0 + z) / (1 - acc * (z0 + z)))
        out.append(float(np.percentile(boot, 100 * adj)))
    return out
