# -*- coding: utf-8 -*-
"""血統レーダー v2 の R4: 2021〜2026年での確認(事前登録 RADAR_V2_PREREG_2026_10_06.md §6、追補1)(2026-10-06)。

**2021〜2026年は確認に1回だけ使う**(この評価を回し直して定義を変えたら、それは探索として扱う)。
係数 β は2018〜2020年のレースで推定し、2021〜2026年では固定する。

モデル(上位3頭を順に選ぶ Plackett–Luce 型の条件付きロジット):
  A0: u = β1·S                         S = 父の総合力(産駒の perf_adj の縮約平均、父→父父→母父、本人の走を除く)/τ
  A1: u = β1·S + β2·F                  F = Σ_a r_a·req_a·z_a(採用軸: 脚質・キレ・芝ダ・距離帯・新馬)
      r_a = 内側検証の父のE1(SB、区分ごと、0未満は0)。req = 脚質・キレはレースの基準/SD(2018〜2020)、条件の軸は1、新馬は新馬戦のとき1。
      z_a = 血統の値/τ(時点表 Y−1年末まで、本人の走を除く)。
主仮説: 2021〜2026年の 1レースあたり ΔLL = LL(A1) − LL(A0) の95%CI下限>0(日付×競馬場のブロックブートストラップ)。
出力: radar_v2/radar_v2_eval.json
"""
import glob
import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402

FIT = (2018, 2020)
TEST = (2021, 2026)
# --dry: 2021年以降を読まずに配線を確かめる試運転(フィット2015〜2017、テスト2018〜2020)。本番(引数なし)は確認に1回だけ。
DRY = "--dry" in sys.argv
if DRY:
    FIT, TEST = (2015, 2017), (2018, 2020)
SEED = 20261006
N_BOOT = 2000
N_PLACEBO = 20
OUT = C.OUT_DIR / ("radar_v2_eval_dry.txt" if DRY else "radar_v2_eval.json")
INNER = json.loads((C.OUT_DIR / "r2_inner_validation.json").read_text(encoding="utf-8"))
REQ_META = json.loads((C.OUT_DIR / "race_req_meta.json").read_text(encoding="utf-8"))
BUCKETS = ["短距離(~1400m)", "マイル(1401-1800m)", "中距離(1801-2200m)", "長距離(2201m~)"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def rel(axis, level="-"):
    v = INNER["axes"][axis]["levels"].get(level, {}).get("E1_sire_sb")
    return max(float(v), 0.0) if v is not None else 0.0


# ------------------------------------------------------------------ 血統の値(本人の走を除く)
class Tables:
    def __init__(self, Y):
        anc = pd.read_parquet(A.TABLE_DIR / f"anc_{Y}.parquet")
        self.anc = {k: g.set_index("ancestor_id") for k, g in anc.groupby(["axis", "level", "role"])}
        hv = pd.read_parquet(A.TABLE_DIR / f"hv_{Y}.parquet")
        self.hv = {k: g.set_index("horse_id")[["xc", "w"]] for k, g in hv.groupby(["axis", "level"])}

    def tau(self, axis, level="-"):
        t = self.anc.get((axis, level, "sire"))
        return float(t["tau"].iloc[0]) if t is not None else np.nan

    def value(self, df, axis, level):
        """df の各行(馬)について、本人の走を除いた縮約値(系統の合わせ方は事前登録 §3)。"""
        cols = {}
        h = self.hv.get((axis, level))
        for role, rc in C.ROLES:
            t = self.anc.get((axis, level, role))
            if t is None:
                cols[role] = pd.Series(np.nan, index=df.index)
                continue
            j = df[[rc, "horse_id"]].join(t[["mean", "sw", "k"]], on=rc)
            if h is not None:
                j = j.join(h, on="horse_id")
                own = j["xc"].notna()
                num = j["sw"] * j["mean"] - np.where(own, j["w"] * j["xc"], 0)
                den = j["sw"] - np.where(own, j["w"], 0) + j["k"]
                cols[role] = num / den
            else:
                cols[role] = j["sw"] * j["mean"] / (j["sw"] + j["k"])
        if axis in A.SIRE_FIRST_AXES:
            return cols["sire"].fillna(cols["ss"]).fillna(cols["bms"])
        return pd.concat(cols.values(), axis=1).mean(axis=1)

    def z(self, df, axis, level):
        return self.value(df, axis, level) / self.tau(axis, level)


# ------------------------------------------------------------------ v1(時点表版、本人の走を除く)
def v1_values(win: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """v1 の定義(勝率差 pt、セル下限N=10、3系統の平均)を Y−1年末までの走で。本人の走は件数から引く。"""
    out = pd.DataFrame(index=df.index)
    specs = {"front": ("leader", True), "kire_lead": ("leader", True), "kire_non": ("leader", False)}
    vals = {k: [] for k in ["front", "kire", "debut"] + [f"surface|{s}" for s in ["芝", "ダ"]] + [f"dbucket|{b}" for b in BUCKETS]}
    win = win.assign(lead=(win["leader"] == True).astype(float), non=(win["leader"] == False).astype(float))  # noqa: E712
    win = win.assign(lw=win["lead"] * win["is_win"], nw=win["non"] * win["is_win"], dw=win["is_debut_race"] * win["is_win"],
                     dn=win["is_debut_race"].astype(float))
    for s in ["芝", "ダ"]:
        win[f"s_{s}_n"] = (win["surface"] == s).astype(float)
        win[f"s_{s}_w"] = win[f"s_{s}_n"] * win["is_win"]
    for i, b in enumerate(BUCKETS):
        win[f"b_{i}_n"] = (win["dbucket"] == b).astype(float)
        win[f"b_{i}_w"] = win[f"b_{i}_n"] * win["is_win"]
    cnt_cols = ["is_win", "lead", "lw", "non", "nw", "dn", "dw"] + [f"s_{s}_{x}" for s in ["芝", "ダ"] for x in "nw"] + \
        [f"b_{i}_{x}" for i in range(4) for x in "nw"]
    win = win.assign(one=1.0)
    own = win.groupby("horse_id")[["one"] + cnt_cols].sum()
    for role, rc in C.ROLES:
        a = win.dropna(subset=[rc]).groupby(rc)[["one"] + cnt_cols].sum()
        j = df[[rc, "horse_id"]].join(a, on=rc).join(own, on="horse_id", rsuffix="_own")
        for c in ["one"] + cnt_cols:
            j[c] = j[c] - j[f"{c}_own"].fillna(0)
        ov = j["is_win"] / j["one"]
        lead_wr, non_wr = j["lw"] / j["lead"], j["nw"] / j["non"]
        vals["front"].append(np.where(j["lead"] >= 10, (lead_wr - ov) * 100, np.nan))
        vals["kire"].append(np.where((j["lead"] >= 10) & (j["non"] >= 10), (non_wr - lead_wr) * 100, np.nan))
        vals["debut"].append(np.where(j["dn"] >= 10, (j["dw"] / j["dn"] - ov) * 100, np.nan))
        for s in ["芝", "ダ"]:
            vals[f"surface|{s}"].append(np.where(j[f"s_{s}_n"] >= 10, (j[f"s_{s}_w"] / j[f"s_{s}_n"] - ov) * 100, np.nan))
        for i, b in enumerate(BUCKETS):
            vals[f"dbucket|{b}"].append(np.where(j[f"b_{i}_n"] >= 10, (j[f"b_{i}_w"] / j[f"b_{i}_n"] - ov) * 100, np.nan))
    for k, lst in vals.items():
        m = np.nanmean(np.vstack(lst), axis=0)
        out[k] = m / np.nanstd(m)   # 標準化(このテスト年の出走馬の間のSD)
    return out


# ------------------------------------------------------------------ 特徴量(年ごと)
def features_for_year(runs: pd.DataFrame, req: pd.DataFrame, Y: int) -> pd.DataFrame:
    T = Tables(Y)
    df = runs[runs["year"] == Y].copy()
    df = df.join(req[["req_front", "req_kire"]], on="race_id")
    sd = {a: REQ_META[f"sd_req_{a}_2018_2020"] for a in ["front", "kire"]}
    df["S"] = T.z(df, "perf_adj", "-")
    df["z_front"] = T.z(df, "front", "-")
    df["z_kire"] = T.z(df, "kire", "-")
    df["z_debut"] = T.z(df, "debut", "-")
    df["z_surface"] = np.nan
    df["r_surface"] = 0.0
    for s in ["芝", "ダ"]:
        m = df["surface"] == s
        df.loc[m, "z_surface"] = T.z(df[m], "surface", s)
        df.loc[m, "r_surface"] = rel("surface", s)
    for b in BUCKETS:   # 全距離帯の値(プラセボで別のレースの距離帯を当てるため)
        df[f"z_b|{b}"] = T.z(df, "dbucket", b)
    df["z_dbucket"] = np.nan
    df["r_dbucket"] = 0.0
    for b in BUCKETS:
        m = df["dbucket"] == b
        df.loc[m, "z_dbucket"] = df.loc[m, f"z_b|{b}"]
        df.loc[m, "r_dbucket"] = rel("dbucket", b)
    df["q_front"] = df["req_front"] / sd["front"]
    df["q_kire"] = df["req_kire"] / sd["kire"]
    df["F"] = fit_score(df)
    # v1(同じ形の適合度、同じ r と req)
    win = runs[runs["year"] <= Y - 1]
    v1 = v1_values(win, df)
    df["F_v1"] = (rel("front") * df["q_front"] * v1["front"].fillna(0) + rel("kire") * df["q_kire"] * v1["kire"].fillna(0)
                  + rel("debut") * df["is_debut_race"] * v1["debut"].fillna(0))
    for s in ["芝", "ダ"]:
        m = df["surface"] == s
        df.loc[m, "F_v1"] += rel("surface", s) * v1.loc[m, f"surface|{s}"].fillna(0)
    for b in BUCKETS:
        m = df["dbucket"] == b
        df.loc[m, "F_v1"] += rel("dbucket", b) * v1.loc[m, f"dbucket|{b}"].fillna(0)
    return df


def fit_score(df, q_front=None, q_kire=None, z_dbucket=None, r_dbucket=None, debut=None):
    q_front = df["q_front"] if q_front is None else q_front
    q_kire = df["q_kire"] if q_kire is None else q_kire
    z_dbucket = df["z_dbucket"] if z_dbucket is None else z_dbucket
    r_dbucket = df["r_dbucket"] if r_dbucket is None else r_dbucket
    debut = df["is_debut_race"] if debut is None else debut
    return (rel("front") * q_front.fillna(0) * df["z_front"].fillna(0)
            + rel("kire") * q_kire.fillna(0) * df["z_kire"].fillna(0)
            + df["r_surface"] * df["z_surface"].fillna(0)
            + r_dbucket * z_dbucket.fillna(0)
            + rel("debut") * debut.astype(float) * df["z_debut"].fillna(0))


# ------------------------------------------------------------------ 実績(本人の過去走)
def add_own(runs: pd.DataFrame, df: pd.DataFrame, Y: int, kh: dict) -> pd.DataFrame:
    """その日より前の本人の走(係数は Y−1年末まで)から、総合力・脚質・キレ・芝ダ・距離帯の値を作り、血統と合成。"""
    fit = runs[runs["year"] <= Y - 1]
    allv = A.run_values(fit, runs[runs["year"] <= Y]).sort_values(["horse_id", "date"])
    g = allv.groupby("horse_id")
    for col in ["perf_adj", "front", "kire"]:
        x = allv[col].fillna(0)
        n = allv[col].notna().astype(float)
        allv[f"{col}_ps"] = x.groupby(allv["horse_id"]).cumsum() - x
        allv[f"{col}_pn"] = n.groupby(allv["horse_id"]).cumsum() - n
    for key, col, levels in [("surface", "surface", ["芝", "ダ"]), ("dbucket", "dbucket", BUCKETS)]:
        for lev in levels:
            m = (allv[col] == lev).astype(float)
            x = allv["perf_adj"].fillna(0) * m
            allv[f"{key}|{lev}_ps"] = x.groupby(allv["horse_id"]).cumsum() - x
            allv[f"{key}|{lev}_pn"] = m.groupby(allv["horse_id"]).cumsum() - m
    allv = allv[allv["year"] == Y].set_index(["race_id", "horse_id"])
    df = df.join(allv[[c for c in allv.columns if c.endswith("_ps") or c.endswith("_pn")]], on=["race_id", "horse_id"])
    T = Tables(Y)

    def combine(z_ped, own_val, n_info, axis_tau, k):
        zo = (own_val / axis_tau).where(n_info > 0, 0)
        return (n_info * zo + k * z_ped.fillna(0)) / (n_info + k)
    tau = {a: T.tau(a) for a in ["perf_adj", "front", "kire"]}
    df["S_own"] = combine(df["S"], df["perf_adj_ps"] / df["perf_adj_pn"].replace(0, np.nan), df["perf_adj_pn"], tau["perf_adj"], kh["perf_adj"])
    df["zo_front"] = combine(df["z_front"], df["front_ps"] / df["front_pn"].replace(0, np.nan), df["front_pn"], tau["front"], kh["front"])
    df["zo_kire"] = combine(df["z_kire"], df["kire_ps"] / df["kire_pn"].replace(0, np.nan), df["kire_pn"], tau["kire"], kh["kire"])
    for key, col, levels in [("surface", "surface", ["芝", "ダ"]), ("dbucket", "dbucket", BUCKETS)]:
        df[f"zo_{key}"] = np.nan
        for lev in levels:
            m = df[col] == lev
            n_l, s_l = df.loc[m, f"{key}|{lev}_pn"], df.loc[m, f"{key}|{lev}_ps"]
            n_all, s_all = df.loc[m, "perf_adj_pn"], df.loc[m, "perf_adj_ps"]
            n_o, s_o = n_all - n_l, s_all - s_l
            contrast = (s_l / n_l.replace(0, np.nan)) - (s_o / n_o.replace(0, np.nan))
            n_info = (n_l * n_o / (n_l + n_o).replace(0, np.nan)).fillna(0)
            df.loc[m, f"zo_{key}"] = combine(df.loc[m, f"z_{key}"], contrast.fillna(0), n_info, T.tau(key, lev), kh["contrast"])
    df["F_own"] = (rel("front") * df["q_front"].fillna(0) * df["zo_front"].fillna(0)
                   + rel("kire") * df["q_kire"].fillna(0) * df["zo_kire"].fillna(0)
                   + df["r_surface"] * df["zo_surface"].fillna(0) + df["r_dbucket"] * df["zo_dbucket"].fillna(0)
                   + rel("debut") * df["is_debut_race"].astype(float) * df["z_debut"].fillna(0))
    return df


def estimate_kh(runs: pd.DataFrame) -> dict:
    """k_h = 馬内の走のばらつき / 馬の真の値のばらつき(2018〜2020年の走、係数は2017年末まで)。"""
    v = A.run_values(runs[runs["year"] <= 2017], runs[runs["year"].between(*FIT)])
    out = {}
    for col in ["perf_adj", "front", "kire"]:
        g = v.dropna(subset=[col]).groupby("horse_id")[col]
        st = pd.DataFrame({"m": g.mean(), "n": g.count(), "v": g.var()})
        st = st[st["n"] >= 3]
        within = float(np.average(st["v"], weights=st["n"] - 1))
        tau2 = float(np.var(st["m"]) - np.mean(within / st["n"]))
        out[col] = within / max(tau2, 1e-9)
    out["contrast"] = out["perf_adj"] * 2   # 2群の差は分散が約2倍
    return out


# ------------------------------------------------------------------ Plackett–Luce(上位3頭)
class PL:
    def __init__(self, df: pd.DataFrame, feats: list):
        d = df.sort_values(["race_id", "pos", "umaban_num"])
        self.race_ids, idx = np.unique(d["race_id"].to_numpy(), return_index=True)
        counts = np.diff(np.append(idx, len(d)))
        R, N = len(counts), int(counts.max())
        self.X = np.zeros((R, N, len(feats)))
        self.mask = np.zeros((R, N), bool)
        pos = np.arange(len(d)) - np.repeat(idx, counts)
        ri = np.repeat(np.arange(R), counts)
        self.X[ri, pos] = d[feats].fillna(0).to_numpy(float)
        self.mask[ri, pos] = True
        self.feats = feats

    def race_ll(self, beta):
        u = self.X @ beta
        u = np.where(self.mask, u, -np.inf)
        ll = np.zeros(len(u))
        for k in range(3):
            rem = u[:, k:]
            mx = np.max(rem, axis=1, keepdims=True)
            lse = mx[:, 0] + np.log(np.sum(np.exp(rem - mx), axis=1))
            ll += u[:, k] - lse
        return ll

    def nll_grad(self, beta):
        u = np.where(self.mask, self.X @ beta, -np.inf)
        nll, grad = 0.0, np.zeros(len(beta))
        for k in range(3):
            rem = u[:, k:]
            mx = np.max(rem, axis=1, keepdims=True)
            e = np.exp(rem - mx)
            p = e / e.sum(axis=1, keepdims=True)
            nll -= np.sum(u[:, k] - (mx[:, 0] + np.log(e.sum(axis=1))))
            grad -= self.X[:, k, :].sum(axis=0) - np.einsum("rn,rnf->f", p, self.X[:, k:, :])
        return nll, grad

    def fit(self):
        r = minimize(self.nll_grad, np.zeros(self.X.shape[2]), jac=True, method="L-BFGS-B")
        return r.x


def block_boot(delta: np.ndarray, blocks: np.ndarray, n=N_BOOT, seed=SEED):
    df = pd.DataFrame({"d": delta, "b": blocks})
    g = df.groupby("b")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy()
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        pick = rng.integers(0, len(s), len(s))
        out[i] = s[pick].sum() / c[pick].sum()
    return [round(float(np.percentile(out, 2.5)), 5), round(float(np.percentile(out, 97.5)), 5)]


def compare(fit_df, test_df, base, extra, name, blocks_col="block"):
    """base→base+extra の ΔLL(テスト、1レースあたり)。β はフィット年で推定。"""
    m0, m1 = PL(fit_df, base), PL(fit_df, base + extra)
    b0, b1 = m0.fit(), m1.fit()
    t0, t1 = PL(test_df, base), PL(test_df, base + extra)
    d = t1.race_ll(b1) - t0.race_ll(b0)
    blocks = test_df.drop_duplicates("race_id").set_index("race_id").loc[t0.race_ids, blocks_col].to_numpy()
    years = pd.Series(t0.race_ids).str[:4].to_numpy()
    by_year = pd.Series(d).groupby(years).mean().round(5).to_dict()
    return {"name": name, "base": base, "extra": extra, "beta_base_only": [round(float(x), 4) for x in b0],
            "beta_full": [round(float(x), 4) for x in b1], "n_races": int(len(d)),
            "dll_per_race": round(float(d.mean()), 5), "ci95_block": block_boot(d, blocks),
            "ci95_year_block": block_boot(d, years, n=N_BOOT), "by_year": by_year, "_delta": d, "_race_ids": t0.race_ids}


# ------------------------------------------------------------------ メイン
def main():
    t0 = time.time()
    runs = pd.read_parquet(A.RUNS)
    th = A.fixed_thresholds(runs)
    runs = A.add_tiers(runs, th)
    runs["umaban_num"] = pd.to_numeric(runs["umaban"], errors="coerce")
    req = pd.read_parquet(C.OUT_DIR / "race_req.parquet")
    # 確定オッズ(市場の副次比較用)
    od = []
    for y in range(FIT[0], TEST[1] + 1):
        for p in sorted(glob.glob(str(C.RESULTS_DIR / str(y) / "*.csv"))):
            od.append(pd.read_csv(p, dtype=str, usecols=["race_id", "horse_id", "odds_final", "popularity", "umaban"]))
    od = pd.concat(od).drop_duplicates(["race_id", "horse_id"])
    od["odds"] = pd.to_numeric(od["odds_final"], errors="coerce")
    od["pop"] = pd.to_numeric(od["popularity"], errors="coerce")
    kh = estimate_kh(runs)
    log("k_h", kh)
    frames = []
    for Y in range(FIT[0], TEST[1] + 1):
        f = features_for_year(runs, req, Y)
        f = add_own(runs, f, Y, kh)
        frames.append(f)
        log("features", Y, len(f))
    D = pd.concat(frames, ignore_index=True)
    D = D.merge(od[["race_id", "horse_id", "odds", "pop"]], on=["race_id", "horse_id"], how="left")
    D["logp"] = np.log((1 / D["odds"]) / D.groupby("race_id")["odds"].transform(lambda s: (1 / s).sum()))
    D["block"] = D["date"].dt.strftime("%Y%m%d") + "_" + D["race_id"].str[4:6]
    D = D[D["field"] >= 5]
    fitD = D[D["year"].between(*FIT)]
    testD = D[D["year"].between(*TEST)]
    res = {"fit_years": FIT, "test_years": TEST, "n_fit_races": int(fitD["race_id"].nunique()),
           "n_test_races": int(testD["race_id"].nunique()), "k_h": kh,
           "coverage": {c: round(float(testD[c].notna().mean()), 3) for c in ["S", "z_front", "z_kire", "z_surface", "z_dbucket", "z_debut", "q_front", "logp"]}}

    # 整合: β2=0 なら A1 の尤度は A0 と一致
    p0, p1 = PL(testD.head(5000), ["S"]), PL(testD.head(5000), ["S", "F"])
    assert np.allclose(p0.race_ll(np.array([0.3])), p1.race_ll(np.array([0.3, 0.0])))

    main_h = compare(fitD, testD, ["S"], ["F"], "主仮説: 父の総合力 → +適合度(血統のみ)")
    res["main"] = {k: v for k, v in main_h.items() if not k.startswith("_")}
    res["main"]["pass"] = bool(main_h["ci95_block"][0] > 0)
    log("main", res["main"]["dll_per_race"], res["main"]["ci95_block"])

    # 主要副次(Holm、この順)
    sec = []
    rid_young = set(testD.loc[testD["career"] <= 3, "race_id"])
    d_main = pd.Series(main_h["_delta"], index=main_h["_race_ids"])
    blocks = testD.drop_duplicates("race_id").set_index("race_id")["block"]
    for name, rids in [("(1) キャリア0〜3走の馬を含むレース", rid_young),
                       ("(1b) 新馬・未勝利戦", set(testD.loc[testD["class_ord"] == 0, "race_id"]))]:
        dd = d_main[d_main.index.isin(rids)]
        sec.append({"name": name, "n_races": int(len(dd)), "dll_per_race": round(float(dd.mean()), 5),
                    "ci95_block": block_boot(dd.to_numpy(), blocks.loc[dd.index].to_numpy())})
    fs = set(testD.loc[(testD["career"] >= 1) & testD["first_in_band"], "race_id"])
    dd = d_main[d_main.index.isin(fs)]
    sec.append({"name": "(2) 距離帯の初出走の馬を含むレース", "n_races": int(len(dd)), "dll_per_race": round(float(dd.mean()), 5),
                "ci95_block": block_boot(dd.to_numpy(), blocks.loc[dd.index].to_numpy())})
    c3 = compare(fitD, testD, ["S", "F_v1"], ["F"], "(3) v1の適合度に対する v2 の上積み(同じ形)")
    c3b = compare(fitD, testD, ["S"], ["F_v1"], "(3b) 参考: v1の適合度だけの ΔLL")
    c4 = compare(fitD, testD, ["S_own"], ["F_own"], "(4) 血統+実績: 自身の総合力 → +適合度(実績合成)")
    c4b = compare(fitD, testD, ["S_own", "F"], ["F_own"], "(4b) 参考: 血統のみの適合度に対する実績合成の上積み")
    ok_race = lambda X: X.groupby("race_id")["logp"].transform(lambda s: bool(np.isfinite(s).all()))  # noqa: E731
    c6 = compare(fitD[ok_race(fitD)], testD[ok_race(testD)], ["logp", "S"], ["F"], "(6) 市場(確定オッズ)+父の総合力 → +適合度")
    for c in [c3, c3b, c4, c4b, c6]:
        sec.append({k: v for k, v in c.items() if not k.startswith("_")})
    # Holm(主要副次 (1)(2)(3)(4)(6)、片側: CI下限>0 の判定を、順に α を厳しくして)
    res["secondary"] = sec
    res["secondary_note"] = "(5) DL挑戦モデル v3 は R3b で別に評価する。Holm 補正は p 値の代わりに、95%CI下限>0 を順に判定し、通らなかった時点で以降を探索扱いとする。"

    # プラセボ: 同じ芝ダの別のレースの基準(脚質・キレの要求、距離帯、新馬かどうか)を当てる
    rng = np.random.default_rng(SEED)
    plac = []
    for rep in range(N_PLACEBO):
        P = []
        for part in (fitD, testD):
            races = part.drop_duplicates("race_id")[["race_id", "surface", "q_front", "q_kire", "dbucket", "is_debut_race"]]
            donor = races.copy()
            for s, idx in races.groupby("surface").groups.items():
                perm = rng.permutation(len(idx))
                donor.loc[idx, ["q_front", "q_kire", "dbucket", "is_debut_race"]] = races.loc[idx, ["q_front", "q_kire", "dbucket", "is_debut_race"]].to_numpy()[perm]
            dmap = donor.set_index("race_id")
            q = part["race_id"].map(dmap["q_front"]), part["race_id"].map(dmap["q_kire"])
            db = part["race_id"].map(dmap["dbucket"])
            zb = pd.Series(np.nan, index=part.index)
            rb = pd.Series(0.0, index=part.index)
            for b in BUCKETS:
                m = db == b
                zb[m] = part.loc[m, f"z_b|{b}"]
                rb[m] = rel("dbucket", b)
            deb = part["race_id"].map(dmap["is_debut_race"]).astype(bool)
            P.append(part.assign(F_p=fit_score(part, q[0], q[1], zb, rb, deb)))
        cp = compare(P[0], P[1], ["S"], ["F_p"], f"placebo{rep}")
        plac.append(cp["dll_per_race"])
    res["placebo"] = {"dll_per_race": plac, "mean": round(float(np.mean(plac)), 5), "sd": round(float(np.std(plac)), 5),
                      "max": round(float(np.max(plac)), 5), "actual_above_max": bool(res["main"]["dll_per_race"] > max(plac)),
                      "note": "同じ芝ダの中で入れ替えるため、芝ダの軸の寄与はプラセボでも残る(事前登録 §6 のとおり)"}
    log("placebo", res["placebo"]["mean"], res["placebo"]["max"])

    # 頑健性: 出走数上位20種牡馬を1頭ずつ含むレースを除いたときの主仮説の ΔLL
    top_sires = testD["ped_S_horse_id"].value_counts().head(20).index
    jk = []
    for s in top_sires:
        rids = set(testD.loc[testD["ped_S_horse_id"] == s, "race_id"])
        dd = d_main[~d_main.index.isin(rids)]
        jk.append(round(float(dd.mean()), 5))
    res["jackknife_top20_sires"] = {"min": min(jk), "max": max(jk), "values": jk}

    # E3: 実際のレースのリフトのコース別の平均、2021〜2023 vs 2024〜2026
    rq = req[req["year"].between(*TEST)]
    e3 = {}
    for a in ["front", "kire"]:
        e = rq[rq["year"] <= 2023].groupby("course_key_io")[f"lift_{a}"].agg(["mean", "count"])
        l = rq[rq["year"] >= 2024].groupby("course_key_io")[f"lift_{a}"].agg(["mean", "count"])
        j = e.join(l, lsuffix="_e", rsuffix="_l", how="inner")
        j = j[(j["count_e"] >= 30) & (j["count_l"] >= 15)]
        r, n = C.corr(j["mean_e"], j["mean_l"])
        e3[a] = {"r": None if r is None else round(r, 3), "n_course_keys": n}
    res["E3"] = e3

    # 参考: 適合度のレース内順位別の複勝率・単勝回収率・複勝回収率(人気帯別)
    pay = []
    for y in range(TEST[0], TEST[1] + 1):
        for p in sorted(glob.glob(str(C.PROJECT_ROOT / "data" / "payouts" / str(y) / "*.csv"))):
            x = pd.read_csv(p, dtype=str)
            pay.append(x[x["bet_type"] == "複勝"][["race_id", "combination", "payout"]])
    pay = pd.concat(pay)
    pay["umaban_num"] = pd.to_numeric(pay["combination"], errors="coerce")
    pay["place_pay"] = pd.to_numeric(pay["payout"], errors="coerce")
    T2 = testD.merge(pay[["race_id", "umaban_num", "place_pay"]].drop_duplicates(["race_id", "umaban_num"]),
                     on=["race_id", "umaban_num"], how="left")
    T2["fit_rank"] = T2.groupby("race_id")["F"].rank(ascending=False, method="first")
    T2["popband"] = pd.cut(T2["pop"], [0, 3, 9, 99], labels=["1-3番人気", "4-9番人気", "10番人気以下"])
    T2["top3"] = (T2["pos"] <= 3).astype(float)
    T2["win_ret"] = np.where(T2["pos"] == 1, T2["odds"] * 100, 0)
    T2["place_ret"] = T2["place_pay"].fillna(0)
    ref = {}
    for band, g in T2.groupby("popband", observed=True):
        rows = {}
        for lab, gg in [("適合度1位", g[g["fit_rank"] == 1]), ("適合度2-3位", g[g["fit_rank"].between(2, 3)]), ("人気帯全体", g)]:
            rows[lab] = {"n": int(len(gg)), "top3_rate": round(float(gg["top3"].mean()), 4),
                         "win_roi": round(float(gg["win_ret"].mean()), 1), "place_roi": round(float(gg["place_ret"].mean()), 1)}
        ref[str(band)] = rows
    res["reference_by_popularity"] = ref

    # manifest
    files = sorted(glob.glob(str(A.TABLE_DIR / "anc_*.parquet")))
    res["manifest"] = {
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip(),
        "runs_manifest": json.loads((C.OUT_DIR / "runs_all_manifest.json").read_text(encoding="utf-8"))["results_sha256"],
        "tables_sha256": hashlib.sha256(b"".join(hashlib.sha256(Path(f).read_bytes()).digest() for f in files)).hexdigest(),
        "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "seed": SEED,
        "prereg_sha256": hashlib.sha256((C.PROJECT_ROOT / "data/jra_pipeline/pedigree_reports/RADAR_V2_PREREG_2026_10_06.md").read_bytes()).hexdigest()}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    log("wrote", OUT)


if __name__ == "__main__":
    main()
