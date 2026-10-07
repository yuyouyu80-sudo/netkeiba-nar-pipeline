# -*- coding: utf-8 -*-
"""血統レーダー v2: 血統から推し量るスタミナの3軸(2026-10-07、事前登録 追補7)。

プラン: C:/Users/yuyou/.claude/plans/wiggly-sniffing-metcalfe.md(2026-10-07 版)。既存の LEVEL_AXES・build_tables・Tables・
runs_all.parquet・確定済みの JSON は変えない(この側で値を作り、既存の関数を呼ぶだけ)。

## 軸(馬の値は馬内の傾き、祖先表は既存の関数、3系統の等重み平均)
- A 持続力: レースの量 pace_hi = −(pace − 平均)/SD(芝ダ×距離帯、2011〜2020年で固定。正 = 前半が速い消耗戦)。
  走の値 perf_A = perf_adj − 「前半位置×pace_hi」の交差の寄与(芝ダ別の回帰、2020年まで)。馬の値 = perf_A の pace_hi への馬内の傾き。
- B 距離延長への反応: 延長 = 前走(runs 上)からの log 距離比 ≥ +0.10、または距離帯が上がる延長。
  走の値 r_B = perf_adj − (前走の perf_adj・log(1+キャリア)・年齢帯 の回帰、2020年まで)。
  馬の値 = 同じ距離帯の中で「延長で来た走の平均 − それ以外(前走あり)の走の平均」を n_e·n_o/(n_e+n_o) で重み付け平均。
- C 上がりのかかる展開での脚: レースの量 c_z = 完走馬の上がり3F の中央値 −(芝ダ×距離×クラス群の平均)− γ·pace_hi を SD で割ったもの
  (2020年まで、正 = 上がりがかかる)。走の値 perf_C = perf_adj − 「前半位置×c_z」の交差の寄与。馬の値 = perf_C の c_z への馬内の傾き。
- 要求 q_A・q_C: 実際の pace_hi・c_z を、発走前に分かる量(同等レースのその日より前の平均(radar_v2_reference の L1〜L4)、
  クラス群、馬場、出走馬の過去の前半位置、頭数)で回帰した予測値(芝ダ別、2020年まで)。京都は改修後(2023-04〜)をコースキーで分ける。

使い方:
  python radar_v2_stamina.py --fixed     # 2020年までで係数・閾値を決める → radar_v2/stamina_fixed.json
  python radar_v2_stamina.py --tables    # 時点表 Y=2013..2026 → radar_v2/tables/stam_anc_{Y}.parquet, stam_hv_{Y}.parquet
  python radar_v2_stamina.py --inner     # 2020年までの内側検証 → radar_v2/stamina_inner.json
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_reference as RF  # noqa: E402

FIX_END = 2020
SEED = 20261006
AXES = ["stamA", "stamB", "stamC"]
L3_CACHE = C.OUT_DIR / "stamina_l3.parquet"
FIXED = C.OUT_DIR / "stamina_fixed.json"
INNER_OUT = C.OUT_DIR / "stamina_inner.json"
EXT_LOG = 0.10            # 延長の判定(log 距離比)
EXT_CAP = float(np.log(1.5))
SXX_MIN = 1.0             # 傾きを出す最小の Σ(x−x̄)²(x は標準化済み)
KYOTO_POST = pd.Timestamp("2023-04-01")
BUCKET_ORDER = {b: i for i, b in enumerate(["短距離(~1400m)", "マイル(1401-1800m)", "中距離(1801-2200m)", "長距離(2201m~)"])}
CLASS_GROUPS = ["新馬", "未勝利", "1勝", "2-3勝", "OP以上"]
GOINGS = ["良", "稍重", "重", "不良"]
log = A.log


# ------------------------------------------------------------------ 入力
def load_l3() -> pd.DataFrame:
    """上がり3F(秒)。race_results から (race_id, horse_id) ごと。Git管理外のキャッシュ。"""
    if L3_CACHE.exists():
        return pd.read_parquet(L3_CACHE)
    frames = []
    for p in sorted(C.RESULTS_DIR.glob("20*/*.csv")):
        frames.append(pd.read_csv(p, dtype=str, usecols=["race_id", "horse_id", "last_3f"]))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    d["l3sec"] = pd.to_numeric(d["last_3f"], errors="coerce")
    d = d[["race_id", "horse_id", "l3sec"]]
    d.to_parquet(L3_CACHE, index=False)
    return d


def load_base(runs: pd.DataFrame | None = None) -> pd.DataFrame:
    """runs_all(または出走表の行を足した走の表、radar_v2_stamina_live)に、閾値の区分・上がり3F・京都の改修後のキー・同等レースのキーを付ける。"""
    runs = pd.read_parquet(A.RUNS) if runs is None else runs.copy()
    runs = A.add_tiers(runs, A.fixed_thresholds(runs))
    runs["umaban_num"] = pd.to_numeric(runs["umaban"], errors="coerce")
    l3 = load_l3()
    runs = runs.merge(l3, on=["race_id", "horse_id"], how="left")
    # 京都は改修後を別コースとして扱う(同等レースのキー)
    post = (runs["racecourse"] == "京都") & (runs["date"] >= KYOTO_POST)
    runs["course_key_io"] = runs["course_key_io"].where(~post, runs["course_key_io"] + "|改修後")
    runs = RF.class_keys(runs)
    return runs.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


def ped_frame(runs: pd.DataFrame) -> pd.DataFrame:
    return runs[["horse_id"] + [c for _, c in C.ROLES]].drop_duplicates("horse_id")


# ------------------------------------------------------------------ レースの量
def race_table(runs: pd.DataFrame) -> pd.DataFrame:
    first = runs.drop_duplicates("race_id").set_index("race_id")
    R = first[["date", "year", "surface", "distance", "dbucket", "course_key_io", "class_key", "class_group", "day_seg",
               "L1", "L2", "L3", "L4", "going", "is_debut_race", "pace", "racecourse"]].copy()
    R["l3_med"] = runs.groupby("race_id")["l3sec"].median()
    R["n_start"] = runs.groupby("race_id")["umaban_num"].max()
    # 出走馬の過去の前半位置(その日より前の自身の走の front_raw の平均)の出走馬平均と、過去走のある馬の割合(R3b と同じ作り方)
    h = runs[["race_id", "horse_id", "front_raw"]]
    x, n = h["front_raw"].fillna(0), h["front_raw"].notna().astype(float)
    pf = (x.groupby(h["horse_id"]).cumsum() - x) / (n.groupby(h["horse_id"]).cumsum() - n).replace(0, np.nan)
    R["field_front"] = pf.groupby(h["race_id"]).mean()
    R["field_hist"] = pf.notna().groupby(h["race_id"]).mean()
    return R


def fit_race_quantities(R: pd.DataFrame) -> dict:
    """pace_hi の標準化、C の基準と γ・SD(2020年まで)。"""
    F = R[R["year"] <= FIX_END]
    assert F["year"].max() <= FIX_END
    fx = {"pace": {}, "c_base": {}, "c_base_b": {}, "c_gamma": {}, "c_sd": {}}
    for key, g in F.groupby(F["surface"] + "|" + F["dbucket"]):
        fx["pace"][key] = [float(g["pace"].mean()), float(g["pace"].std())]
    R = apply_pace(R, fx)
    F = R[R["year"] <= FIX_END]
    k3 = F["surface"] + "|" + F["distance"].astype(int).astype(str) + "|" + F["class_group"]
    m = F.groupby(k3)["l3_med"].agg(["mean", "count"])
    fx["c_base"] = {k: float(v) for k, v in m.loc[m["count"] >= 30, "mean"].items()}
    kb = F["surface"] + "|" + F["dbucket"] + "|" + F["class_group"]
    fx["c_base_b"] = {k: float(v) for k, v in F.groupby(kb)["l3_med"].mean().items()}
    R = apply_c_base(R, fx)
    F = R[R["year"] <= FIX_END]
    for s, g in F.dropna(subset=["c_res0", "pace_hi"]).groupby("surface"):
        gam = float(np.polyfit(g["pace_hi"], g["c_res0"], 1)[0])
        fx["c_gamma"][s] = gam
        fx["c_sd"][s] = float((g["c_res0"] - gam * g["pace_hi"]).std())
    return fx


def apply_pace(R: pd.DataFrame, fx: dict) -> pd.DataFrame:
    R = R.copy()
    key = R["surface"] + "|" + R["dbucket"]
    mu = key.map(lambda k: fx["pace"].get(k, [np.nan, np.nan])[0])
    sd = key.map(lambda k: fx["pace"].get(k, [np.nan, np.nan])[1])
    R["pace_hi"] = -(R["pace"] - mu) / sd
    return R


def apply_c_base(R: pd.DataFrame, fx: dict) -> pd.DataFrame:
    R = R.copy()
    k3 = R["surface"] + "|" + R["distance"].astype(int).astype(str) + "|" + R["class_group"]
    kb = R["surface"] + "|" + R["dbucket"] + "|" + R["class_group"]
    base = k3.map(fx["c_base"]).fillna(kb.map(fx["c_base_b"]))
    R["c_res0"] = R["l3_med"] - base
    return R


def apply_race_quantities(R: pd.DataFrame, fx: dict) -> pd.DataFrame:
    R = apply_c_base(apply_pace(R, fx), fx)
    gam = R["surface"].map(fx["c_gamma"])
    sd = R["surface"].map(fx["c_sd"])
    R["c_z"] = (R["c_res0"] - gam * R["pace_hi"]) / sd
    return R


# ------------------------------------------------------------------ 要求 q(発走前に分かる量による予測)
def q_design(R: pd.DataFrame, prior: pd.Series) -> np.ndarray:
    cols = [np.ones(len(R)), prior.fillna(0).to_numpy(), R["field_front"].fillna(0.5).to_numpy(),
            R["field_hist"].fillna(0).to_numpy(), R["n_start"].fillna(14).to_numpy() / 18.0]
    cols += [(R["class_group"] == c).astype(float).to_numpy() for c in CLASS_GROUPS[1:]]
    cols += [(R["going"] == g).astype(float).to_numpy() for g in GOINGS[1:]]
    return np.column_stack(cols)


def q_priors(R: pd.DataFrame, fx: dict) -> pd.DataFrame:
    """同等レース(L1〜L4、その日より前)の pace_hi・c_z の縮約平均(radar_v2_reference.build_req、k は2020年まで)。"""
    out = pd.DataFrame(index=R.index)
    for val in ["pace_hi", "c_z"]:
        k = fx.setdefault("q_k", {}).get(val)
        if k is None:
            k = RF.estimate_k(R, val)
            fx["q_k"][val] = float(k)
        rq = RF.build_req(R.sort_values("date"), val, k)
        out[f"pri_{val}"] = rq["req"].reindex(R.index)
    return out


def fit_q(R: pd.DataFrame, fx: dict) -> dict:
    P = q_priors(R, fx)
    R = R.join(P)
    F = R[(R["year"] <= FIX_END) & (R["year"] >= 2013)]
    assert F["year"].max() <= FIX_END
    fx["q"] = {}
    for val, name in [("pace_hi", "qA"), ("c_z", "qC")]:
        for s, g in F.dropna(subset=[val]).groupby("surface"):
            X = q_design(g, g[f"pri_{val}"])
            b, *_ = np.linalg.lstsq(X, g[val].to_numpy(), rcond=None)
            pred = X @ b
            r2 = 1 - np.var(g[val].to_numpy() - pred) / np.var(g[val].to_numpy())
            fx["q"][f"{name}|{s}"] = {"beta": [float(x) for x in b], "r2_in_sample": float(r2)}
    return fx


def apply_q(R: pd.DataFrame, fx: dict, going=None) -> pd.DataFrame:
    """q_A・q_C。going を与えると全レースその馬場と仮定した値(発走前の4通り用)。"""
    if "pri_pace_hi" not in R.columns:
        R = R.join(q_priors(R, fx))
    R = R.copy()
    G = R if going is None else R.assign(going=going)
    for val, name in [("pace_hi", "qA"), ("c_z", "qC")]:
        R[name] = np.nan
        for s in ["芝", "ダ"]:
            m = (G["surface"] == s).to_numpy()
            b = np.array(fx["q"][f"{name}|{s}"]["beta"])
            R.loc[m, name] = q_design(G[m], G.loc[m, f"pri_{val}"]) @ b
    return R


# ------------------------------------------------------------------ 走ごとの値
def adj_fit(v: pd.DataFrame, xcol: str) -> dict:
    """perf_adj ~ 1 + f + f² + x + f·x + f²·x(芝ダ別、f = front_raw − 0.5)。交差の係数を返す。"""
    out = {}
    for s, g in v.dropna(subset=["perf_adj", "front_raw", xcol]).groupby("surface"):
        f, x = g["front_raw"].to_numpy() - 0.5, g[xcol].to_numpy()
        X = np.column_stack([np.ones(len(g)), f, f * f, x, f * x, f * f * x])
        b, *_ = np.linalg.lstsq(X, g["perf_adj"].to_numpy(), rcond=None)
        out[s] = [float(b[4]), float(b[5])]
    return out


def adj_apply(v: pd.DataFrame, xcol: str, coef: dict) -> pd.Series:
    f = (v["front_raw"] - 0.5).fillna(0)
    b1 = v["surface"].map(lambda s: coef[s][0])
    b2 = v["surface"].map(lambda s: coef[s][1])
    return v["perf_adj"] - (b1 * f + b2 * f * f) * v[xcol].fillna(0)


def b_design(v: pd.DataFrame) -> np.ndarray:
    cols = [np.ones(len(v)), v["prev_perf_adj"].to_numpy(), np.log1p(v["career"].to_numpy())]
    cols += [(v["age_band"] == a).astype(float).to_numpy() for a in ["3", "4+"]]
    return np.column_stack(cols)


def add_run_cols(v: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    """v(run_values 済み、horse_id・date 順)にレースの量と延長の判定、前走の perf_adj を足す。"""
    v = v.join(R[["pace_hi", "c_z"]], on="race_id")
    g = v.groupby("horse_id")
    v["prev_perf_adj"] = g["perf_adj"].shift(1)
    bi = v["dbucket"].map(BUCKET_ORDER)
    prev_bi = g["dbucket"].shift(1).map(BUCKET_ORDER)
    d = v["logd"] - v["prev_logd"]
    has_prev = v["prev_logd"].notna()
    v["dlogd"] = d.where(has_prev)
    v["ext"] = has_prev & ((d >= EXT_LOG) | ((bi > prev_bi) & (d > 0)))
    v["short"] = has_prev & ((d <= -EXT_LOG) | ((bi < prev_bi) & (d < 0)))
    v["ext_amt"] = np.where(v["ext"], d.clip(0, EXT_CAP), 0.0)
    return v


def fit_run_adjust(v: pd.DataFrame, fx: dict) -> dict:
    assert v["year"].max() <= FIX_END
    fx["adj_A"] = adj_fit(v, "pace_hi")
    fx["adj_C"] = adj_fit(v, "c_z")
    fx["B"] = {}
    for s, g in v.dropna(subset=["perf_adj", "prev_perf_adj"]).groupby("surface"):
        b, *_ = np.linalg.lstsq(b_design(g), g["perf_adj"].to_numpy(), rcond=None)
        fx["B"][s] = [float(x) for x in b]
    return fx


def apply_run_adjust(v: pd.DataFrame, fx: dict) -> pd.DataFrame:
    v = v.copy()
    v["perf_A"] = adj_apply(v, "pace_hi", fx["adj_A"])
    v["perf_C"] = adj_apply(v, "c_z", fx["adj_C"])
    v["r_B"] = np.nan
    for s in ["芝", "ダ"]:
        m = (v["surface"] == s) & v["prev_perf_adj"].notna() & v["perf_adj"].notna()
        v.loc[m, "r_B"] = v.loc[m, "perf_adj"] - b_design(v[m]) @ np.array(fx["B"][s])
    return v


# ------------------------------------------------------------------ 馬の値
def horse_slopes(v: pd.DataFrame, y: str, x: str, clip=None) -> pd.DataFrame:
    """馬内の傾き Sxy/Sxx。n = Sxx(傾きの精度に比例、重み w = min(n, 5) は既存の関数が付ける)。"""
    d = v[["horse_id", y, x]].dropna()
    g = d.groupby("horse_id")
    mx, my = g[x].transform("mean"), g[y].transform("mean")
    d = d.assign(sxy=(d[x] - mx) * (d[y] - my), sxx=(d[x] - mx) ** 2)
    s = d.groupby("horse_id").agg(sxy=("sxy", "sum"), sxx=("sxx", "sum"), cnt=(x, "count"))
    s = s[(s["cnt"] >= 3) & (s["sxx"] >= SXX_MIN)]
    x_ = s["sxy"] / s["sxx"]
    if clip is not None:
        x_ = x_.clip(*clip)
    return pd.DataFrame({"x": x_, "n": s["sxx"]})


def horse_ext(v: pd.DataFrame) -> pd.DataFrame:
    d = v[v["r_B"].notna()]
    st = d.groupby(["horse_id", "dbucket", "ext"])["r_B"].agg(["mean", "count"]).unstack("ext")
    st = st.dropna()
    if st.empty:
        return pd.DataFrame(columns=["x", "n"])
    ne, no = st[("count", True)], st[("count", False)]
    diff = st[("mean", True)] - st[("mean", False)]
    info = ne * no / (ne + no)
    num = (info * diff).groupby(level="horse_id").sum()
    den = info.groupby(level="horse_id").sum()
    return pd.DataFrame({"x": num / den, "n": den})


def horse_values(v: pd.DataFrame, fx: dict) -> dict:
    return {("stamA", "-"): horse_slopes(v, "perf_A", "pace_hi", fx.get("clip_A")),
            ("stamB", "-"): horse_ext(v),
            ("stamC", "-"): horse_slopes(v, "perf_C", "c_z", fx.get("clip_C"))}


def build_stam_tables(v: pd.DataFrame, ped: pd.DataFrame, fx: dict, horses=None, return_hv=False):
    if horses is not None:
        v = v[v["horse_id"].isin(horses)]
    out, hvs = {}, {}
    for key, h in horse_values(v, fx).items():
        if len(h) < 20:
            continue
        hc = C.center_horse_values(h)
        hvs[key] = hc
        for role, rc in C.ROLES:
            out[key + (role,)] = C.ancestor_table(hc, ped, rc)
    return (out, hvs) if return_hv else out


# ------------------------------------------------------------------ 時点表の読み出し(既存の Tables の __init__ だけ差し替え)
def stam_tables_class():
    import radar_v2_eval as E

    class StamTables(E.Tables):
        def __init__(self, Y):
            anc = pd.read_parquet(A.TABLE_DIR / f"stam_anc_{Y}.parquet")
            self.anc = {k: g.set_index("ancestor_id") for k, g in anc.groupby(["axis", "level", "role"])}
            hv = pd.read_parquet(A.TABLE_DIR / f"stam_hv_{Y}.parquet")
            self.hv = {k: g.set_index("horse_id")[["xc", "w"]] for k, g in hv.groupby(["axis", "level"])}
    return StamTables


# ------------------------------------------------------------------ 固定の係数(2020年まで)
def prepare(runs: pd.DataFrame, fx: dict | None = None):
    """fx が無ければ 2020年までで推定して返す。R(レース表、q 付き)と fx。"""
    R = race_table(runs)
    if fx is None:
        fx = fit_race_quantities(R)
        R = apply_race_quantities(R, fx)
        fx = fit_q(R, fx)
    else:
        R = apply_race_quantities(R, fx)
    R = apply_q(R, fx)
    return R, fx


def fixed_params(runs: pd.DataFrame) -> dict:
    t0 = time.time()
    R, fx = prepare(runs)
    le = runs[runs["year"] <= FIX_END]
    assert le["year"].max() <= FIX_END
    v = add_run_cols(A.run_values(le, le), R)
    fx = fit_run_adjust(v, fx)
    v = apply_run_adjust(v, fx)
    for ax, y, x in [("A", "perf_A", "pace_hi"), ("C", "perf_C", "c_z")]:
        s = horse_slopes(v, y, x)["x"]
        fx[f"clip_{ax}"] = [float(s.quantile(0.01)), float(s.quantile(0.99))]
    # q の当てはまり(2013〜2017年で推定し、2018〜2020年で確かめた決定係数。採否の判断材料、q 自体は2020年までの係数)
    fx["q_r2_oos"] = {}
    Rq = R[R["year"].between(2013, FIX_END)]
    for val, name in [("pace_hi", "qA"), ("c_z", "qC")]:
        for s, g in Rq.dropna(subset=[val]).groupby("surface"):
            tr, te = g[g["year"] <= 2017], g[g["year"] >= 2018]
            b, *_ = np.linalg.lstsq(q_design(tr, tr[f"pri_{val}"]), tr[val].to_numpy(), rcond=None)
            pred = q_design(te, te[f"pri_{val}"]) @ b
            fx["q_r2_oos"][f"{name}|{s}"] = float(1 - np.var(te[val].to_numpy() - pred) / np.var(te[val].to_numpy()))
    fx["consts"] = {"FIX_END": FIX_END, "EXT_LOG": EXT_LOG, "EXT_CAP": EXT_CAP, "SXX_MIN": SXX_MIN, "KYOTO_POST": str(KYOTO_POST.date())}
    fx["elapsed_sec"] = round(time.time() - t0, 1)
    return fx


def load_fixed() -> dict:
    return json.loads(FIXED.read_text(encoding="utf-8"))


# ------------------------------------------------------------------ 時点表
def tables(runs: pd.DataFrame, fx: dict, years=None):
    R, _ = prepare(runs, fx)
    ped = ped_frame(runs)
    meta = {}
    for Y in years or range(2013, 2027):
        t0 = time.time()
        win = runs[runs["year"] <= Y - 1]
        v = apply_run_adjust(add_run_cols(A.run_values(win, win), R), fx)
        tb, hvs = build_stam_tables(v, ped, fx, return_hv=True)
        info = A.save_tables(tb, A.TABLE_DIR / f"stam_anc_{Y}.parquet")
        part = set(runs.loc[runs["year"] == Y, "horse_id"])
        hrows = [hc.loc[hc.index.isin(part), ["xc", "w"]].assign(axis=k[0], level=k[1]).reset_index() for k, hc in hvs.items()]
        pd.concat(hrows, ignore_index=True).to_parquet(A.TABLE_DIR / f"stam_hv_{Y}.parquet", index=False)
        info.update({"window_end": str(win["date"].max().date()), "n_horses": {k[0]: int(len(h)) for k, h in hvs.items()},
                     "sec": round(time.time() - t0, 1)})
        meta[str(Y)] = info
        log("stam table", Y, info)
    (A.TABLE_DIR / "stam_tables_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    R[["date", "year", "surface", "course_key_io", "class_group", "going", "pace_hi", "c_z", "qA", "qC",
       "pri_pace_hi", "pri_c_z"]].to_parquet(C.OUT_DIR / "stamina_race.parquet")


# ------------------------------------------------------------------ 内側検証(2020年まで)
def r3(x):
    return None if x is None else round(float(x), 3)


def inner(runs: pd.DataFrame, fx: dict) -> dict:
    t0 = time.time()
    runs = runs[runs["year"] <= FIX_END]
    assert runs["year"].max() <= FIX_END
    R, _ = prepare(runs, fx)
    ped = ped_frame(runs)
    anc_runs = runs[runs["year"] <= 2017]
    anc = apply_run_adjust(add_run_cols(A.run_values(anc_runs, anc_runs), R), fx)
    allv = apply_run_adjust(add_run_cols(A.run_values(anc_runs, runs), R), fx)
    tb = build_stam_tables(anc, ped, fx)
    vals, meta = A.shrink_tables(tb)
    hh = anc["horse_id"].drop_duplicates()
    half = hh.map(lambda h: int(hashlib.md5(h.encode()).hexdigest(), 16) % 2)
    T = [build_stam_tables(anc, ped, fx, horses=set(hh[half == i])) for i in (0, 1)]
    first_year = runs.groupby("horse_id")["year"].min()
    chk_h = set(first_year[first_year >= 2018].index)
    chk = allv[allv["horse_id"].isin(chk_h)]
    hv_chk = horse_values(chk, fx)
    out = {"anc_years": "2011-2017", "check": "2018-2020年デビュー馬の 2018〜2020年の走", "axes": {}}
    for axis in AXES:
        rec = {}
        t = tb.get((axis, "-", "sire"))
        a = t[t["n_horses"] >= 30]["mean"]
        share, ratio = float((a > 0).mean()), float(abs(a.median()) / a.std())
        rec["E4"] = {"n_anc": int(len(a)), "share_pos": r3(share), "abs_median_over_sd": r3(ratio),
                     "pass": bool(0.35 <= share <= 0.65 and ratio < 0.25)}
        a0, a1 = T[0][(axis, "-", "sire")], T[1][(axis, "-", "sire")]
        a0, a1 = a0[a0["n_horses"] >= A.MIN_HORSES_HALF], a1[a1["n_horses"] >= A.MIN_HORSES_HALF]
        cm = a0.index.intersection(a1.index)
        r, n = C.corr(a0.loc[cm, "mean"], a1.loc[cm, "mean"])
        rec["E1_sire_sb"], rec["E1_n_anc"] = r3(C.sb(r)), n
        rec["k_sire"], rec["tau_sire"] = round(meta[(axis, "-", "sire")]["k"], 2), round(meta[(axis, "-", "sire")]["tau"], 5)
        hvx = hv_chk[(axis, "-")]
        hvx = hvx[hvx["n"] >= (1.0 if axis == "stamB" else 2.0)].join(ped.set_index("horse_id"))
        p = A.ped_value(hvx, vals, axis)
        r, n = C.corr(p, hvx["x"])
        rec["confirm"] = {"n_horses": n, "r": r3(r), "ci": C.cluster_boot_corr(p, hvx["x"], hvx["ped_S_horse_id"], seed=SEED)}
        rec["judge"] = {"pass": bool(rec["E1_sire_sb"] is not None and rec["E1_sire_sb"] >= 0.3 and rec["E4"]["pass"]
                                     and rec["confirm"]["ci"][0] > 0)}
        out["axes"][axis] = rec
    # 独自性: 父単位で既存の軸(anc_2018 = 2017年末まで)との相関、スタミナの軸どうし
    ex = pd.read_parquet(A.TABLE_DIR / "anc_2018.parquet")
    ex = ex[(ex["role"] == "sire") & (ex["n_horses"] >= 30)]
    exs = {f"{a}|{l}": g.set_index("ancestor_id")["shrunk"] for (a, l), g in ex.groupby(["axis", "level"])}
    want = ["front|-", "kire|-", "logd|-", "perf_adj|-", "surface|芝", "dbucket|長距離(2201m~)", "dbucket|短距離(~1400m)",
            "track|高速", "track|時計かかる", "going|道悪", "growth|-", "debut|-"]
    st = {ax: vals[(ax, "-", "sire")][tb[(ax, "-", "sire")]["n_horses"] >= 30] for ax in AXES}
    out["uniqueness_sire_r"] = {}
    for ax in AXES:
        row = {}
        for w in want + AXES:
            other = exs.get(w) if w in exs else st.get(w)
            if other is None or w == ax:
                continue
            cm = st[ax].index.intersection(other.index)
            r, n = C.corr(st[ax].loc[cm], other.loc[cm])
            row[w] = {"r": r3(r), "n": n}
        out["uniqueness_sire_r"][ax] = row
        out["axes"][ax]["judge"]["duplicate(|r|>=0.7)"] = [k for k, v in row.items() if v["r"] is not None and abs(v["r"]) >= 0.7]
    # レースの区分: A×C のクロス表(3分位、2011〜2020年)と q の当てはまり
    rr = R.dropna(subset=["pace_hi", "c_z"])
    ct = pd.crosstab(pd.qcut(rr["pace_hi"], 3, labels=["遅い", "平均", "速い"]), pd.qcut(rr["c_z"], 3, labels=["上がり速い", "平均", "上がりかかる"]))
    out["race_AxC_crosstab"] = {str(i): {str(c): int(ct.loc[i, c]) for c in ct.columns} for i in ct.index}
    out["race_corr"] = {"pace_hi~c_z": r3(rr[["pace_hi", "c_z"]].corr().iloc[0, 1]),
                        "c_z~track_index": r3(R.join(runs.drop_duplicates("race_id").set_index("race_id")["track_index_num"])[["c_z", "track_index_num"]].corr().iloc[0, 1])}
    out["q_r2_in_sample"] = {k: r3(v["r2_in_sample"]) for k, v in fx["q"].items()}
    out["q_r2_oos_2018_2020"] = {k: r3(v) for k, v in fx["q_r2_oos"].items()}
    out["upper_bound"] = upper_bound(runs, R)
    out["elapsed_sec"] = round(time.time() - t0, 1)
    return out


def upper_bound(runs: pd.DataFrame, R: pd.DataFrame) -> dict:
    """判断材料(予測には使わない): 2018〜2019年で β を推定し 2020年で、C2+主効果 → +交差項 の ΔLL。
    交差項に実際のレースの量(結果の情報)を使った上限と、発走前の q を使った値。時点表(Y−1年末まで、本人除外)を使う。"""
    import radar_v2_eval as E
    ST = stam_tables_class()
    dl = pd.read_parquet(C.OUT_DIR / "dl_table.parquet")
    dl = dl[dl["year"].between(2018, FIX_END)]
    assert dl["year"].max() <= FIX_END
    comps = ["F_front", "F_kire", "F_surf", "F_db", "F_deb"]
    frames = []
    v = add_run_cols(runs[["race_id", "horse_id", "date", "year", "logd", "prev_logd", "dbucket", "perf"]].assign(perf_adj=np.nan), R)
    v = v.set_index(["race_id", "horse_id"])[["ext", "short", "ext_amt"]]
    for Y in (2018, 2019, 2020):
        T = ST(Y)
        d = dl[dl["year"] == Y].copy()
        d = d.join(runs.drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])[["ped_DS_horse_id", "ped_SS_horse_id"]],
                   on=["race_id", "horse_id"])
        for ax in AXES:
            d[f"z_{ax}"] = T.z(d, ax, "-")
        frames.append(d)
    D = pd.concat(frames, ignore_index=True).join(v, on=["race_id", "horse_id"]).join(R[["pace_hi", "c_z", "qA", "qC"]], on="race_id")
    D["ext"], D["short"] = D["ext"].fillna(False).astype(float), D["short"].fillna(False).astype(float)
    D["ext_amt"] = D["ext_amt"].fillna(0.0)
    mains = ["z_stamA", "z_stamB", "z_stamC", "ext", "short", "ext_amt"]
    D["IA_q"], D["IC_q"] = D["qA"] * D["z_stamA"], D["qC"] * D["z_stamC"]
    D["IA_real"], D["IC_real"] = D["pace_hi"] * D["z_stamA"], D["c_z"] * D["z_stamC"]
    D["IB"] = D["ext_amt"] * D["z_stamB"]
    fitD, testD = D[D["year"] <= 2019], D[D["year"] == 2020]
    base = ["S"] + comps
    res = {"fit": "2018-2019", "test": "2020", "coverage_2020": {c: r3(testD[c].notna().mean()) for c in mains[:3] + ["qA", "qC", "pace_hi", "c_z"]}}
    for name, b, extra in [("主効果だけ(C2 → C2+主効果)", base, mains),
                           ("総合: q の交差項3本", base + mains, ["IA_q", "IB", "IC_q"]),
                           ("A: q_A·z_A", base + mains, ["IA_q"]), ("B: 延長の幅·z_B", base + mains, ["IB"]),
                           ("C: q_C·z_C", base + mains, ["IC_q"]),
                           ("上限: 実際の量の交差項(A・C)", base + mains, ["IA_real", "IC_real"]),
                           ("上限 A: 実際の pace_hi·z_A", base + mains, ["IA_real"]), ("上限 C: 実際の c_z·z_C", base + mains, ["IC_real"])]:
        c = E.compare(fitD, testD, b, extra, name)
        res[name] = {k: c[k] for k in ["dll_per_race", "ci95_block", "p_one_sided", "se_block", "n_races", "beta_full"]}
        log("upper", name, c["dll_per_race"], c["ci95_block"])
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fixed", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--inner", action="store_true")
    args = ap.parse_args()
    runs = load_base()
    log("runs", len(runs))
    if args.fixed:
        fx = fixed_params(runs)   # レース表は全期間(その日より前の量だけ)、係数の推定は関数内で2020年までに限る
        FIXED.write_text(json.dumps(fx, ensure_ascii=False, indent=1), encoding="utf-8")
        log("wrote", FIXED.name, {k: v for k, v in fx.items() if k in ("q_r2_oos", "clip_A", "clip_C")})
    if args.tables:
        tables(runs, load_fixed())
    if args.inner:
        R = inner(runs, load_fixed())
        INNER_OUT.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
        log("wrote", INNER_OUT.name)


if __name__ == "__main__":
    main()
