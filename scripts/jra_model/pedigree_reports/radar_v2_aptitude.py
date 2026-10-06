# -*- coding: utf-8 -*-
"""血統レーダー v2 の R2: 祖先の値(時点表)と、残りの軸の内側検証(2026-10-06)。

事前登録 RADAR_V2_PREREG_2026_10_06.md §2〜§3。入力は R1 の radar_v2/runs_all.parquet。

使い方:
  python radar_v2_aptitude.py --inner      # 2020年までで残りの軸を判定(祖先2011〜2017、確認2018〜2020年デビュー馬)
                                           #   → radar_v2/r2_inner_validation.json(事前登録の追補に記録してからR4へ)
  python radar_v2_aptitude.py --tables     # 時点表 Y=2013..2026(Y−1年末までの走)と表示用(全期間)
                                           #   → radar_v2/tables/anc_{Y}.parquet, anc_display.parquet, tables_meta.json

## 軸
- 傾向の軸(馬の走の平均): front(脚質、着順の残差)・kire(キレ、上がり順位の残差)・logd(主戦距離、記述のみ)。
- 条件の軸(馬内差 perf_adj − 同じ馬の他の走の平均 を、条件の区分ごとに馬単位で集計して中心化):
  surface(芝ダ)・dbucket(距離帯)・circ/elev/straight/turn/turf_type(コース形態)・rest(間隔)・class_high(上級条件)・
  track(時計の速さ: 馬場指数の3分位、閾値は2011〜2020年で固定)・going(馬場: 良/道悪)。
- 特別な軸: debut(新馬戦の馬内差)・growth(3歳7月以降の perf_adj 平均 − 2歳時の平均)。
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

RUNS = C.OUT_DIR / "runs_all.parquet"
TABLE_DIR = C.OUT_DIR / "tables"
SEED = 20261006
THRESH_YEARS = (2011, 2020)   # 馬場指数・ペースの3分位の閾値を決める期間(事前登録 §5)
LEVEL_AXES = {"surface": "surface", "dbucket": "dbucket", "circ": "circ_tier", "elev": "elev_tier",
              "straight": "straight_tier", "turn": "turn_dir", "turf_type": "turf_type", "rest": "rest_bucket",
              "class_high": "class_high", "track": "track_tier", "going": "going2"}
TRAIT_AXES = ["front", "kire", "logd", "perf_adj"]   # perf_adj = 父の総合力(主仮説の基準モデル用、レーダーの軸ではない)
SIRE_FIRST_AXES = {"front", "kire", "logd", "perf_adj"}   # 事前登録 §3: 父の値(無ければ父父→母父)
MIN_HORSES_HALF = 10


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r3(x):
    return None if x is None else round(float(x), 3)


# ------------------------------------------------------------------ 閾値(固定)
def fixed_thresholds(runs: pd.DataFrame) -> dict:
    base = runs[runs["year"].between(*THRESH_YEARS)].drop_duplicates("race_id")
    th = {"track": {}, "pace": {}}
    for s, g in base.groupby("surface"):
        th["track"][s] = [float(x) for x in g["track_index_num"].quantile([1 / 3, 2 / 3])]
    for (s, b), g in base.groupby(["surface", "dbucket"]):
        th["pace"][f"{s}|{b}"] = [float(x) for x in g["pace"].quantile([1 / 3, 2 / 3])]
    return th


def add_tiers(runs: pd.DataFrame, th: dict) -> pd.DataFrame:
    runs = runs.copy()
    lo = runs["surface"].map(lambda s: th["track"][s][0])
    hi = runs["surface"].map(lambda s: th["track"][s][1])
    t = runs["track_index_num"]
    runs["track_tier"] = np.where(t.isna(), None, np.where(t <= lo, "高速", np.where(t >= hi, "時計かかる", "標準")))
    key = runs["surface"] + "|" + runs["dbucket"]
    plo = key.map(lambda k: th["pace"].get(k, [np.nan, np.nan])[0])
    phi = key.map(lambda k: th["pace"].get(k, [np.nan, np.nan])[1])
    p = runs["pace"]
    runs["pace_tier"] = np.where(p.isna(), None, np.where(p <= plo, "速い", np.where(p >= phi, "遅い", "平均")))
    return runs


# ------------------------------------------------------------------ 1走ごとの値(係数は fit で推定し apply に当てる)
def run_values(fit: pd.DataFrame, apply: pd.DataFrame) -> pd.DataFrame:
    out = apply.copy()
    keys = ["course_key", "first_corner", "um_bin"]
    mu = fit.groupby(keys)["front_raw"].mean().rename("front_mu")
    out = out.join(mu, on=keys)
    out["front_mu"] = out["front_mu"].fillna(out["first_corner"].map(fit.groupby("first_corner")["front_raw"].mean()))
    out["front_std"] = out["front_raw"] - out["front_mu"]
    fit = fit.join(mu, on=keys)
    fit["front_std"] = fit["front_raw"] - fit["front_mu"]
    out["front"] = np.nan
    out["kire"] = np.nan
    for s in ["芝", "ダ"]:
        Xf = lambda d: np.column_stack([np.ones(len(d)), d["perf"], d["perf"] ** 2])  # noqa: E731
        mf = (fit["surface"] == s) & fit["front_std"].notna()
        b, *_ = np.linalg.lstsq(Xf(fit[mf]), fit.loc[mf, "front_std"], rcond=None)
        ma = (out["surface"] == s) & out["front_std"].notna()
        out.loc[ma, "front"] = out.loc[ma, "front_std"].to_numpy() - Xf(out[ma]) @ b

        def Xk(d):
            p, f = d["perf"].to_numpy(), d["front4"].to_numpy()
            return np.column_stack([np.ones(len(d)), p, f, p * f, p * p, f * f])
        mf = (fit["surface"] == s) & fit["kire_raw"].notna() & fit["front4"].notna()
        b, *_ = np.linalg.lstsq(Xk(fit[mf]), fit.loc[mf, "kire_raw"], rcond=None)
        ma = (out["surface"] == s) & out["kire_raw"].notna() & out["front4"].notna()
        out.loc[ma, "kire"] = out.loc[ma, "kire_raw"].to_numpy() - Xk(out[ma]) @ b
    strata = ["age_band", "career_band", "class_change", "surface"]
    pm = fit.groupby(strata)["perf"].mean().rename("perf_mu")
    out = out.join(pm, on=strata)
    out["perf_adj"] = out["perf"] - out["perf_mu"].fillna(fit["perf"].mean())
    return out.drop(columns=["front_mu", "front_std", "perf_mu"])


def within_horse(runs: pd.DataFrame) -> pd.Series:
    g = runs.groupby("horse_id")["perf_adj"]
    n, s = g.transform("count"), g.transform("sum")
    return pd.Series(np.where(n >= 2, runs["perf_adj"] - (s - runs["perf_adj"]) / (n - 1), np.nan), index=runs.index)


def growth_values(runs: pd.DataFrame) -> pd.DataFrame:
    m = runs["date"].dt.month
    late = (runs["age"] >= 4) | ((runs["age"] == 3) & (m >= 7))
    a2 = runs[runs["age"] == 2].groupby("horse_id")["perf_adj"].agg(["mean", "count"])
    a3 = runs[late].groupby("horse_id")["perf_adj"].agg(["mean", "count"])
    j = a2.join(a3, lsuffix="_2", rsuffix="_3", how="inner")
    return pd.DataFrame({"x": j["mean_3"] - j["mean_2"], "n": np.minimum(j["count_2"], j["count_3"])})


# ------------------------------------------------------------------ 祖先表
def horse_level_values(runs: pd.DataFrame) -> dict:
    """{(axis, level): 馬単位の値(x, n)}。runs は run_values 済み。"""
    hv = {}
    for a in TRAIT_AXES:
        hv[(a, "-")] = C.horse_values(runs, a)
    c = within_horse(runs)
    rr = runs.assign(c=c)
    for axis, col in LEVEL_AXES.items():
        for lev, g in rr.dropna(subset=[col]).groupby(col):
            hv[(axis, lev)] = C.horse_values(g, "c")
    hv[("debut", "-")] = C.horse_values(rr[(rr["career"] == 0) & rr["is_debut_race"]], "c")
    hv[("growth", "-")] = growth_values(runs)
    return hv


def build_tables(runs: pd.DataFrame, ped: pd.DataFrame, horses=None, return_hv=False):
    if horses is not None:
        runs = runs[runs["horse_id"].isin(horses)]
    out, hvs = {}, {}
    for key, h in horse_level_values(runs).items():
        if len(h) < 20:
            continue
        hc = C.center_horse_values(h)
        hvs[key] = hc
        for role, rc in C.ROLES:
            out[key + (role,)] = C.ancestor_table(hc, ped, rc)
    return (out, hvs) if return_hv else out


def shrink_tables(tables: dict) -> tuple[dict, dict]:
    vals, meta = {}, {}
    for key, t in tables.items():
        if (t["n_horses"] >= 5).sum() < 10:
            continue
        k, within, tau2 = C.eb_k(t)
        vals[key] = C.shrink(t, k)
        meta[key] = {"k": k, "tau": float(np.sqrt(tau2)), "within": within}
    return vals, meta


def ped_value(df: pd.DataFrame, vals: dict, axis: str, level: str = "-") -> pd.Series:
    """事前登録 §3: 傾向の軸は父(無ければ父父→母父)、条件の軸は使える系統の等重み平均。"""
    cols = {}
    for role, rc in C.ROLES:
        v = vals.get((axis, level, role))
        cols[role] = df[rc].map(v) if v is not None else pd.Series(np.nan, index=df.index)
    if axis in SIRE_FIRST_AXES:
        return cols["sire"].fillna(cols["ss"]).fillna(cols["bms"])
    return pd.concat(cols.values(), axis=1).mean(axis=1)


def ped_value_levels(df: pd.DataFrame, vals: dict, axis: str, level_col: str) -> pd.Series:
    """行ごとに異なる区分(level_col の値)の血統値。"""
    out = pd.Series(np.nan, index=df.index)
    for lev, g in df.dropna(subset=[level_col]).groupby(level_col):
        out.loc[g.index] = ped_value(g, vals, axis, lev)
    return out


# ------------------------------------------------------------------ 内側検証(2020年まで)
def inner_validation(runs: pd.DataFrame, ped: pd.DataFrame) -> dict:
    runs = runs[runs["year"] <= 2020]
    assert runs["year"].max() <= 2020
    anc = run_values(runs[runs["year"] <= 2017], runs[runs["year"] <= 2017])
    allv = run_values(runs[runs["year"] <= 2017], runs)
    first_year = runs.groupby("horse_id")["year"].min()
    chk_h = set(first_year[first_year >= 2018].index)
    chk = allv[allv["horse_id"].isin(chk_h)].sort_values(["horse_id", "date"]).copy()
    tables = build_tables(anc, ped)
    vals, meta = shrink_tables(tables)
    hh = anc["horse_id"].drop_duplicates()
    half = hh.map(lambda h: int(hashlib.md5(h.encode()).hexdigest(), 16) % 2)
    T = [build_tables(anc, ped, horses=set(hh[half == i])) for i in (0, 1)]

    def e1_e4(axis, level):
        rec = {}
        t = tables.get((axis, level, "sire"))
        if t is not None:
            a = t[t["n_horses"] >= 30]["mean"]
            if len(a) >= 10:
                share, ratio = float((a > 0).mean()), float(abs(a.median()) / a.std())
                rec["E4"] = {"n_anc": int(len(a)), "share_pos": r3(share), "abs_median_over_sd": r3(ratio),
                             "pass": bool(0.35 <= share <= 0.65 and ratio < 0.25)}
        a0, a1 = T[0].get((axis, level, "sire")), T[1].get((axis, level, "sire"))
        if a0 is not None and a1 is not None:
            a0, a1 = a0[a0["n_horses"] >= MIN_HORSES_HALF], a1[a1["n_horses"] >= MIN_HORSES_HALF]
            cm = a0.index.intersection(a1.index)
            r, n = C.corr(a0.loc[cm, "mean"], a1.loc[cm, "mean"])
            rec["E1_sire_sb"], rec["E1_n_anc"] = r3(C.sb(r)), n
        m = meta.get((axis, level, "sire"))
        if m:
            rec["k_sire"], rec["tau_sire"] = round(m["k"], 2), round(m["tau"], 5)
        return rec

    g = chk.groupby("horse_id")["perf_adj"]
    prev_mean = (g.cumsum() - chk["perf_adj"]) / chk.groupby("horse_id").cumcount().replace(0, np.nan)
    chk["y_prev"] = chk["perf_adj"] - prev_mean
    R = {"anc_years": "2011-2017", "check": "2018-2020年デビュー馬", "n_check_horses": len(chk_h), "axes": {}}
    for axis, col in LEVEL_AXES.items():
        levels = sorted(chk[col].dropna().unique())
        rec = {"levels": {lev: e1_e4(axis, lev) for lev in levels}}
        prev = chk.groupby("horse_id")[col].shift(1)
        first = ~chk.duplicated(["horse_id", col]) & chk[col].notna()
        m = first & (chk["career"] >= 1) & prev.notna() & (prev != chk[col]) & chk["y_prev"].notna()
        d = chk[m].assign(prev_level=prev[m])
        d["y"] = d["y_prev"] - d.groupby(["prev_level", col])["y_prev"].transform("mean")
        pred = ped_value_levels(d, vals, axis, col) - ped_value_levels(d.assign(_p=d["prev_level"]), vals, axis, "_p")
        r, n = C.corr(pred, d["y"])
        rec["first_time"] = {"n_runs": n, "r": r3(r),
                             "ci": C.cluster_boot_corr(pred, d["y"], d["ped_S_horse_id"], seed=SEED) if n >= 30 else None}
        R["axes"][axis] = rec
    # 傾向の軸
    hv = chk.groupby("horse_id").agg(front=("front", "mean"), kire=("kire", "mean"), logd=("logd", "mean"),
                                     n=("race_id", "count"))
    hv = hv[hv["n"] >= 3].join(ped.set_index("horse_id"))
    for a in TRAIT_AXES:
        p = ped_value(hv, vals, a)
        r, n = C.corr(p, hv[a])
        R["axes"][a] = {"levels": {"-": e1_e4(a, "-")}, "horse": {"n": n, "r": r3(r),
                        "ci": C.cluster_boot_corr(p, hv[a], hv["ped_S_horse_id"], seed=SEED)}}
    # 新馬: 新馬戦の馬内差(2走以上)
    cc = within_horse(chk)
    dd = chk.assign(c=cc)
    dd = dd[(dd["career"] == 0) & dd["is_debut_race"] & dd["c"].notna()]
    p = ped_value(dd, vals, "debut")
    r, n = C.corr(p, dd["c"])
    R["axes"]["debut"] = {"levels": {"-": e1_e4("debut", "-")}, "horse": {"n": n, "r": r3(r),
                          "ci": C.cluster_boot_corr(p, dd["c"], dd["ped_S_horse_id"], seed=SEED)}}
    # 成長力: 2018年デビュー(2歳)で、2019年7月以降の走がある馬
    gv = growth_values(chk).join(ped.set_index("horse_id"))
    p = ped_value(gv, vals, "growth")
    r, n = C.corr(p, gv["x"])
    R["axes"]["growth"] = {"levels": {"-": e1_e4("growth", "-")}, "horse": {"n": n, "r": r3(r),
                           "ci": C.cluster_boot_corr(p, gv["x"], gv["ped_S_horse_id"], seed=SEED) if n >= 30 else None}}
    # 判定(事前登録 §3: E1≥0.3 かつ E4合格 かつ 確認の相関の95%下限>0)
    for axis, rec in R["axes"].items():
        lv = [v for v in rec["levels"].values() if v]
        e1 = [v["E1_sire_sb"] for v in lv if v.get("E1_sire_sb") is not None]
        e4 = [v["E4"]["pass"] for v in lv if "E4" in v]
        conf = rec.get("first_time") or rec.get("horse")
        rec["judge"] = {"E1_median": r3(np.median(e1)) if e1 else None, "E4_all_pass": bool(all(e4)) if e4 else None,
                        "E4_failed_levels": [k for k, v in rec["levels"].items() if v and "E4" in v and not v["E4"]["pass"]],
                        "ci_lower": conf["ci"][0] if conf and conf.get("ci") else None}
        j = rec["judge"]
        rec["judge"]["pass"] = bool(j["E1_median"] is not None and j["E1_median"] >= 0.3 and j["E4_all_pass"]
                                    and j["ci_lower"] is not None and j["ci_lower"] > 0)
    return R


# ------------------------------------------------------------------ 時点表
def save_tables(tables: dict, path: Path) -> dict:
    vals, meta = shrink_tables(tables)
    rows = []
    for key, t in tables.items():
        if key not in vals:
            continue
        axis, level, role = key
        d = t.reset_index().rename(columns={t.index.name or "index": "ancestor_id"})
        d.columns = ["ancestor_id"] + list(d.columns[1:])
        d["shrunk"] = vals[key].to_numpy()
        d["axis"], d["level"], d["role"] = axis, level, role
        d["k"], d["tau"] = meta[key]["k"], meta[key]["tau"]
        rows.append(d)
    df = pd.concat(rows, ignore_index=True)
    df.to_parquet(path, index=False)
    return {"n_rows": int(len(df)), "n_keys": len(vals)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inner", action="store_true")
    ap.add_argument("--tables", action="store_true")
    args = ap.parse_args()
    runs = pd.read_parquet(RUNS)
    th = fixed_thresholds(runs)
    runs = add_tiers(runs, th)
    ped = runs[["horse_id"] + [c for _, c in C.ROLES]].drop_duplicates("horse_id")
    if args.inner:
        t0 = time.time()
        R = inner_validation(runs, ped)
        R["thresholds"] = th
        R["elapsed_sec"] = round(time.time() - t0, 1)
        (C.OUT_DIR / "r2_inner_validation.json").write_text(json.dumps(R, ensure_ascii=False, indent=1,
                                                                        default=lambda o: None), encoding="utf-8")
        log("wrote r2_inner_validation.json")
    if args.tables:
        TABLE_DIR.mkdir(exist_ok=True)
        meta = {"thresholds": th, "tables": {}}
        for Y in list(range(2013, 2027)) + ["display"]:
            t0 = time.time()
            win = runs if Y == "display" else runs[runs["year"] <= Y - 1]
            v = run_values(win, win)
            tables, hvs = build_tables(v, ped, return_hv=True)
            info = save_tables(tables, TABLE_DIR / f"anc_{Y}.parquet")
            # 「血統のみ」の値から本人の走を除くため、その年(表示用は2026年)に走る馬の馬単位の値(xc, w)を保存
            part = set(runs.loc[runs["year"] == (2026 if Y == "display" else Y), "horse_id"])
            hrows = [hc.loc[hc.index.isin(part), ["xc", "w"]].assign(axis=k[0], level=k[1]).reset_index()
                     for k, hc in hvs.items()]
            pd.concat(hrows, ignore_index=True).to_parquet(TABLE_DIR / f"hv_{Y}.parquet", index=False)
            info.update({"window_end": str(win["date"].max().date()), "n_runs": int(len(win)),
                         "sec": round(time.time() - t0, 1)})
            meta["tables"][str(Y)] = info
            log(f"table {Y}: {info}")
        (TABLE_DIR / "tables_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
