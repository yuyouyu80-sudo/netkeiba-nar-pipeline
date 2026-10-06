# -*- coding: utf-8 -*-
"""血統レーダー v2 の R−0.5(R−1の修正と設計確定、2026-10-06)。

第2版プラン(C:/Users/yuyou/.claude/plans/wiggly-sniffing-metcalfe.md)のR−0.5。**2020年までのデータだけを読む**
(2021年以降は R4 の確認に1回だけ使うため、ここでは読み込まない)。
- 祖先の値: 2011〜2017年の走(ANC_END)。確認: 2018〜2020年にデビューした馬(祖先の値に本人の走が入らない)。
- 3系統の重み: 2018〜2019年デビュー馬で推定し、2020年デビュー馬で確認。
- 出力: data/jra_pipeline/pedigree_reports/radar_v2/r05_result.json

## 軸の定義(R−0.5で比較する候補)
- 脚質 front_v2: 前半位置 −(コース×最初のコーナー番号×馬番位置の平均)を、着順perf(1次・2次)で回帰した残差。
- キレ kire_v2: 上がり3F順位 − (着順・4角位置とその2次・交差項による予測)。
- perf_adj: perf −(年齢帯×キャリア段階×クラス変化×芝ダ の平均)。距離系の候補はこれを使う。
- 距離の候補:
  D1 平均距離: 馬の出走距離(log)の平均。調教師の選択を含む(市場も知っている情報)。
  D2 距離傾き: 馬内で perf_adj を log距離に回帰した傾き(3走以上・距離が2種以上の馬)。
  D3 距離帯の馬内差: perf_adj − 同じ馬の他の走の平均、を距離帯ごとに馬単位で集計(馬単位で中心化)。
  評価は「その距離帯を初めて走った回」(キャリア2走目以降)の、それまでの走の平均との差で行う。
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402

MAX_YEAR = 2020
ANC_END = 2017
CHECK_FROM = 2018
W_FIT_YEARS = (2018, 2019)   # 3系統の重みの推定
W_EVAL_YEAR = 2020           # 3系統の重みの確認
MIN_HORSES_HALF = 10
V1_MIN_N = 10
SEED = 20261006
OUT_JSON = C.OUT_DIR / "r05_result.json"
SURFACES = ["芝", "ダ"]
BUCKETS = ["短距離(~1400m)", "マイル(1401-1800m)", "中距離(1801-2200m)", "長距離(2201m~)"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def r3(x):
    return None if x is None else round(float(x), 3)


# ------------------------------------------------------------------ 1走ごとの v2 値(係数は ANC_END まで)
def add_run_values(runs: pd.DataFrame) -> pd.DataFrame:
    fit = runs["year"] <= ANC_END
    keys = ["course_key", "first_corner", "um_bin"]
    mu = runs[fit].groupby(keys)["front_raw"].mean().rename("front_mu")
    runs = runs.join(mu, on=keys)
    mu2 = runs[fit].groupby(["first_corner"])["front_raw"].mean()
    runs["front_mu"] = runs["front_mu"].fillna(runs["first_corner"].map(mu2))
    runs["front_std"] = runs["front_raw"] - runs["front_mu"]
    runs["front_v2"] = np.nan
    runs["kire_v2"] = np.nan
    for s in ["芝", "ダ"]:
        # 脚質: 着順で説明できる分を除く
        m = (runs["surface"] == s) & runs["front_std"].notna()
        Xf = lambda d: np.column_stack([np.ones(len(d)), d["perf"], d["perf"] ** 2])  # noqa: E731
        b, *_ = np.linalg.lstsq(Xf(runs[m & fit]), runs.loc[m & fit, "front_std"], rcond=None)
        runs.loc[m, "front_v2"] = runs.loc[m, "front_std"].to_numpy() - Xf(runs[m]) @ b
        # キレ: 着順と4角位置で説明できる分を除く
        m = (runs["surface"] == s) & runs["kire_raw"].notna() & runs["front4"].notna()

        def Xk(d):
            p, f = d["perf"].to_numpy(), d["front4"].to_numpy()
            return np.column_stack([np.ones(len(d)), p, f, p * f, p * p, f * f])
        b, *_ = np.linalg.lstsq(Xk(runs[m & fit]), runs.loc[m & fit, "kire_raw"], rcond=None)
        runs.loc[m, "kire_v2"] = runs.loc[m, "kire_raw"].to_numpy() - Xk(runs[m]) @ b
    strata = ["age_band", "career_band", "class_change", "surface"]
    pm = runs[fit].groupby(strata)["perf"].mean().rename("perf_mu")
    runs = runs.join(pm, on=strata)
    runs["perf_adj"] = runs["perf"] - runs["perf_mu"].fillna(runs.loc[fit, "perf"].mean())
    return runs


def within_horse(runs: pd.DataFrame) -> pd.DataFrame:
    g = runs.groupby("horse_id")["perf_adj"]
    n, s = g.transform("count"), g.transform("sum")
    out = runs.assign(c=np.where(n >= 2, runs["perf_adj"] - (s - runs["perf_adj"]) / (n - 1), np.nan))
    return out


def horse_slope(runs: pd.DataFrame, min_runs=3) -> pd.DataFrame:
    d = runs.dropna(subset=["perf_adj", "logd"])
    g = d.groupby("horse_id")
    n = g.size()
    mx, my = g["logd"].transform("mean"), g["perf_adj"].transform("mean")
    d = d.assign(dx=d["logd"] - mx, dy=d["perf_adj"] - my)
    sxx = (d["dx"] ** 2).groupby(d["horse_id"]).sum()
    sxy = (d["dx"] * d["dy"]).groupby(d["horse_id"]).sum()
    ok = (n >= min_runs) & (sxx > 1e-4)
    return pd.DataFrame({"x": (sxy / sxx)[ok], "n": n[ok]})


# ------------------------------------------------------------------ 祖先の値
def build_tables(anc_runs: pd.DataFrame, ped: pd.DataFrame, horses=None):
    """軸ごとの祖先表(各ロール)。horses を渡すとその馬だけで作る(E1の分割用)。"""
    if horses is not None:
        anc_runs = anc_runs[anc_runs["horse_id"].isin(horses)]
    hv = {}
    hv["front"] = C.horse_values(anc_runs, "front_v2")
    hv["kire"] = C.horse_values(anc_runs, "kire_v2")
    hv["D1"] = C.horse_values(anc_runs, "logd")
    hv["D2"] = horse_slope(anc_runs)
    wh = within_horse(anc_runs)
    for b in BUCKETS:
        hv[f"D3|{b}"] = C.horse_values(wh[wh["dbucket"] == b], "c")
    for sfc in SURFACES:
        hv[f"S|{sfc}"] = C.horse_values(wh[wh["surface"] == sfc], "c")
    tables = {}
    for axis, h in hv.items():
        hc = C.center_horse_values(h)
        for role, rc in C.ROLES:
            tables[(axis, role)] = C.ancestor_table(hc, ped, rc)
    return tables


def v1_table(anc_runs: pd.DataFrame, ped: pd.DataFrame, role_col: str) -> pd.DataFrame:
    d = anc_runs.dropna(subset=[role_col])  # 祖先IDは結合済み
    g = d.groupby(role_col)
    out = pd.DataFrame({"overall": g["is_win"].mean()})
    for name, mask in [("lead", d["leader"] == True), ("non", d["leader"] == False)]:  # noqa: E712
        gg = d[mask].groupby(role_col)["is_win"]
        out[f"{name}_wr"], out[f"{name}_n"] = gg.mean(), gg.size()
    out["front"] = np.where(out["lead_n"] >= V1_MIN_N, (out["lead_wr"] - out["overall"]) * 100, np.nan)
    out["kire"] = np.where((out["lead_n"] >= V1_MIN_N) & (out["non_n"] >= V1_MIN_N),
                           (out["non_wr"] - out["lead_wr"]) * 100, np.nan)
    # v1+縮約: 先頭時勝率の差を二項の標準誤差で経験ベイズ縮約
    for col, p, n in [("front", "lead_wr", "lead_n")]:
        e = out[col]
        se2 = (out[p] * (1 - out[p]) / out[n]) * 1e4
        tau2 = max(float(np.nanvar(e) - np.nanmean(se2[e.notna()])), 1e-9)
        out[f"{col}_shrunk"] = e * tau2 / (tau2 + se2)
    se2k = ((out["non_wr"] * (1 - out["non_wr"]) / out["non_n"]) + (out["lead_wr"] * (1 - out["lead_wr"]) / out["lead_n"])) * 1e4
    tau2 = max(float(np.nanvar(out["kire"]) - np.nanmean(se2k[out["kire"].notna()])), 1e-9)
    out["kire_shrunk"] = out["kire"] * tau2 / (tau2 + se2k)
    for b in BUCKETS:
        gb = d[d["dbucket"] == b].groupby(role_col)["is_win"]
        wr, n = gb.mean().reindex(out.index), gb.size().reindex(out.index).fillna(0)
        out[f"D3|{b}"] = np.where(n >= V1_MIN_N, (wr - out["overall"]) * 100, np.nan)
    return out


PREC = {}  # (axis, role) -> 事後の精度(1/分散)


def shrunk_values(tables):
    vals, ks = {}, {}
    for (axis, role), t in tables.items():
        k, within, tau2 = C.eb_k(t)
        vals[(axis, role)] = C.shrink(t, k)
        PREC[(axis, role)] = (t["sw"] + k) / within  # 事後分散 = within/(sw+k)
        ks[f"{axis}/{role}"] = {"k": round(k, 2), "tau": round(float(np.sqrt(tau2)), 5)}
    return vals, ks


def ped_value(df: pd.DataFrame, vals, axis: str, weights=None, roles=("sire", "bms", "ss")):
    """馬(行)ごとの血統値。weights=None なら使える系統の単純平均、dictなら加重(欠損は親=0)。"""
    cols = []
    for role, rc in C.ROLES:
        if role in roles:
            cols.append(df[rc].map(vals[(axis, role)]).rename(role))
    m = pd.concat(cols, axis=1)
    if weights == "precision":
        pw = pd.concat([df[rc].map(PREC[(axis, role)]).rename(role) for role, rc in C.ROLES if role in roles], axis=1)
        pw = pw.where(m.notna())
        return (m * pw).sum(axis=1) / pw.sum(axis=1)
    if weights is None:
        return m.mean(axis=1)
    return sum(m[r].fillna(0) * w for r, w in weights.items() if r in m) + weights.get("const", 0)


# ------------------------------------------------------------------ メイン
def main():
    t0 = time.time()
    cache = C.OUT_DIR / f"runs_le{MAX_YEAR}.parquet"
    if cache.exists():
        runs = pd.read_parquet(cache)
        meta = json.loads((C.OUT_DIR / f"runs_le{MAX_YEAR}_meta.json").read_text(encoding="utf-8"))
    else:
        runs, meta = C.load_runs(max_year=MAX_YEAR)
        runs.to_parquet(cache, index=False)
        (C.OUT_DIR / f"runs_le{MAX_YEAR}_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    assert runs["year"].max() <= MAX_YEAR
    log(f"出走行 {len(runs)}、{meta}")
    ped = C.load_ped_ids(runs["horse_id"].unique().tolist(), C.OUT_DIR / "ped_ids_v2.parquet")
    runs = runs.merge(ped, on="horse_id", how="left")
    runs = add_run_values(runs)
    anc_runs = runs[runs["year"] <= ANC_END]
    first_year = runs.groupby("horse_id")["year"].min()
    check_horses = set(first_year[first_year >= CHECK_FROM].index)
    chk = runs[runs["horse_id"].isin(check_horses)].copy()
    R = {"meta": meta, "n_runs": int(len(runs)), "anc_years": f"2011-{ANC_END}", "check_debut": f"{CHECK_FROM}-{MAX_YEAR}",
         "n_check_horses": len(check_horses),
         "first_corner_share": runs["first_corner"].value_counts(normalize=True).round(3).to_dict(),
         "front_missing": r3(runs["front_raw"].isna().mean()), "kire_missing": r3(runs["kire_v2"].isna().mean())}

    # ---------- 祖先表(全体)と縮約
    tables = build_tables(anc_runs, ped)
    vals, ks = shrunk_values(tables)
    R["eb"] = ks
    v1 = {role: v1_table(anc_runs, ped, rc) for role, rc in C.ROLES}

    # ---------- E4: 祖先値の符号の偏り(縮約前、産駒30頭以上)
    e4 = {}
    for (axis, role), t in tables.items():
        if role != "sire":
            continue
        a = t[t["n_horses"] >= 30]["mean"]
        share = float((a > 0).mean())
        ratio = float(abs(a.median()) / a.std())
        e4[axis] = {"n_anc": int(len(a)), "share_pos": r3(share), "abs_median_over_sd": r3(ratio),
                    "pass": bool(0.35 <= share <= 0.65 and ratio < 0.25)}
    for axis in ["front", "kire"] + [f"D3|{b}" for b in BUCKETS]:
        a = v1["sire"][axis].dropna()
        e4[f"v1:{axis}"] = {"n_anc": int(len(a)), "share_pos": r3((a > 0).mean()),
                            "abs_median_over_sd": r3(abs(a.median()) / a.std())}
    R["E4_ancestor"] = e4

    # ---------- E1: 産駒の2分割(縮約前)、v1と同じ祖先集合で
    hh = anc_runs["horse_id"].drop_duplicates()
    half = hh.map(lambda h: int(hashlib.md5(h.encode()).hexdigest(), 16) % 2)
    halves = [set(hh[half == i]) for i in (0, 1)]
    T = [build_tables(anc_runs, ped, horses=h) for h in halves]
    V1 = [{role: v1_table(anc_runs[anc_runs["horse_id"].isin(h)], ped, rc) for role, rc in C.ROLES} for h in halves]
    e1 = {}
    for axis in ["front", "kire", "D1", "D2"] + [f"D3|{b}" for b in BUCKETS] + [f"S|{x}" for x in SURFACES]:
        for role, _ in C.ROLES:
            a0, a1 = T[0][(axis, role)], T[1][(axis, role)]
            a0, a1 = a0[a0["n_horses"] >= MIN_HORSES_HALF], a1[a1["n_horses"] >= MIN_HORSES_HALF]
            common = a0.index.intersection(a1.index)
            r2, n2 = C.corr(a0.loc[common, "mean"], a1.loc[common, "mean"])
            rec = {"v2_sb": r3(C.sb(r2)), "n_anc": n2}
            if axis in V1[0][role].columns:
                b0, b1 = V1[0][role][axis].dropna(), V1[1][role][axis].dropna()
                cs = common.intersection(b0.index).intersection(b1.index)
                r1c, n1c = C.corr(b0.loc[cs], b1.loc[cs])
                r2c, _ = C.corr(a0.loc[cs, "mean"], a1.loc[cs, "mean"])
                rec.update({"same_set_n": n1c, "same_set_v1_sb": r3(C.sb(r1c)), "same_set_v2_sb": r3(C.sb(r2c))})
            e1[f"{axis}/{role}"] = rec
    R["E1"] = e1

    # ---------- E2: 2018〜2020年デビュー馬の実際の走り(脚質・キレ・D1・D2)、種牡馬クラスタのブートストラップ
    hv_chk = pd.DataFrame({"front": chk.groupby("horse_id")["front_v2"].mean(),
                           "kire": chk.groupby("horse_id")["kire_v2"].mean(),
                           "D1": chk.groupby("horse_id")["logd"].mean(),
                           "n": chk.groupby("horse_id").size(),
                           "debut_year": chk.groupby("horse_id")["year"].min()})
    hv_chk = hv_chk.join(horse_slope(chk).rename(columns={"x": "D2", "n": "n_d2"}))
    hv_chk = hv_chk.join(ped.set_index("horse_id"))
    e2 = {}
    sub = hv_chk[hv_chk["n"] >= 3]
    for axis in ["front", "kire", "D1", "D2"]:
        y = sub[axis]
        p_eq = ped_value(sub, vals, axis)
        p_sire = ped_value(sub, vals, axis, roles=("sire",))
        rec = {"n": int(y.notna().sum())}
        r, n = C.corr(p_eq, y)
        rec["v2_3line_r"] = r3(r)
        rec["v2_3line_ci"] = C.cluster_boot_corr(p_eq, y, sub["ped_S_horse_id"], seed=SEED)
        r, _ = C.corr(p_sire, y)
        rec["v2_sire_r"] = r3(r)
        if axis in ("front", "kire"):
            for lab, col in [("v1", axis), ("v1_shrunk", f"{axis}_shrunk")]:
                p1 = pd.concat([sub[rc].map(v1[role][col]) for role, rc in C.ROLES], axis=1).mean(axis=1)
                r, _ = C.corr(p1, y)
                rec[f"{lab}_r"] = r3(r)
                rec[f"v2_minus_{lab}_ci"] = C.cluster_boot_corr_diff(p_eq, p1, y, sub["ped_S_horse_id"], seed=SEED)
        e2[axis] = rec
    # D2 の目的変数そのものの信頼性(馬の走を奇数・偶数に分けた傾きの一致、6走以上)
    c6 = chk[chk["horse_id"].isin(sub[sub["n"] >= 6].index)].sort_values(["horse_id", "date"])
    c6 = c6.assign(k=c6.groupby("horse_id").cumcount() % 2)
    s0, s1 = horse_slope(c6[c6["k"] == 0], 3)["x"], horse_slope(c6[c6["k"] == 1], 3)["x"]
    r_tt, n_tt = C.corr(s0.reindex(s1.index), s1)
    e2["D2"]["target_split_half_r"] = r3(r_tt)
    e2["D2"]["target_split_half_n"] = n_tt
    R["E2_horse"] = e2

    # ---------- E2: 距離帯の初出走(キャリア2走目以降)。それまでの走の平均との差を、各候補で当てる
    chk = chk.sort_values(["horse_id", "date"])
    g = chk.groupby("horse_id")["perf_adj"]
    prev_mean = (g.cumsum() - chk["perf_adj"]) / chk.groupby("horse_id").cumcount().replace(0, np.nan)
    chk["y_first"] = chk["perf_adj"] - prev_mean
    chk["prev_bucket"] = chk.groupby("horse_id")["dbucket"].shift(1)
    fb = chk[chk["first_in_band"] & (chk["career"] >= 1) & chk["y_first"].notna()].copy()
    # 中心化(層: 新旧の距離帯の組み合わせ)
    fb["y_first"] = fb["y_first"] - fb.groupby(["prev_bucket", "dbucket"])["y_first"].transform("mean")
    p3 = pd.Series(np.nan, index=fb.index)
    for b in BUCKETS:
        for bp in BUCKETS:
            m = (fb["dbucket"] == b) & (fb["prev_bucket"] == bp)
            if m.any():
                p3[m] = ped_value(fb[m], vals, f"D3|{b}") - ped_value(fb[m], vals, f"D3|{bp}")
    p2 = ped_value(fb, vals, "D2") * (fb["logd"] - fb["prev_logd"])
    d1 = ped_value(fb, vals, "D1")  # 中心化済みなので、全体平均 log 距離を足し戻す
    d1_abs = d1 + float(C.center_horse_values(C.horse_values(anc_runs, "logd"))["mu"].iloc[0])
    p1 = (fb["prev_logd"] - d1_abs).abs() - (fb["logd"] - d1_abs).abs()  # 最適距離に近づいたら正
    fbr = {"n_runs": int(len(fb))}
    for lab, p in [("D3_bucket_diff", p3), ("D2_slope_x_change", p2), ("D1_closer_to_mean_distance", p1)]:
        r, n = C.corr(p, fb["y_first"])
        fbr[lab] = {"r": r3(r), "n": n, "ci": C.cluster_boot_corr(p, fb["y_first"], fb["ped_S_horse_id"], seed=SEED)}
    R["E2_first_in_band"] = fbr

    # ---------- E2: 芝ダの初出走(キャリア2走目以降、初芝・初ダ)
    chk["prev_surface"] = chk.groupby("horse_id")["surface"].shift(1)
    chk["first_in_surface"] = ~chk.duplicated(["horse_id", "surface"])
    fs = chk[chk["first_in_surface"] & (chk["career"] >= 1) & chk["y_first"].notna()].copy()
    fs["y_first"] = fs["y_first"] - fs.groupby(["prev_surface", "surface"])["y_first"].transform("mean")
    ps = pd.Series(np.nan, index=fs.index)
    for sfc in SURFACES:
        for sp in SURFACES:
            m = (fs["surface"] == sfc) & (fs["prev_surface"] == sp)
            if m.any():
                ps[m] = ped_value(fs[m], vals, f"S|{sfc}") - ped_value(fs[m], vals, f"S|{sp}")
    fsr = {"n_runs": int(len(fs))}
    for lab, roles, wt in [("3line", ("sire", "bms", "ss"), None), ("sire_only", ("sire",), None),
                           ("precision", ("sire", "bms", "ss"), "precision")]:
        pp = pd.Series(np.nan, index=fs.index)
        for sfc in SURFACES:
            for sp in SURFACES:
                m = (fs["surface"] == sfc) & (fs["prev_surface"] == sp)
                if m.any():
                    pp[m] = ped_value(fs[m], vals, f"S|{sfc}", roles=roles, weights=wt) - ped_value(fs[m], vals, f"S|{sp}", roles=roles, weights=wt)
        r, n = C.corr(pp, fs["y_first"])
        fsr[lab] = {"r": r3(r), "n": n, "ci": C.cluster_boot_corr(pp, fs["y_first"], fs["ped_S_horse_id"], seed=SEED)}
    R["E2_first_in_surface"] = fsr
    # 距離帯の初出走も父のみで
    p3s = pd.Series(np.nan, index=fb.index)
    for b in BUCKETS:
        for bp in BUCKETS:
            m = (fb["dbucket"] == b) & (fb["prev_bucket"] == bp)
            if m.any():
                p3s[m] = ped_value(fb[m], vals, f"D3|{b}", roles=("sire",)) - ped_value(fb[m], vals, f"D3|{bp}", roles=("sire",))
    p3p = pd.Series(np.nan, index=fb.index)
    for b in BUCKETS:
        for bp in BUCKETS:
            m = (fb["dbucket"] == b) & (fb["prev_bucket"] == bp)
            if m.any():
                p3p[m] = ped_value(fb[m], vals, f"D3|{b}", weights="precision") - ped_value(fb[m], vals, f"D3|{bp}", weights="precision")
    r, n = C.corr(p3p, fb["y_first"])
    R["E2_first_in_band"]["D3_bucket_diff_precision"] = {"r": r3(r), "n": n, "ci": C.cluster_boot_corr(p3p, fb["y_first"], fb["ped_S_horse_id"], seed=SEED)}
    r, n = C.corr(p3s, fb["y_first"])
    R["E2_first_in_band"]["D3_bucket_diff_sire_only"] = {"r": r3(r), "n": n, "ci": C.cluster_boot_corr(p3s, fb["y_first"], fb["ped_S_horse_id"], seed=SEED)}

    # ---------- 3系統の重み(2018〜2019デビューで推定、2020デビューで確認)
    wres = {}
    for axis in ["front", "kire", "D1"]:
        fit = sub[sub["debut_year"].isin(W_FIT_YEARS)]
        ev = sub[sub["debut_year"] == W_EVAL_YEAR]
        Xf = np.column_stack([fit[rc].map(vals[(axis, role)]).fillna(0) for role, rc in C.ROLES] + [np.ones(len(fit))])
        yf = fit[axis].to_numpy()
        ok = ~np.isnan(yf)
        beta, *_ = np.linalg.lstsq(Xf[ok], yf[ok], rcond=None)
        w = {role: float(beta[i]) for i, (role, _) in enumerate(C.ROLES)}
        w["const"] = float(beta[-1])
        r_eq, _ = C.corr(ped_value(ev, vals, axis), ev[axis])
        r_w, _ = C.corr(ped_value(ev, vals, axis, weights=w), ev[axis])
        r_s, _ = C.corr(ped_value(ev, vals, axis, roles=("sire",)), ev[axis])
        r_p, _ = C.corr(ped_value(ev, vals, axis, weights="precision"), ev[axis])
        wres[axis] = {"weights": {k: round(v, 3) for k, v in w.items()}, "eval_2020_r_equal": r3(r_eq),
                      "eval_2020_r_weighted": r3(r_w), "eval_2020_r_sire_only": r3(r_s), "eval_2020_r_precision": r3(r_p), "n_eval": int(ev[axis].notna().sum())}
    R["three_line_weights"] = wres

    # ---------- E4b: レース基準(2018〜2020年のレース、全出走馬)。v1もv2もリフトで比べる
    races = runs[runs["year"] >= CHECK_FROM].copy()
    lift = {}
    for axis in ["front", "kire"]:
        races["p2"] = ped_value(races, vals, axis)
        races["p1"] = pd.concat([races[rc].map(v1[role][axis]) for role, rc in C.ROLES], axis=1).mean(axis=1)
        top = races["pos"] <= 3
        rec = {}
        for lab, col in [("v1", "p1"), ("v2", "p2")]:
            L = (races[top].groupby("race_id")[col].mean() - races.groupby("race_id")[col].mean()).dropna()
            sd = float(races.groupby("race_id")[col].std().mean())
            ck = races.groupby("race_id")["course_key"].first().reindex(L.index)
            gm = L.groupby(ck).agg(["mean", "count", "std"])
            gm = gm[gm["count"] >= 30]
            se2 = (gm["std"] / np.sqrt(gm["count"])) ** 2
            wgt = 1 / se2
            grand = float(np.sum(wgt * gm["mean"]) / np.sum(wgt))
            Q = float(np.sum(wgt * (gm["mean"] - grand) ** 2))
            dfree = len(gm) - 1
            yr = L.index.str[:4].astype(int)
            e = L[yr <= 2019].groupby(ck[yr <= 2019]).agg(["mean", "count"])
            lt = L[yr == 2020].groupby(ck[yr == 2020]).agg(["mean", "count"])
            j = e.join(lt, lsuffix="_e", rsuffix="_l", how="inner")
            j = j[(j["count_e"] >= 30) & (j["count_l"] >= 15)]
            r_el, n_el = C.corr(j["mean_e"], j["mean_l"])
            rec[lab] = {"n_races": int(len(L)), "share_lift_pos": r3((L > 0).mean()), "mean_lift_in_sd": r3(L.mean() / sd),
                        "I2_inv_var": r3(max(0.0, (Q - dfree) / Q)) if Q > 0 else None, "n_course_keys": int(len(gm)),
                        "E3_2018_19_vs_2020_r": r3(r_el), "E3_n": n_el}
        lift[axis] = rec
    R["E4_race_lift"] = lift
    R["elapsed_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(R, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
