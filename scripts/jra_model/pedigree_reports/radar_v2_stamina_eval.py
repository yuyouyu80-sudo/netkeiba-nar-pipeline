# -*- coding: utf-8 -*-
"""血統レーダー v2: スタミナの軸(A 持続力・B 距離延長への反応)の確認(事前登録 追補7、2026-10-07)。

追補7-3 で確認に進めた A・B の交差項が、C2(父の総合力+v2 の5成分)+主効果 を超えて着順(上位3頭の Plackett–Luce)を説明するか。
係数は毎年 2014〜Y−1年で推定し直す(R3b と同じ、`radar_v2_dl_condition_encoder.fold`・`pl_fit_eval` と同じ定義をここに写した。torch 不要)。

- 基準 M0: S + C2の5成分 + 主効果(z_A・z_B・延長・短縮の有無・延長の幅)
- 総合検定: M0 → M0 + [q_A·z_A, 延長の幅·z_B] の1レースあたり ΔLL、片側 p < 0.025(日付×競馬場のブロックブートストラップ 2,000回)
- 合格なら各交差項を1本ずつ: Holm m=2(片側 α=0.025)
- 副次(記述)・プラセボ・感度は追補7-4 のとおり。

使い方:
  python radar_v2_stamina_eval.py --final-dry   # 2018〜2020年で同じ処理(2021年以降は読まない)→ radar_v2/stamina_eval_dry.txt
  python radar_v2_stamina_eval.py --final       # 2021〜2026年(1回だけ、ガードあり)→ radar_v2/radar_v2_stamina_eval.json
"""
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_eval as E  # noqa: E402
import radar_v2_stamina as S  # noqa: E402

DRY = "--final-dry" in sys.argv
FINAL = "--final" in sys.argv
YEARS = range(2018, 2021) if DRY else range(2021, 2027)
TRAIN_START = 2014
OUT = C.OUT_DIR / ("stamina_eval_dry.txt" if DRY else "radar_v2_stamina_eval.json")
TAG = "radar-v2-prereg-addendum7"
PREREG = C.PROJECT_ROOT / "data/jra_pipeline/pedigree_reports/RADAR_V2_PREREG_2026_10_06.md"
COMPS = ["F_front", "F_kire", "F_surf", "F_db", "F_deb"]
C2 = ["S"] + COMPS
MAINS = ["z_stamA", "z_stamB", "ext", "short", "ext_amt"]
INTER = ["IA", "IB"]
N_PLACEBO = 20
SEED = 20261007
EXPECT_FINAL = {"n_races": 19209, "last_date": "2026-10-04"}
log = A.log


# ------------------------------------------------------------------ 特徴量
def own_stamina(runs: pd.DataFrame, R: pd.DataFrame, fx: dict, Y: int) -> pd.DataFrame:
    """その日より前の本人の走から、A(馬内の傾き)と B(同じ距離帯の延長 − それ以外)の値と情報量(係数は Y−1年末まで)。"""
    fit = runs[runs["year"] <= Y - 1]
    v = S.apply_run_adjust(S.add_run_cols(A.run_values(fit, runs[runs["year"] <= Y]), R), fx)
    g = v["horse_id"]
    ok = v["perf_A"].notna() & v["pace_hi"].notna()
    x, y, o = v["pace_hi"].where(ok, 0), v["perf_A"].where(ok, 0), ok.astype(float)
    cs = lambda s: s.groupby(g).cumsum() - s  # noqa: E731  その走を含まない累積
    n, sx, sy, sxy, sxx = cs(o), cs(x), cs(y), cs(x * y), cs(x * x)
    Sxx = sxx - sx ** 2 / n.replace(0, np.nan)
    Sxy = sxy - sx * sy / n.replace(0, np.nan)
    valid = (n >= 3) & (Sxx >= S.SXX_MIN)
    v["ownA"] = (Sxy / Sxx).where(valid).clip(*fx["clip_A"])
    v["ownA_n"] = Sxx.where(valid, 0).fillna(0)
    num, den = 0.0, 0.0
    okb = v["r_B"].notna()
    for b in S.BUCKET_ORDER:
        st = {}
        for e in (True, False):
            m = (okb & (v["dbucket"] == b) & (v["ext"] == e)).astype(float)
            st[e] = (cs(m), cs(v["r_B"].fillna(0) * m))
        ne, se_ = st[True]
        no, so = st[False]
        both = (ne > 0) & (no > 0)
        info = (ne * no / (ne + no).replace(0, np.nan)).where(both, 0).fillna(0)
        diff = (se_ / ne.replace(0, np.nan) - so / no.replace(0, np.nan)).where(both, 0).fillna(0)
        num, den = num + info * diff, den + info
    v["ownB"] = (num / den.replace(0, np.nan))
    v["ownB_n"] = den
    return v[v["year"] == Y].set_index(["race_id", "horse_id"])[["ownA", "ownA_n", "ownB", "ownB_n"]]


def k_own(runs: pd.DataFrame, R: pd.DataFrame, fx: dict) -> dict:
    """本人の値と血統の値を混ぜる k(馬内のばらつき / 馬の真の値のばらつき)。2018〜2020年の走、係数は2017年末まで(E.estimate_kh と同じ窓)。"""
    v = S.apply_run_adjust(S.add_run_cols(A.run_values(runs[runs["year"] <= 2017], runs[runs["year"].between(2018, 2020)]), R), fx)
    out = {}
    d = v[["horse_id", "perf_A", "pace_hi"]].dropna()
    gg = d.groupby("horse_id")
    mx, my = gg["pace_hi"].transform("mean"), gg["perf_A"].transform("mean")
    d = d.assign(sxy=(d["pace_hi"] - mx) * (d["perf_A"] - my), sxx=(d["pace_hi"] - mx) ** 2, syy=(d["perf_A"] - my) ** 2)
    s = d.groupby("horse_id").agg(sxy=("sxy", "sum"), sxx=("sxx", "sum"), syy=("syy", "sum"), n=("sxy", "size"))
    s = s[(s["n"] >= 4) & (s["sxx"] >= S.SXX_MIN)]
    b = s["sxy"] / s["sxx"]
    sig2 = float(((s["syy"] - b * s["sxy"]).sum()) / (s["n"] - 2).sum())
    tau2 = float(np.var(b) - np.mean(sig2 / s["sxx"]))
    out["A"] = sig2 / max(tau2, 1e-9)
    h = S.horse_ext(v)
    sigb = float(v.dropna(subset=["r_B"]).groupby(["horse_id", "dbucket", "ext"])["r_B"].var().mean())
    tau2b = float(np.var(h["x"]) - np.mean(sigb / h["n"]))
    out["B"] = sigb / max(tau2b, 1e-9)
    return out


def build(years) -> tuple[pd.DataFrame, dict]:
    fx = S.load_fixed()
    runs = S.load_base()
    last = max(years)
    if DRY:
        runs = runs[runs["year"] <= 2020]
        assert runs["year"].max() <= 2020
    R, _ = S.prepare(runs, fx)
    ST = S.stam_tables_class()
    dl = pd.read_parquet(C.OUT_DIR / "dl_table.parquet")
    dl = dl[dl["year"].between(TRAIN_START, last)]
    if DRY:
        assert dl["year"].max() <= 2020
    # E.features_for_year / add_own(S_own・F_own、(4) 用)
    base_runs = pd.read_parquet(A.RUNS)
    base_runs = A.add_tiers(base_runs, A.fixed_thresholds(base_runs))
    base_runs["umaban_num"] = pd.to_numeric(base_runs["umaban"], errors="coerce")
    if DRY:
        base_runs = base_runs[base_runs["year"] <= 2020]
    req = pd.read_parquet(C.OUT_DIR / "race_req.parquet")
    kh = E.estimate_kh(base_runs)
    ko = k_own(runs, R, fx)
    log("k_h", kh, "k_own", ko)
    flags = S.add_run_cols(runs[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan), R)
    flags = flags.set_index(["race_id", "horse_id"])[["ext", "short", "ext_amt"]]
    logd_mu = runs[runs["year"] <= 2020].groupby("surface")["logd"].mean()
    rinfo = runs.drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])[
        ["ped_DS_horse_id", "ped_SS_horse_id", "turn_dir", "career", "dbucket"]]   # surface・logd・class_ord・is_debut_race は dl_table にある
    frames = []
    for Y in range(TRAIN_START, last + 1):
        d = dl[dl["year"] == Y].join(rinfo, on=["race_id", "horse_id"])
        T, TS = E.Tables(Y), ST(Y)
        for ax in ["stamA", "stamB"]:
            d[f"z_{ax}"] = TS.z(d, ax, "-")
        d["tau_A"], d["tau_B"] = TS.tau("stamA"), TS.tau("stamB")
        d["z_logd"] = T.z(d, "logd", "-")
        d["z_turn"] = np.nan
        for lev in ["右回り", "左回り"]:
            m = d["turn_dir"] == lev
            d.loc[m, "z_turn"] = T.z(d[m], "turn", lev)
        f = E.add_own(base_runs, E.features_for_year(base_runs, req, Y), Y, kh)
        d = d.join(f.set_index(["race_id", "horse_id"])[["S_own", "F_own"]], on=["race_id", "horse_id"])
        d = d.join(own_stamina(runs, R, fx, Y), on=["race_id", "horse_id"])
        frames.append(d)
        log("features", Y, len(d))
    D = pd.concat(frames, ignore_index=True)
    D = D.join(flags, on=["race_id", "horse_id"]).join(R[["qA", "pace_hi"]], on="race_id")
    D["ext"] = D["ext"].fillna(False).astype(float)
    D["short"] = D["short"].fillna(False).astype(float)
    D["ext_amt"] = D["ext_amt"].fillna(0.0)
    D["IA"] = D["qA"] * D["z_stamA"]
    D["IB"] = D["ext_amt"] * D["z_stamB"]
    D["F_logd"] = D["z_logd"] * (D["logd"] - D["surface"].map(logd_mu))
    D["IA_turn"], D["IB_turn"] = D["qA"] * D["z_turn"], D["ext_amt"] * D["z_turn"]
    D["IA_real"] = D["pace_hi"] * D["z_stamA"]   # 参考(結果の情報、予測には使えない)
    # (4) 用: 本人の値と血統の値の合成(E.add_own と同じ形)
    for ax, own, n, k in [("A", "ownA", "ownA_n", ko["A"]), ("B", "ownB", "ownB_n", ko["B"])]:
        zo = (D[own] / D[f"tau_{ax}"]).where(D[n] > 0, 0).fillna(0)
        D[f"zo_{ax}"] = (D[n] * zo + k * D[f"z_stam{ax}"].fillna(0)) / (D[n] + k)
    D["IA_own"], D["IB_own"] = D["qA"] * D["zo_A"], D["ext_amt"] * D["zo_B"]
    return D, {"k_h": kh, "k_own": ko}


# ------------------------------------------------------------------ 評価
def pl_fit_eval(fit_df, test_df, feats):
    beta = E.PL(fit_df, feats).fit()
    t = E.PL(test_df, feats)
    return t.race_ll(beta), t.race_ids, beta


def rolling(D, feats, years, transform=None):
    """毎年 2014〜Y−1年で推定し Y年で評価した、レースごとの対数尤度(Series)と年ごとの β。"""
    lls, betas = [], {}
    for Y in years:
        tr, te = D[D["year"].between(TRAIN_START, Y - 1)], D[D["year"] == Y]
        if transform is not None:
            tr, te = transform(tr), transform(te)
        ll, rid, b = pl_fit_eval(tr, te, feats)
        lls.append(pd.Series(ll, index=rid))
        betas[Y] = [round(float(x), 4) for x in b]
    return pd.concat(lls), betas


def summarize(d: pd.Series, blk: pd.Series, name: str) -> dict:
    d = d.dropna()
    ci, p, se = E.block_boot(d.to_numpy(), blk.loc[d.index].to_numpy(), with_p=True)
    return {"name": name, "n_races": int(len(d)), "dll_per_race": round(float(d.mean()), 6), "ci95_block": ci,
            "p_one_sided": round(p, 5), "se_block": round(se, 7),
            "by_year": d.groupby(d.index.str[:4]).mean().round(6).to_dict(),
            "years_positive": int((d.groupby(d.index.str[:4]).mean() > 0).sum())}


def evaluate(D: pd.DataFrame, years) -> dict:
    t0 = time.time()
    blk = D.drop_duplicates("race_id").set_index("race_id")["block"]
    races = D.drop_duplicates("race_id").set_index("race_id")
    ll = {}
    betas = {}
    specs = {"C2": C2, "M0": C2 + MAINS, "M_full": C2 + MAINS + INTER, "M_A": C2 + MAINS + ["IA"], "M_B": C2 + MAINS + ["IB"],
             "M0_logd": C2 + MAINS + ["F_logd"], "M_full_logd": C2 + MAINS + ["F_logd"] + INTER,
             "O0": ["S_own", "F_own", "ext", "short", "ext_amt", "zo_A", "zo_B"],
             "O_full": ["S_own", "F_own", "ext", "short", "ext_amt", "zo_A", "zo_B", "IA_own", "IB_own"],
             "M_turn": C2 + MAINS + ["IA_turn", "IB_turn"], "M_realA": C2 + MAINS + ["IA_real"]}
    for k, f in specs.items():
        ll[k], betas[k] = rolling(D, f, years)
        log("rolling", k, round(float(ll[k].mean()), 5))
    res = {"years": list(years), "n_races_test": int(len(ll["M0"])),
           "last_test_date": str(D.loc[D["year"].isin(list(years)), "date"].max().date()),
           "coverage": {c: round(float(D.loc[D["year"].isin(list(years)), c].notna().mean()), 3)
                        for c in ["z_stamA", "z_stamB", "qA", "S_own", "zo_A", "z_logd", "z_turn"]}}
    # 主: 総合検定 → Holm m=2
    res["omnibus"] = summarize(ll["M_full"] - ll["M0"], blk, "総合: 主効果 → +[q_A·z_A, 延長の幅·z_B]")
    res["omnibus"]["pass"] = bool(res["omnibus"]["p_one_sided"] < 0.025)
    each = {"A": summarize(ll["M_A"] - ll["M0"], blk, "A: +q_A·z_A"), "B": summarize(ll["M_B"] - ll["M0"], blk, "B: +延長の幅·z_B")}
    res["each"] = each
    res["holm_m2"] = E.holm({k: v["p_one_sided"] for k, v in each.items()}) if res["omnibus"]["pass"] else \
        {k: {"p": v["p_one_sided"], "pass": False, "note": "総合検定が不合格のため判定しない"} for k, v in each.items()}
    # 副次(記述)
    d_full = ll["M_full"] - ll["M0"]
    sec = {"mains_only": summarize(ll["M0"] - ll["C2"], blk, "主効果だけ: C2 → C2+主効果"),
           "novelty_vs_logd": summarize(ll["M_full_logd"] - ll["M0_logd"], blk, "新規性: C2+主効果+F_logd → +交差項"),
           "own_4": summarize(ll["O_full"] - ll["O0"], blk, "(4) 血統+実績: S_own+F_own+主効果(本人合成) → +交差項(本人合成)"),
           "real_pace_A": summarize(ll["M_realA"] - ll["M0"], blk, "参考: 実際の pace_hi·z_A(結果の情報)")}
    cls = np.where(races["is_debut_race"].astype(bool), "新馬戦", np.where(races["class_ord"] == 0, "未勝利戦", "1勝クラス以上"))
    strata = {"class": pd.Series(cls, index=races.index)}
    longr = pd.Series(races["dbucket"] == "長距離(2201m~)", index=races.index)
    qt = races["qA"].rank(pct=True) > 2 / 3
    ext_r = D.groupby("race_id")["ext_amt"].max() > 0
    sec["strata"] = {}
    for lab, g in strata["class"].groupby(strata["class"]):
        sec["strata"][lab] = summarize(d_full[d_full.index.isin(g.index)], blk, lab)
    for lab, m in [("長距離帯(2201m〜)", longr), ("q_A 上位3分位のレース", qt), ("延長の馬を含むレース", ext_r)]:
        ids = m[m].index
        sec["strata"][lab] = summarize(d_full[d_full.index.isin(ids)], blk, lab)
    res["secondary"] = sec
    # プラセボ
    rng = np.random.default_rng(SEED)

    def perm_q(X):
        r = X.drop_duplicates("race_id")[["race_id", "surface", "qA"]].copy()
        for s, idx in r.groupby("surface").groups.items():
            r.loc[idx, "qA"] = r.loc[idx, "qA"].to_numpy()[rng.permutation(len(idx))]
        q = X["race_id"].map(r.set_index("race_id")["qA"])
        return X.assign(IA=q * X["z_stamA"])

    def perm_z(X):
        """同じレースの中だけで z_A・z_B を馬の間で入れ替える(主効果は元のまま、交差項だけ)。"""
        key = X["race_id"].to_numpy()
        order = np.lexsort((rng.random(len(X)), key))   # レース順・レース内はランダム
        srt = np.argsort(key, kind="stable")            # レース順・レース内は元の順
        src = np.empty(len(X), int)
        src[srt] = order
        return X.assign(IA=X["qA"].to_numpy() * X["z_stamA"].to_numpy()[src],
                        IB=X["ext_amt"].to_numpy() * X["z_stamB"].to_numpy()[src])
    def wrong_ext(X):
        """延長の幅を、同じ芝ダの別の馬の走のものに入れ替える(z_B との交差項だけ。主効果は元のまま)。"""
        e = X["ext_amt"].to_numpy().copy()
        for s, idx in X.groupby("surface").indices.items():
            e[idx] = e[idx][rng.permutation(len(idx))]
        return X.assign(IB=e * X["z_stamB"].to_numpy())
    # 軸ごとの「誤った要求」は、その軸の交差項だけを加えた ΔLL(each)と比べる。レース内の入れ替えは総合と比べる。
    plac = {"wrong_qA(芝ダ内でレースの要求を入れ替え、A と比較)": (perm_q, ["IA"], each["A"]["dll_per_race"]),
            "wrong_ext(芝ダ内で延長の幅を入れ替え、B と比較)": (wrong_ext, ["IB"], each["B"]["dll_per_race"]),
            "within_race_z(レース内で血統の値を入れ替え、総合と比較)": (perm_z, INTER, res["omnibus"]["dll_per_race"])}
    res["placebo"] = {}
    for name, (fn, extra, actual) in plac.items():
        vals = []
        for _ in range(N_PLACEBO):
            P = fn(D)
            l1, _ = rolling(P, C2 + MAINS + extra, years)
            vals.append(float((l1 - ll["M0"]).mean()))
        res["placebo"][name] = {"dll": [round(v, 6) for v in vals], "mean": round(float(np.mean(vals)), 6),
                                "max": round(float(np.max(vals)), 6), "actual": actual,
                                "actual_above_max": bool(actual > max(vals))}
        log("placebo", name, res["placebo"][name]["mean"], res["placebo"][name]["max"], "actual", actual)
    n_fit = float(np.mean([D[D["year"].between(TRAIN_START, Y - 1)]["race_id"].nunique() for Y in years]))
    res["placebo"]["negative_control_turn(回り)"] = summarize(ll["M_turn"] - ll["M0"], blk, "陰性対照: 回りの z で同じ交差項")
    res["placebo"]["expected_noise_2params"] = round(-2 / (2 * n_fit), 7)
    # 感度: 勝ち馬の父で約50群に分けた1群除外ジャックナイフ、勝ち馬の父の上位10頭への集中度
    win = D[D["pos"] == 1].drop_duplicates("race_id").set_index("race_id")["ped_S_horse_id"].fillna("na")
    sires = np.sort(win.unique())
    grp = pd.Series(np.random.default_rng(SEED).integers(0, 50, len(sires)), index=sires)
    wg = win.map(grp).reindex(d_full.index)
    jk = [float(d_full[wg != g].mean()) for g in range(50)]
    contrib = d_full.groupby(win.reindex(d_full.index)).sum().sort_values(ascending=False)
    res["sensitivity"] = {"jackknife_50_sire_groups": {"min": round(min(jk), 6), "max": round(max(jk), 6),
                                                         "se": round(float(np.sqrt(49 / 50 * np.sum((np.array(jk) - np.mean(jk)) ** 2))), 7)},
                          "top10_winner_sires_share_of_total_dll": round(float(contrib.head(10).sum() / d_full.sum()), 3) if d_full.sum() != 0 else None}
    res["betas"] = {k: betas[k] for k in ["M0", "M_full", "O_full"]}
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def power(res: dict) -> dict:
    """試運転の SE から、2021〜2026年(19,209レース)での検出力の概算(片側 α=0.025)。"""
    from scipy.stats import norm
    se = res["omnibus"]["se_block"] * np.sqrt(res["n_races_test"] / EXPECT_FINAL["n_races"])
    out = {"se_final_approx": round(float(se), 7)}
    for eff in [res["omnibus"]["dll_per_race"], 0.0002, 0.0003, 0.0005, 0.001]:
        out[f"power_at_{eff:.6f}"] = round(float(1 - norm.cdf(1.96 - eff / se)), 3)
    return out


# ------------------------------------------------------------------ ガード
def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip()


def input_shas() -> dict:
    T = A.TABLE_DIR
    return {"stamina_inner.json": E.file_sha([C.OUT_DIR / "stamina_inner.json"]),
            "stamina_fixed.json": E.file_sha([C.OUT_DIR / "stamina_fixed.json"]),
            "stam_tables": E.file_sha(sorted(T.glob("stam_anc_*.parquet")) + sorted(T.glob("stam_hv_*.parquet"))),
            "dl_table.parquet": E.file_sha([C.OUT_DIR / "dl_table.parquet"]),
            "runs_all.parquet": E.file_sha([A.RUNS]),
            "stamina_l3.parquet": E.file_sha([S.L3_CACHE])}


def guard_final():
    if OUT.exists():
        sys.exit(f"{OUT.name} は既にあります(2021〜2026年のスタミナの確認は1回だけ)。")
    dirty = git("status", "--porcelain", "--", "scripts/jra_model/pedigree_reports", str(PREREG.relative_to(C.PROJECT_ROOT)),
                "tests/test_radar_v2_stamina.py")
    if dirty:
        sys.exit(f"未コミットの変更があります:\n{dirty}")
    if not git("tag", "-l", TAG):
        sys.exit(f"タグ {TAG} がありません。")
    text = PREREG.read_text(encoding="utf-8")
    missing = [k for k, v in input_shas().items() if v not in text]
    if missing:
        sys.exit(f"入力の sha256 が追補7に記録された値と一致しません: {missing}")


def main():
    if not (DRY or FINAL):
        sys.exit("--final-dry か --final を指定してください。")
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    if FINAL:
        guard_final()
    t0 = time.time()
    D, meta = build(YEARS)
    res = evaluate(D, YEARS)
    if FINAL:
        assert res["n_races_test"] == EXPECT_FINAL["n_races"], res["n_races_test"]
        assert res["last_test_date"] == EXPECT_FINAL["last_date"], res["last_test_date"]
    else:
        assert max(YEARS) <= 2020 and D["year"].max() <= 2020
        res["power"] = power(res)
    res["meta"] = meta
    res["manifest"] = E.manifest() | {"input_shas": input_shas(), "tag": TAG, "tag_commit": git("rev-list", "-n", "1", TAG)}
    res["elapsed_total_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    E.append_ledger({"script": "radar_v2_stamina_eval.py", "mode": "dry (2018-2020)" if DRY else "confirm (追補7)",
                     "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "git_commit": res["manifest"]["git_commit"], "git_dirty_files": res["manifest"]["git_dirty_files"],
                     "output": OUT.name, "output_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()})
    log("omnibus", res["omnibus"]["dll_per_race"], res["omnibus"]["ci95_block"], "p", res["omnibus"]["p_one_sided"])


if __name__ == "__main__":
    main()
