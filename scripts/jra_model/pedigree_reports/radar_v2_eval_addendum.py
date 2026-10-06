# -*- coding: utf-8 -*-
"""血統レーダー v2 の R4 追補(2026-10-06、**R4本番の結果を見た後に計算した探索。確認検証ではない**)。

R4本番(radar_v2_eval.json、commit 8610183b)のあと、DLエンジニア・シニアエンジニアの2体のレビューで指摘された点を計算する。
2021〜2026年はすでに確認に1回使っているので、ここでの数値は採否の確認には使わず、結論の文面と表示の判断材料にする(事前登録の追補3)。

1. 本番の主仮説の再現(同じ D から ΔLL が radar_v2_eval.json と一致することを assert)
2. 主仮説の軸ごとの内訳(1軸だけ・1軸を抜く・5軸の β を自由に推定)、長距離帯を除いた F、クラス別の内訳、年の符号検定
3. 副次(4)「血統+実績」の別の形(基準を [S, 本人の過去平均, 過去走の有無] にする、近走3走の成績を基準に入れる、市場を入れる)
4. 効果量の翻訳(一様・S・S+F・市場の対数尤度、McFadden の擬似R²、1着の段の確率、1位予想の的中率)
5. プラセボの判定の見直し(§6 の文言どおりの判定、雑音1変数を足したときの期待値 −1/(2·N_fit))
6. E4(本番用の時点表 2021〜2026年・表示用、全採用軸と総合力、産駒30頭以上の父。v1 も同じ基準で)
7. E1(2026年の時点表と同じ窓=2025年末まで、産駒を2分割した SB、産駒数の帯別、v1・v1+縮約・v2(縮約前)・v2(縮約後))
8. レース基準のコース間異質性 I²(2021〜2026年のリフト、逆分散重み、v2のみ)
出力: radar_v2/radar_v2_eval_addendum.json
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
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_eval as E  # noqa: E402

assert not E.DRY
OUT = C.OUT_DIR / "radar_v2_eval_addendum.json"
MAIN = json.loads((C.OUT_DIR / "radar_v2_eval.json").read_text(encoding="utf-8"))
LONG = E.BUCKETS[3]
log = E.log


def r5(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 5)


def strip(c):
    return {k: v for k, v in c.items() if not k.startswith("_") and k not in ("ci95_year_block",)}


# ------------------------------------------------------------------ 2〜5: 尤度の比較
def components(D: pd.DataFrame) -> pd.DataFrame:
    rel = E.rel
    D = D.copy()
    D["F_front"] = rel("front") * D["q_front"].fillna(0) * D["z_front"].fillna(0)
    D["F_kire"] = rel("kire") * D["q_kire"].fillna(0) * D["z_kire"].fillna(0)
    D["F_surf"] = D["r_surface"] * D["z_surface"].fillna(0)
    D["F_db"] = D["r_dbucket"] * D["z_dbucket"].fillna(0)
    D["F_deb"] = rel("debut") * D["is_debut_race"].astype(float) * D["z_debut"].fillna(0)
    comps = ["F_front", "F_kire", "F_surf", "F_db", "F_deb"]
    assert np.allclose(D[comps].sum(axis=1), D["F"]), "F の分解が一致しない"
    D["F_nolong"] = D["F"] - np.where(D["dbucket"] == LONG, D["F_db"], 0)
    for c in comps:
        D[f"F_minus_{c}"] = D["F"] - D[c]
    # 本人の過去平均(perf_adj、その日より前)と近走3走の成績(perf、係数不要)
    D["own_mean"] = (D["perf_adj_ps"] / D["perf_adj_pn"].replace(0, np.nan)).fillna(0)
    D["has_own"] = (D["perf_adj_pn"] > 0).astype(float)
    return D


def add_form3(runs: pd.DataFrame, D: pd.DataFrame) -> pd.DataFrame:
    r = runs.sort_values(["horse_id", "date"])[["race_id", "horse_id", "perf"]].copy()
    r["form3"] = r.groupby("horse_id")["perf"].transform(lambda s: s.shift(1).rolling(3, min_periods=1).mean())
    D = D.merge(r[["race_id", "horse_id", "form3"]], on=["race_id", "horse_id"], how="left")
    D["no_form"] = D["form3"].isna().astype(float)
    D["form3"] = D["form3"].fillna(0)
    return D


def effect_size(fitD, testD):
    out = {}
    p = E.PL(testD, ["S"])
    nr = p.mask.sum(1)
    ll_u = -(np.log(nr) + np.log(nr - 1) + np.log(nr - 2))
    out["mean_field"] = round(float(nr.mean()), 2)
    out["LL_uniform"] = round(float(ll_u.mean()), 4)
    for name, feats in [("S", ["S"]), ("S+F", ["S", "F"]), ("市場", ["logp"])]:
        D_f, D_t = (fitD, testD) if name != "市場" else (fitD[ok_race(fitD)], testD[ok_race(testD)])
        b = E.PL(D_f, feats).fit()
        pt = E.PL(D_t, feats)
        ll = pt.race_ll(b)
        nr_t = pt.mask.sum(1)
        llu_t = -(np.log(nr_t) + np.log(nr_t - 1) + np.log(nr_t - 2))
        u = np.where(pt.mask, pt.X @ b, -np.inf)
        mx = u.max(1, keepdims=True)
        lp1 = u[:, 0] - (mx[:, 0] + np.log(np.exp(u - mx).sum(1)))
        out[name] = {"LL_per_race": round(float(ll.mean()), 4), "gain_vs_uniform": round(float((ll - llu_t).mean()), 4),
                     "mcfadden_r2": round(float(1 - ll.sum() / llu_t.sum()), 5),
                     "winner_stage_logp": round(float(lp1.mean()), 4), "top1_accuracy": round(float((np.argmax(u, 1) == 0).mean()), 4)}
    d = out["S+F"]["LL_per_race"] - out["S"]["LL_per_race"]
    out["dLL_over_S_gain"] = round(d / out["S"]["gain_vs_uniform"], 3)
    out["dLL_over_market_gain"] = round(d / out["市場"]["gain_vs_uniform"], 4)
    out["winner_stage_dLL"] = round(out["S+F"]["winner_stage_logp"] - out["S"]["winner_stage_logp"], 5)
    out["note"] = "主仮説のΔLLを、父の総合力(S)が一様に対して得た改善・市場(確定オッズ)の改善と比べた比。1着の段は PL の第1段だけ。"
    return out


def ok_race(X):
    return X.groupby("race_id")["logp"].transform(lambda s: bool(np.isfinite(s).all()))


# ------------------------------------------------------------------ 6〜7: E4・E1
E4_AXES = [("front", "-"), ("kire", "-"), ("debut", "-"), ("surface", "芝"), ("surface", "ダ")] + \
          [("dbucket", b) for b in E.BUCKETS] + [("perf_adj", "-")]


def e4_stat(a: pd.Series):
    a = a.dropna()
    if len(a) < 10:
        return None
    share, ratio = float((a > 0).mean()), float(abs(a.median()) / a.std())
    return {"n_anc": int(len(a)), "share_pos": round(share, 3), "abs_median_over_sd": round(ratio, 3),
            "pass": bool(0.35 <= share <= 0.65 and ratio < 0.25)}


def e4_tables():
    out = {}
    for Y in list(range(2021, 2027)) + ["display"]:
        anc = pd.read_parquet(A.TABLE_DIR / f"anc_{Y}.parquet")
        rec = {}
        for ax, lv in E4_AXES:
            t = anc[(anc["axis"] == ax) & (anc["level"] == lv) & (anc["role"] == "sire")]
            rec[f"{ax}|{lv}"] = e4_stat(t.loc[t["n_horses"] >= 30, "mean"])
        out[str(Y)] = rec
    return out


V1_KEYS = ["front", "kire", "debut", "surface|芝", "surface|ダ"] + [f"dbucket|{b}" for b in E.BUCKETS]


def v1_counts(r: pd.DataFrame) -> pd.DataFrame:
    """種牡馬(父)ごとの v1 の件数。r は1走1行(is_win, leader, is_debut_race, surface, dbucket, ped_S_horse_id)。"""
    w = r["is_win"].astype(float)
    lead = (r["leader"] == True).astype(float)  # noqa: E712
    non = (r["leader"] == False).astype(float)  # noqa: E712
    x = pd.DataFrame({"sire": r["ped_S_horse_id"], "one": 1.0, "w": w, "lead": lead, "lw": lead * w, "non": non, "nw": non * w,
                      "dn": r["is_debut_race"].astype(float), "dw": r["is_debut_race"].astype(float) * w})
    for s in ["芝", "ダ"]:
        m = (r["surface"] == s).astype(float)
        x[f"s{s}n"], x[f"s{s}w"] = m, m * w
    for i, b in enumerate(E.BUCKETS):
        m = (r["dbucket"] == b).astype(float)
        x[f"b{i}n"], x[f"b{i}w"] = m, m * w
    g = x.dropna(subset=["sire"]).groupby("sire").sum()
    g["n_horses"] = r.dropna(subset=["ped_S_horse_id"]).groupby("ped_S_horse_id")["horse_id"].nunique()
    return g


def v1_values_var(g: pd.DataFrame) -> dict:
    """{軸: DataFrame(value pt, var pt², n_horses)}。セル下限N=10(v1 と同じ)。分散は二項の近似。"""
    ov = g["w"] / g["one"]
    out = {}

    def rate(wc, nc):
        p = g[wc] / g[nc]
        return p, p * (1 - p) / g[nc]
    pl, vl = rate("lw", "lead")
    pn, vn = rate("nw", "non")
    out["front"] = pd.DataFrame({"v": np.where(g["lead"] >= 10, (pl - ov) * 100, np.nan), "var": vl * 1e4})
    out["kire"] = pd.DataFrame({"v": np.where((g["lead"] >= 10) & (g["non"] >= 10), (pn - pl) * 100, np.nan), "var": (vl + vn) * 1e4})
    pdb, vdb = rate("dw", "dn")
    out["debut"] = pd.DataFrame({"v": np.where(g["dn"] >= 10, (pdb - ov) * 100, np.nan), "var": vdb * 1e4})
    for s in ["芝", "ダ"]:
        p, v = rate(f"s{s}w", f"s{s}n")
        out[f"surface|{s}"] = pd.DataFrame({"v": np.where(g[f"s{s}n"] >= 10, (p - ov) * 100, np.nan), "var": v * 1e4})
    for i, b in enumerate(E.BUCKETS):
        p, v = rate(f"b{i}w", f"b{i}n")
        out[f"dbucket|{b}"] = pd.DataFrame({"v": np.where(g[f"b{i}n"] >= 10, (p - ov) * 100, np.nan), "var": v * 1e4})
    for k in out:
        out[k].index = g.index
        out[k]["n_horses"] = g["n_horses"]
    return out


BANDS = [(10, 50), (50, 150), (150, 10 ** 9)]


def band_corr(a: pd.Series, b: pd.Series, n: pd.Series):
    res = {}
    for lo, hi in BANDS + [(10, 10 ** 9)]:
        m = n.between(lo, hi - 1) & a.notna() & b.notna()
        r, k = C.corr(a[m], b[m])
        res[f"{lo}-" + ("" if hi >= 10 ** 9 else str(hi - 1))] = {"n_sires": int(k), "r_half": r5(r), "SB": r5(C.sb(r)) if r is not None else None}
    return res


def e1_and_v1_e4(runs: pd.DataFrame):
    win = runs[runs["year"] <= 2025]   # anc_2026 と同じ窓
    ped = runs[["horse_id"] + [c for _, c in C.ROLES]].drop_duplicates("horse_id")
    v = A.run_values(win, win)
    hh = v["horse_id"].drop_duplicates()
    half = hh.map(lambda h: int(hashlib.md5(h.encode()).hexdigest(), 16) % 2)   # inner_validation と同じ分け方
    halves = [set(hh[half == i]) for i in (0, 1)]
    T = [A.build_tables(v, ped, horses=hs) for hs in halves]
    S = [A.shrink_tables(t)[0] for t in T]
    v2map = {"front": ("front", "-"), "kire": ("kire", "-"), "debut": ("debut", "-")}
    v2map.update({f"surface|{s}": ("surface", s) for s in ["芝", "ダ"]})
    v2map.update({f"dbucket|{b}": ("dbucket", b) for b in E.BUCKETS})
    e1 = {}
    G = [v1_values_var(v1_counts(win[win["horse_id"].isin(hs)])) for hs in halves]
    for key in V1_KEYS:
        rec = {}
        k2 = v2map[key] + ("sire",)
        a0, a1 = T[0].get(k2), T[1].get(k2)
        if a0 is not None and a1 is not None:
            cm = a0.index.intersection(a1.index)
            n = np.minimum(a0.loc[cm, "n_horses"], a1.loc[cm, "n_horses"])
            rec["v2_raw"] = band_corr(a0.loc[cm, "mean"], a1.loc[cm, "mean"], n)
            if k2 in S[0] and k2 in S[1]:
                rec["v2_shrunk"] = band_corr(S[0][k2].reindex(cm), S[1][k2].reindex(cm), n)
        g0, g1 = G[0][key], G[1][key]
        cm = g0.index.intersection(g1.index)
        n = np.minimum(g0.loc[cm, "n_horses"], g1.loc[cm, "n_horses"])
        rec["v1_raw"] = band_corr(g0.loc[cm, "v"], g1.loc[cm, "v"], n)
        # v1+縮約: 真の値の分散 τ² = 2分割の共分散、各父の雑音は二項の近似分散 → v·τ²/(τ²+var)
        ok = g0.loc[cm, "v"].notna() & g1.loc[cm, "v"].notna()
        tau2 = float(np.cov(g0.loc[cm][ok]["v"], g1.loc[cm][ok]["v"])[0, 1]) if ok.sum() > 10 else np.nan
        if np.isfinite(tau2) and tau2 > 0:
            sh = [g.loc[cm, "v"] * tau2 / (tau2 + g.loc[cm, "var"]) for g in (g0, g1)]
            rec["v1_shrunk"] = band_corr(sh[0], sh[1], n)
        rec["v1_tau2_pt2"] = r5(tau2)
        e1[key] = rec
    # v1 の E4(全産駒、産駒30頭以上の父)
    gv = v1_values_var(v1_counts(win))
    v1_e4 = {k: e4_stat(gv[k].loc[gv[k]["n_horses"] >= 30, "v"]) for k in V1_KEYS}
    return e1, v1_e4


# ------------------------------------------------------------------ 8: I²
def i2_req(req: pd.DataFrame):
    out = {}
    rq = req[req["year"].between(2021, 2026)]
    for a in ["front", "kire"]:
        g = rq.dropna(subset=[f"lift_{a}"]).groupby("course_key_io")[f"lift_{a}"].agg(["mean", "std", "count"])
        g = g[g["count"] >= 10]
        w = g["count"] / g["std"] ** 2
        m = float(np.sum(w * g["mean"]) / w.sum())
        Q = float(np.sum(w * (g["mean"] - m) ** 2))
        df = len(g) - 1
        out[a] = {"n_course_keys": int(len(g)), "Q": round(Q, 1), "df": df, "I2": round(max(0.0, (Q - df) / Q), 3)}
    return out


def main():
    t0 = time.time()
    runs, req, kh, D = E.build_D()
    D = add_form3(runs, components(D))
    fitD, testD = D[D["year"].between(*E.FIT)], D[D["year"].between(*E.TEST)]
    res = {"status": "探索(R4本番の結果を見た後に計算。確認検証ではない)", "fit_years": E.FIT, "test_years": E.TEST}

    # 1. 再現
    m = E.compare(fitD, testD, ["S"], ["F"], "主仮説の再現")
    assert abs(m["dll_per_race"] - MAIN["main"]["dll_per_race"]) < 1e-5, (m["dll_per_race"], MAIN["main"]["dll_per_race"])
    res["reproduced_main"] = strip(m)
    log("reproduced", m["dll_per_race"])

    # 2. 軸ごとの内訳
    comps = ["F_front", "F_kire", "F_surf", "F_db", "F_deb"]
    names = {"F_front": "脚質(要求×脚質)", "F_kire": "キレ(要求×キレ)", "F_surf": "芝ダ", "F_db": "距離帯", "F_deb": "新馬"}
    res["axis_single"] = {names[c]: strip(E.compare(fitD, testD, ["S"], [c], f"S → +{c}")) for c in comps}
    res["axis_drop_one"] = {names[c]: strip(E.compare(fitD, testD, ["S"], [f"F_minus_{c}"], f"S → +(F − {c})")) for c in comps}
    res["axis_free_beta"] = strip(E.compare(fitD, testD, ["S"], comps, "S → +5軸(β を自由に推定)"))
    res["main_without_long_bucket"] = strip(E.compare(fitD, testD, ["S"], ["F_nolong"], "S → +F(長距離帯を除く)"))
    log("axis", {k: v["dll_per_race"] for k, v in res["axis_single"].items()})
    # クラス別(主仮説の ΔLL をレースの種類で分ける)
    d = pd.Series(m["_delta"], index=m["_race_ids"])
    races = testD.drop_duplicates("race_id").set_index("race_id").loc[d.index]
    kind = np.where(races["is_debut_race"], "新馬戦", np.where(races["class_ord"] == 0, "未勝利戦", "1勝クラス以上"))
    res["main_by_race_class"] = {k: {"n_races": int((kind == k).sum()), "dll_per_race": r5(d[kind == k].mean()),
                                     "share_of_total_dll": round(float(d[kind == k].sum() / d.sum()), 3)}
                                 for k in ["新馬戦", "未勝利戦", "1勝クラス以上"]}
    by_year = MAIN["main"]["by_year"]
    res["main_year_sign_test"] = {"n_years": len(by_year), "n_positive": int(sum(v > 0 for v in by_year.values())),
                                  "p_one_sided": round(0.5 ** sum(v > 0 for v in by_year.values()), 4) if all(v > 0 for v in by_year.values()) else None,
                                  "note": "年ブロックのブートストラップCIはクラスタ6個で狭く出すぎるため、符号検定を併記"}

    # 3. 副次(4)の別の形
    alt = [(["S", "own_mean", "has_own"], ["F_own"], "基準 [S, 本人の過去平均, 過去走の有無] → +F_own"),
           (["S", "own_mean", "has_own"], ["F"], "基準 [S, 本人の過去平均, 過去走の有無] → +F(血統のみ)"),
           (["S_own"], ["F"], "基準 S_own → +F(血統のみ)"),
           (["S_own", "form3", "no_form"], ["F_own"], "基準 S_own+近走3走 → +F_own"),
           (["S_own", "form3", "no_form"], ["F"], "基準 S_own+近走3走 → +F(血統のみ)")]
    res["own_alternatives"] = {name: strip(E.compare(fitD, testD, b, x, name)) for b, x, name in alt}
    res["own_with_market"] = strip(E.compare(fitD[ok_race(fitD)], testD[ok_race(testD)], ["logp", "S_own"], ["F_own"],
                                             "市場+S_own → +F_own"))
    log("own", {k: v["dll_per_race"] for k, v in res["own_alternatives"].items()})

    # 4. 効果量
    res["effect_size"] = effect_size(fitD, testD)

    # 5. プラセボの判定の見直し
    pw, pa = MAIN["placebo"]["within"], MAIN["placebo"]["all"]
    n_fit = MAIN["n_fit_races"]
    res["placebo_review"] = {
        "within_literal_s6": {"mean": pw["mean"], "2se": round(2 * pw["se_single"], 6), "pass": bool(abs(pw["mean"]) <= 2 * pw["se_single"]),
                              "note": "§6 の文言(平均が0の±2SE以内)を同じ芝ダ内のプラセボにそのまま当てると不合格。芝ダ軸の寄与が残るため(追補2)。"},
        "within_vs_surface_only": {"surface_only_dll": res["axis_single"]["芝ダ"]["dll_per_race"],
                                   "note": "同じ芝ダ内のプラセボの平均は、芝ダ項を雑音で薄めた分でほぼ説明できる(DLエンジニアの指摘)。"},
        "all_mean": pa["mean"], "all_se_of_mean_of_20": round(pa["sd"] / np.sqrt(len(pa["dll_per_race"])), 6),
        "all_expected_noise_one_param": round(-1 / (2 * n_fit), 6),
        "all_actual_above_max": bool(MAIN["main"]["dll_per_race"] > pa["max"]),
        "note": "全レース版は、別の芝ダの値を当てると逆向きの信号になり打ち消し合うため、平均≈0は構造上ほぼ当然で情報は弱い。"
                "雑音1変数の期待値は0でなく約 −1/(2·N_fit)。要求ベクトル q の妥当性はこのプラセボでは裏付けられない。"}

    # 6〜8
    res["E4_tables_v2"] = e4_tables()
    log("E4 done")
    res["E1_split_half_window_le2025"], res["E4_v1_window_le2025"] = e1_and_v1_e4(runs)
    log("E1 done")
    res["I2_req_2021_2026"] = i2_req(req)
    res["not_done"] = {
        "two_stage_shrinkage_by_lineage": "§2 の系統を親にする2段階縮約の感度分析は未実施(系統カテゴリの軸は追補1で不採用、主は全体0)。",
        "req_from_pedigree_lift": "§5 の血統値リフト版の基準(感度分析)は未実施。",
        "race_req_E4_v1_lift": "レース基準の v1 リフトとの比較は R−0.5(2020年まで)で実施済み、テスト期間では未実施。"}
    res["manifest"] = E.manifest()
    res["elapsed_sec"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)),
                   encoding="utf-8")
    E.append_ledger({"script": "radar_v2_eval_addendum.py", "mode": "explore(after confirm)",
                     "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)), "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
                     "git_commit": res["manifest"]["git_commit"], "git_dirty_files": res["manifest"]["git_dirty_files"],
                     "output": OUT.name, "output_sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()})
    log("wrote", OUT)


if __name__ == "__main__":
    main()
