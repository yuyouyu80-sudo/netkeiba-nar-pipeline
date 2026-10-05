# -*- coding: utf-8 -*-
"""荒れ度分析 手順2・3: 2026-07-11〜09-27(JRA平地、ファクターDBの794レースから障害戦を除く)。

手順2: 3区分ごとの万馬券率・条件別集計、15年スコア(s1_models.pkl、2011〜2022学習)の当てはまり、
       馬柱/ファクターDB由来のレース単位「荒れサイン」10個を足したときの上乗せを日付単位の
       leave-one-date-out 交差検証で確認。
手順3: 「馬柱データ予想２」系の予想(予想順位 pred_rank、3連複印 fmark)で的中/外れ×荒れ/荒れないの4分割、
       外れて荒れたレースの理由A〜Gを結果データから機械的に判定。

入力(読み取りのみ):
  out/races_all.csv.gz, out/s1_models.pkl
  data/jra_pipeline/factor_database.json(予想順位・凡走予測・展開予想など、794レース)
  data/jra_pipeline/jra_formation_base_2026_09_30.json(通常戦のscore5/score4)
  data/jra_pipeline/col1_special/fmark_backfill_404.json(3連複印、通常戦400レース)
  data/newspaper/{race_id}.csv(予想印・スピード指数・調教評価)
  data/race_results/2026/*.csv(着順・通過順・馬体重 → 手順3の事後診断にだけ使う)
出力(out/): s2_*.csv, s3_*.csv
"""
import json
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s2_stats_model as S  # noqa: E402

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[2]
OUT = S.OUT  # ARERE_TARGET=katai なら out_katai/
CATS = ["一般", "新馬", "未勝利"]
RNG = np.random.default_rng(1005)
RT_MAP = {"normal": "一般", "shinba": "新馬", "mishoubi": "未勝利"}

SIGNALS = {
    "honmei_split": "◎の割れ(◎が付いた馬の数÷◎を打った記者数)",
    "fav_honmei_share": "1番人気に◎を打った記者の割合",
    "ana_honmei_share": "4番人気以下に◎を打った記者の割合",
    "speed_top_gap": "スピード指数(最高)の1位−3位の差",
    "speed_cv": "スピード指数(近5走平均)のばらつき(変動係数)",
    "longshot_training_ab": "6番人気以下で調教評価A/Bの頭数",
    "model_top_pop": "予想1位馬の人気(対数)",
    "model_market_rho": "予想順位と人気の順位相関",
    "fav_flop_rank": "1番人気の凡走予測(安全度)順位÷頭数",
    "top3pop_unproven": "上位3人気のうち初コース・初距離・前走なしの頭数",
}


def load_fdb():
    d = json.load(open(ROOT / "data/jra_pipeline/factor_database.json", encoding="utf-8"))
    return {r["race_id"]: r for r in d["races"]}


def ninki_of(hs, res):
    """確定人気(race_results)を優先、無ければbias_ninki。"""
    out = {}
    for h in hs:
        u = h["umaban"]
        out[u] = res.get(u, {}).get("pop") if res.get(u, {}).get("pop") == res.get(u, {}).get("pop") else None
        if out[u] is None:
            out[u] = h.get("bias_ninki")
    return out


def num(x):
    try:
        v = float(str(x).replace(",", ""))
        return v
    except ValueError:
        return np.nan


def race_signals(r, res):
    hs = r["horses"]
    n = len(hs)
    pop = ninki_of(hs, res)
    news = pd.read_csv(ROOT / f"data/newspaper/{r['race_id']}.csv", dtype=str)
    news["umaban"] = news["umaban"].astype(int)
    news = news.set_index("umaban")
    mcols = [c for c in news.columns if c.startswith("mark_raw_")]
    fav = min((u for u in pop if pop[u] is not None), key=lambda u: pop[u], default=None)
    hon = {c: news.index[news[c] == "◎"].tolist() for c in mcols}
    hon = {c: v for c, v in hon.items() if v}
    nexp = len(hon)
    honmei_horses = set(u for v in hon.values() for u in v)
    sig = {}
    sig["honmei_split"] = len(honmei_horses) / nexp if nexp else np.nan
    sig["fav_honmei_share"] = np.mean([fav in v for v in hon.values()]) if nexp else np.nan
    sig["ana_honmei_share"] = np.mean([any((pop.get(u) or 99) >= 4 for u in v) for v in hon.values()]) if nexp else np.nan
    smax = news["speed_max_index"].map(num).dropna().sort_values(ascending=False)
    sig["speed_top_gap"] = smax.iloc[0] - smax.iloc[2] if len(smax) >= 3 else np.nan
    savg = news["speed_avg_index_5races"].map(num).dropna()
    sig["speed_cv"] = savg.std() / savg.mean() if len(savg) >= 3 and savg.mean() > 0 else np.nan
    tr = news["training_rank"]
    sig["longshot_training_ab"] = sum(1 for u in news.index if (pop.get(u) or 0) >= 6 and tr.get(u) in ("A", "B"))
    top = min(hs, key=lambda h: h["pred_rank"] if h["pred_rank"] is not None else 99)
    sig["model_top_pop"] = np.log(pop.get(top["umaban"]) or n)
    pr = [(h["pred_rank"], pop.get(h["umaban"])) for h in hs if h["pred_rank"] is not None and pop.get(h["umaban"]) is not None]
    sig["model_market_rho"] = spearmanr([a for a, _ in pr], [b for _, b in pr]).statistic if len(pr) >= 4 else np.nan
    fh = next((h for h in hs if h["umaban"] == fav), None)
    sig["fav_flop_rank"] = fh["flop_safety_rank"] / n if fh and fh.get("flop_safety_rank") else np.nan
    sig["top3pop_unproven"] = sum(1 for h in hs if (pop.get(h["umaban"]) or 99) <= 3 and
                                  (h.get("first_time_flag") != "experienced" or h.get("layoff_flag") == "no_history"))
    return sig


def load_results_2026():
    import glob
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/race_results/2026/*.csv"))):
        d = pd.read_csv(f, dtype=str)
        if d["race_date"].iloc[0] < "2026-07-11" or d["race_date"].iloc[0] > "2026-09-27":
            continue
        rows.append(d)
    d = pd.concat(rows)
    res = {}
    for _, x in d.iterrows():
        fp = str(x["finish_pos"])
        m = re.match(r"^(\d+)", fp)
        po = str(x["passing_order"]) if pd.notna(x["passing_order"]) else ""
        lastc = re.findall(r"\d+", po)
        w = re.search(r"\(([+-]?\d+)\)", str(x["weight"]))
        res.setdefault(x["race_id"], {})[int(x["umaban"])] = dict(
            fin=int(m.group(1)) if m else None, fin_raw=fp, pop=num(x["popularity"]),
            c4=int(lastc[-1]) if lastc else None, wdiff=int(w.group(1)) if w else None, waku=int(x["waku"]))
    return res


def fit_eval_lodo(X, y, groups, C=0.5):
    """leave-one-date-out。Xは列標準化をfold内で実施。"""
    p = np.zeros(len(y))
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        sc = StandardScaler().fit(X[tr])
        m = LogisticRegression(C=C, max_iter=2000).fit(sc.transform(X[tr]), y[tr])
        p[te] = m.predict_proba(sc.transform(X[te]))[:, 1]
    return p


def ll_vec(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def boot_delta(y, p0, p1, groups, B=2000):
    """日付クラスタ・ブートストラップで(base logloss − 追加後 logloss)の合計差と95%CI。"""
    d = ll_vec(y, p0) - ll_vec(y, p1)
    ug = np.unique(groups)
    by = np.array([d[groups == g].sum() for g in ug])
    s = [by[RNG.integers(0, len(ug), len(ug))].sum() for _ in range(B)]
    return d.sum(), np.percentile(s, 2.5), np.percentile(s, 97.5)


def main():
    races = S.apply_target(pd.read_csv(S.RACES, dtype={"race_id": str}))
    fdb = load_fdb()
    res = load_results_2026()
    d = races[races["race_id"].isin(fdb)].copy()
    d = d[d["arere"].notna()].copy()
    d["arere"] = d["arere"].astype(int)
    jumps = sorted(set(fdb) - set(races["race_id"]))
    print("FDB races", len(fdb), "-> 平地・払戻あり", len(d), "除外(障害等):", jumps)
    chk = d.apply(lambda r: RT_MAP[fdb[r["race_id"]]["race_type"]] == r["category"], axis=1)
    print("区分一致", int(chk.sum()), "/", len(d))

    # 15年スコア(2011〜2022学習モデルを2026へそのまま適用=完全な時系列外)
    models = pickle.load(open(OUT / "s1_models.pkl", "rb"))
    dm = S.add_bins(S.prep_numeric(d))
    d["score15_odds"] = np.nan
    d["score15_cond"] = np.nan
    for cat in CATS:
        msk = (dm["category"] == cat).to_numpy()
        d.loc[msk, "score15_odds"] = models[(cat, "条件+オッズ")].predict_proba(dm[msk])[:, 1]
        d.loc[msk, "score15_cond"] = models[(cat, "条件のみ(オッズなし)")].predict_proba(dm[msk])[:, 1]

    sig = pd.DataFrame([dict(race_id=rid, **race_signals(fdb[rid], res.get(rid, {}))) for rid in d["race_id"]])
    d = d.merge(sig, on="race_id")

    # ---- 手順2: 概要・条件別 ----
    ov = []
    d15 = races[(races["year"].between(2011, 2025)) & races["arere"].notna()]
    for cat in CATS + ["全体"]:
        x = d if cat == "全体" else d[d["category"] == cat]
        h = d15 if cat == "全体" else d15[d15["category"] == cat]
        lo, hi = S.wilson(x["arere"].sum(), len(x))
        ov.append(dict(category=cat, n_races=len(x), n_arere=int(x["arere"].sum()), rate=x["arere"].mean(), ci_lo=lo, ci_hi=hi,
                       rate_15y=h["arere"].astype(int).mean(), mean_score15_odds=x["score15_odds"].mean(),
                       mean_score15_cond=x["score15_cond"].mean(),
                       AUC_score15_odds=roc_auc_score(x["arere"], x["score15_odds"]) if x["arere"].nunique() > 1 else np.nan,
                       AUC_score15_cond=roc_auc_score(x["arere"], x["score15_cond"]) if x["arere"].nunique() > 1 else np.nan))
    pd.DataFrame(ov).to_csv(OUT / "s2_overview_2026.csv", index=False, encoding="utf-8-sig")
    rates, imp = S.factor_tables(S.add_bins(S.prep_numeric(d)))
    rates.to_csv(OUT / "s2_factor_rates_2026.csv", index=False, encoding="utf-8-sig")

    # 荒れサインの単変量(区分別): 荒れ/荒れない平均とAUC
    uni = []
    for cat in CATS:
        x = d[d["category"] == cat]
        for s_, lab in SIGNALS.items():
            v = x[[s_, "arere"]].dropna()
            if v["arere"].nunique() < 2:
                continue
            uni.append(dict(category=cat, signal=s_, label=lab, n=len(v), mean_arere=v.loc[v.arere == 1, s_].mean(),
                            mean_not=v.loc[v.arere == 0, s_].mean(), AUC=roc_auc_score(v["arere"], v[s_]),
                            spearman_vs_score15=spearmanr(x.loc[v.index, s_], x.loc[v.index, "score15_odds"]).statistic))
    pd.DataFrame(uni).to_csv(OUT / "s2_signal_univariate.csv", index=False, encoding="utf-8-sig")

    # 上乗せ検定: base = logit(score15_odds) の再較正のみ / +10サイン(欠損は区分内中央値で補完)
    inc = []
    for cat in CATS + ["全体(区分ダミー付)"]:
        x = d if cat.startswith("全体") else d[d["category"] == cat]
        x = x.copy()
        for s_ in SIGNALS:
            x[s_] = x[s_].fillna(x[s_].median()).fillna(0.0)  # 新馬は指数等が全欠損→定数0
        y = x["arere"].to_numpy()
        grp = x["date"].to_numpy()
        lg = np.log(x["score15_odds"] / (1 - x["score15_odds"])).to_numpy()[:, None]
        Xb = lg
        if cat.startswith("全体"):
            Xb = np.hstack([lg, pd.get_dummies(x["category"]).to_numpy(float)[:, 1:]])
        Xs = np.hstack([Xb, x[list(SIGNALS)].to_numpy(float)])
        p0 = fit_eval_lodo(Xb, y, grp, C=10.0)
        p1 = fit_eval_lodo(Xs, y, grp, C=0.5)
        dl, lo, hi = boot_delta(y, p0, p1, grp)
        pc = fit_eval_lodo(np.log(x["score15_cond"] / (1 - x["score15_cond"])).to_numpy()[:, None], y, grp, C=10.0)
        inc.append(dict(category=cat, n=len(x), n_arere=int(y.sum()), n_dates=len(np.unique(grp)),
                        AUC_cond_only=roc_auc_score(y, pc), AUC_score15=roc_auc_score(y, p0), AUC_score15_plus10=roc_auc_score(y, p1),
                        logloss_score15=log_loss(y, p0), logloss_plus10=log_loss(y, p1),
                        dLL_total=dl, dLL_lo=lo, dLL_hi=hi,
                        verdict="上乗せあり(CI>0)" if lo > 0 else ("悪化(CI<0)" if hi < 0 else "差は判別できず(CIが0をまたぐ)")))
        # サイン1個ずつ足した場合
        for s_ in SIGNALS:
            p1s = fit_eval_lodo(np.hstack([Xb, x[[s_]].to_numpy(float)]), y, grp, C=0.5)
            dl, lo, hi = boot_delta(y, p0, p1s, grp, B=1000)
            inc.append(dict(category=cat, n=len(x), n_arere=int(y.sum()), signal_alone=s_, AUC_score15=roc_auc_score(y, p0),
                            AUC_score15_plus10=roc_auc_score(y, p1s), dLL_total=dl, dLL_lo=lo, dLL_hi=hi))
        print(cat, inc[-11], flush=True)
    pd.DataFrame(inc).to_csv(OUT / "s2_increment_lodo.csv", index=False, encoding="utf-8-sig")

    # ---- 手順3: 予想的中/外れ × 荒れ/荒れない、外れ理由 ----
    fm = {r["race_id"]: r for r in json.load(open(ROOT / "data/jra_pipeline/col1_special/fmark_backfill_404.json", encoding="utf-8"))["races"]}
    rows, reasons = [], []
    for _, rr in d.iterrows():
        rid = rr["race_id"]
        r = fdb[rid]
        rs = res.get(rid, {})
        hs = {h["umaban"]: h for h in r["horses"]}
        n = rr["n_horses"]
        top3 = [u for u, v in rs.items() if v["fin"] is not None and v["fin"] <= 3]
        pr = {u: h["pred_rank"] for u, h in hs.items() if h["pred_rank"] is not None}
        box5 = {u for u, k in pr.items() if k <= 5}
        hit_box5 = len(top3) >= 3 and set(top3) <= box5
        hit_fm = None
        if rid in fm:
            c1, c2, c3 = (set(fm[rid][k]) for k in ("c1", "c2", "c3"))
            from itertools import permutations
            hit_fm = any(a in c1 and b in c2 and c in c3 for a, b, c in permutations(top3, 3))
        pop = {u: v["pop"] for u, v in rs.items()}
        # 理由判定(事後診断。予測側には使っていない)
        # E: 予想上位5頭か上位3人気が 取消・除外・競走中止・失格、または当日馬体重±20kg以上
        acc = [u for u in set(box5) | {u for u, p in pop.items() if p == p and p <= 3}
               if rs.get(u) and (rs[u]["fin"] is None or (rs[u]["wdiff"] is not None and abs(rs[u]["wdiff"]) >= 20))]
        favs_out = [u for u, p in pop.items() if p == p and p <= 3 and (rs[u]["fin"] is None or rs[u]["fin"] >= 4)]
        fav1 = [u for u, p in pop.items() if p == 1]
        A = (fav1 and (rs[fav1[0]]["fin"] is None or rs[fav1[0]]["fin"] >= 4)) or len(favs_out) >= 2
        B = any((pop.get(u) or 0) >= 7 and pr.get(u, 99) >= 6 for u in top3)
        dev = [abs(rs[u]["c4"] - hs[u]["corner4_expected_rank"]) / n for u in top3
               if u in hs and rs[u]["c4"] is not None and hs[u].get("corner4_expected_rank") is not None]
        C = bool(dev) and np.mean(dev) >= 0.30
        wk = [rs[u]["waku"] for u in top3]
        D = (len(wk) >= 3 and (max(wk) <= 3 or min(wk) >= 6)) or str(rr["going"]) in ("重", "不良")
        F = any(u in hs and (hs[u].get("layoff_flag") == "no_history" or hs[u].get("first_time_flag") == "both_first") for u in top3)
        E = bool(acc)
        # G(モデルの弱点): 人気上位5頭は3着内馬を2頭以上含むのに、予想上位5頭は1頭以下(=人気順より悪い外れ方)
        G = sum((pop.get(u) or 99) <= 5 for u in top3) >= 2 and sum(u in box5 for u in top3) <= 1
        flags = dict(A=bool(A), B=bool(B), C=bool(C), D=bool(D), E=E, F=bool(F), G=bool(G))
        # 主理由の優先順: 予測不能・構造要因(E→C→D→F)→モデル固有(G)→一般的な現象(A→B)。どれも無ければ「-」
        primary = next((k for k in "ECDFGAB" if flags[k]), "-")
        rows.append(dict(race_id=rid, date=rr["date"], category=rr["category"], race_name=rr["race_name"], n_horses=n,
                         sanrenpuku=rr["sanrenpuku"], arere=rr["arere"], hit_box5=hit_box5, hit_fmark=hit_fm,
                         score15_odds=rr["score15_odds"], primary_reason=primary, **{f"reason_{k}": v for k, v in flags.items()},
                         top3_umaban="-".join(map(str, sorted(top3))), top3_pop="-".join(str(int(pop[u])) if pop.get(u) == pop.get(u) else "?" for u in sorted(top3)),
                         top3_pred_rank="-".join(str(pr.get(u, "?")) for u in sorted(top3))))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "s3_race_level.csv", index=False, encoding="utf-8-sig")

    tab = []
    for cat in CATS + ["全体"]:
        x = t if cat == "全体" else t[t["category"] == cat]
        for key, lab in [("hit_box5", "予想上位5頭BOX(3連複)"), ("hit_fmark", "3連複印フォーメーション(通常戦のみ)")]:
            xx = x[x[key].notna()]
            if not len(xx):
                continue
            for h in (True, False):
                for a in (1, 0):
                    tab.append(dict(category=cat, prediction=lab, hit="的中" if h else "外れ", arere=(("堅い(1,000円以下)" if a else "1,000円超") if S.TARGET == "katai" else ("荒れた" if a else "荒れない")),
                                    n=int(((xx[key] == h) & (xx["arere"] == a)).sum()), total=len(xx)))
    pd.DataFrame(tab).to_csv(OUT / "s3_hit_x_arere.csv", index=False, encoding="utf-8-sig")
    miss = t[(~t["hit_box5"]) & (t["arere"] == 1)]
    rc = []
    for cat in CATS + ["全体"]:
        x = miss if cat == "全体" else miss[miss["category"] == cat]
        for k in "ABCDEFG-":
            rc.append(dict(category=cat, reason=k, n_primary=int((x["primary_reason"] == k).sum()),
                           n_any=int(x[f"reason_{k}"].sum()) if k != "-" else 0, n_races=len(x)))
    pd.DataFrame(rc).to_csv(OUT / "s3_miss_reasons.csv", index=False, encoding="utf-8-sig")
    print(pd.DataFrame(ov).to_string())
    print(pd.DataFrame(tab).to_string())
    print(pd.DataFrame(rc).pivot(index="reason", columns="category", values=["n_primary", "n_any"]).to_string())


if __name__ == "__main__":
    main()
