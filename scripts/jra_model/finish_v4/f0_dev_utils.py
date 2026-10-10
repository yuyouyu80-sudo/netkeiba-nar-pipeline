# -*- coding: utf-8 -*-
"""finish_v4 F0-0: 開発期間(2018〜2020年)の M1・M5 の効用を保存し、3連複の指標を出す(2026-10-10、プラン第3版 F0)。

- M1 = 血統+実績(S_own・F_own)、M5 = M1 + 距離の変更・能力値・斤量・休み明け・前走の能力値(radar_v2_ability の開発評価と同じ特徴量)。
- 毎年 2014〜Y−1年で学習し Y年で評価する(Y=2018・2019・2020)。2021年以降は読まない(assert)。
- 組の確率は2通り: PL(2・3着段階の縮め係数 λ2・λ3 を推定)/ 3頭の集合の条件付きロジット。
- 指標: 組の対数スコア(実際の3頭の組に付けた確率の log、同着は当たり組の確率の平均)、確率順 N 点買いの的中率と回収率(N=1〜50)。
- 候補は発走した全馬(F0-1、2026-10-10): 競走中止・失格の馬も、3着以内に入らない候補として学習・評価の両方に入れる(nonfinish.py)。
  取消・除外は候補に入れない。市場(オッズ)は一切使わない(中止の判定に確定単勝オッズの有無だけを使う)。
  特徴量の作成は radar_v2_ability(内部で radar_v2_eval の Tables・add_own を使う。期間に依存する build_D は呼ばない)。
出力: data/jra_pipeline/finish_v4/dev_utils.parquet(Git管理外)、dev_f0_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
import pl_ext as P  # noqa: E402
import nonfinish as NF  # noqa: E402
import radar_v2_ability as AB  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_common as C  # noqa: E402

OUT_DIR = C.PROJECT_ROOT / "data" / "jra_pipeline" / "finish_v4"
OUT_DIR.mkdir(parents=True, exist_ok=True)
YEARS = (2018, 2019, 2020)
TRAIN_START = 2014
SPECS = {"M1": ["S_own", "F_own"],
         "M5": ["S_own", "F_own", "ext", "short", "ext_amt", "ab_mu", "ab_sd", "ab_new", "kin", "d_kin", "rest_long", "last_fig"]}
AB_COLS = ["ab_mu", "ab_sd", "ab_new", "ew_mu", "last_fig", "kin", "d_kin", "rest_long"]
NS = list(range(1, 51))
log = AB.log


def load_sanrenpuku(years) -> dict:
    """{race_id: [(組の馬番 frozenset, 払戻円)]}。同着で複数の当たり組があれば全部。"""
    out = {}
    for y in years:
        for p in sorted((C.PROJECT_ROOT / "data" / "payouts" / str(y)).glob("*.csv")):
            d = pd.read_csv(p, dtype=str)
            d = d[d["bet_type"] == "3連複"]
            for rid, comb, pay in zip(d["race_id"], d["combination"], d["payout"]):
                try:
                    s = frozenset(int(x) for x in comb.split("-"))
                    out.setdefault(rid, []).append((s, int(float(pay))))
                except (ValueError, AttributeError):
                    pass
    return out


def load_aug_runs(aug: pd.DataFrame) -> pd.DataFrame:
    """radar_v2_ability.load と同じ処理を、中止の行を足した走の表に当てる。"""
    import radar_v2_stamina as S
    runs = S.load_base(aug)
    ext = pd.read_parquet(AB.RUNS_EXT)
    runs = runs.merge(ext, on=["race_id", "horse_id"], how="left")
    sr = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")[["pace_hi"]]
    runs = runs.join(sr, on="race_id")
    runs["km"] = runs["distance"] / 1000.0
    runs["tpk"] = runs["time_sec"] / runs["km"]
    runs["day_key"] = runs["date"].dt.strftime("%Y%m%d") + "|" + runs["racecourse"] + "|" + runs["surface"]
    runs["cls_key"] = runs["surface"] + "|" + runs["class_key"].astype(str)
    return runs.sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)


def ghost_features(tp: dict, runs_fin: pd.DataFrame, D: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """中止の行(D にあるレースだけ)の発走前の特徴量。係数はすべて完走馬だけの表で推定し、中止の行を足した表に当てる。"""
    import radar_v2_eval as E   # Tables・add_own・features_for_year だけを年を明示して使う(build_D は呼ばない)
    import radar_v2_stamina as S
    base_all = pd.read_parquet(A.RUNS)
    th = A.fixed_thresholds(base_all)
    aug, meta = NF.augment(base_all, AB.DEV_END)
    aug = aug[~aug["nonfinish"] | aug["race_id"].isin(set(D["race_id"]))]
    assert aug["year"].max() <= 2020
    bra = A.add_tiers(aug, th)
    bra["umaban_num"] = pd.to_numeric(bra["umaban"], errors="coerce")
    bfin = A.add_tiers(base_all[base_all["year"] <= AB.DEV_END], th)
    bfin["umaban_num"] = pd.to_numeric(bfin["umaban"], errors="coerce")
    req = pd.read_parquet(C.OUT_DIR / "race_req.parquet")
    kh = E.estimate_kh(bfin)
    own = []
    for Y in range(TRAIN_START, AB.DEV_END + 1):
        gY = bra[bra["nonfinish"] & (bra["year"] == Y)]
        df = E.features_for_year(pd.concat([bra[(bra["year"] < Y)], gY]), req, Y)
        f = E.add_own(bra[bra["year"] <= Y], df, Y, kh)
        own.append(f[["race_id", "horse_id", "S_own", "F_own"]])
    own = pd.concat(own).set_index(["race_id", "horse_id"])
    ra = load_aug_runs(aug)
    R = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")
    flags = S.add_run_cols(ra[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan), R)
    flags = flags[ra["nonfinish"].to_numpy()].set_index(["race_id", "horse_id"])[["ext", "short", "ext_amt"]]
    abf = []
    S_fin, S_aug = AB.sire_S(runs_fin), AB.sire_S(ra)    # 年ごとの時点表なので、全期間で一度に作ってよい
    for Y in range(TRAIN_START, AB.DEV_END + 1):
        rf, rA = runs_fin[runs_fin["year"] <= Y], ra[ra["year"] <= Y]
        P_ = AB.fit_fig(rf, Y - 1)
        fig_f = AB.apply_fig(rf, P_)
        a, b = AB.prior_fit(rf, fig_f, S_fin[rf.index], Y - 1)
        gr = AB.growth_table(rf, fig_f, Y - 1)
        fig_a = AB.apply_fig(rA, P_)
        st = AB.kalman(rA, fig_a, a + b * S_aug[rA.index].fillna(0), tp["kalman"], gr)
        g = rA.groupby("horse_id")
        st["last_fig"] = fig_a.groupby(rA["horse_id"]).shift(1)
        st["kin"] = rA["kinryo"] - 55.0
        st["d_kin"] = (rA["kinryo"] - g["kinryo"].shift(1)).fillna(0)
        st["rest_long"] = (rA["rest_days"] > 90).astype(float)
        st["ab_new"] = (st["ab_n"] == 0).astype(float)
        st["ew_mu"] = np.nan          # M5 では使わない
        m = (rA["nonfinish"] & (rA["year"] == Y)).to_numpy()
        abf.append(pd.concat([rA.loc[m, ["race_id", "horse_id"]], st.loc[m, AB_COLS]], axis=1))
        log("ghost ability", Y, int(m.sum()))
    abf = pd.concat(abf).set_index(["race_id", "horse_id"])
    gh = ra[ra["nonfinish"] & ra["year"].between(TRAIN_START, AB.DEV_END)].copy()
    G = gh[["race_id", "horse_id", "year", "umaban_num", "is_debut_race", "class_ord"]].copy()
    G["pos"] = np.nan
    G["block"] = gh["date"].dt.strftime("%Y%m%d") + "_" + gh["race_id"].str[4:6]
    G = G.join(own, on=["race_id", "horse_id"]).join(flags, on=["race_id", "horse_id"]).join(abf, on=["race_id", "horse_id"])
    for c in ["ext", "short"]:
        G[c] = G[c].fillna(False).astype(float)
    G["ext_amt"] = G["ext_amt"].fillna(0.0)
    G["nonfinish"] = True
    meta["ghost_rows_used"] = int(len(G))
    meta["ghost_feature_nan"] = {c: int(G[c].isna().sum()) for c in ["S_own", "F_own", "ab_mu", "kin"]}
    return G.reset_index(drop=True), meta


TABLES_CACHE = OUT_DIR / "dev_tables.parquet"   # 評価年ごとの特徴量の表(学習+評価の行、Git管理外)。F2 以降の比較の土台


def dev_tables() -> tuple[dict, dict]:
    """{評価年 Y: 2014〜Y年の行(完走馬+中止の行)と M5 までの特徴量}。能力値の係数は Y−1年末まで。"""
    tp = json.loads(AB.TUNE_OUT.read_text(encoding="utf-8"))
    runs, D = AB.build_eval_table(tp)
    assert D["year"].max() <= 2020 and runs["year"].max() <= 2020
    G, gmeta = ghost_features(tp, runs, D)
    D = D.assign(nonfinish=False)
    keep = ["race_id", "horse_id", "year", "pos", "umaban_num", "block", "is_debut_race", "class_ord", "nonfinish"] + \
        sorted(set(SPECS["M5"]) | set(AB_COLS))
    tabs = {}
    for Y in YEARS:
        F = AB.ability_features(runs[runs["year"] <= Y], Y, tp).set_index(["race_id", "horse_id"])[AB_COLS]
        tabs[Y] = pd.concat([D[D["year"] <= Y].join(F, on=["race_id", "horse_id"]), G[G["year"] <= Y]], ignore_index=True)[keep]
    pd.concat([t.assign(eval_year=Y) for Y, t in tabs.items()], ignore_index=True).to_parquet(TABLES_CACHE, index=False)
    return tabs, gmeta


def load_tables() -> dict:
    T = pd.read_parquet(TABLES_CACHE)
    assert T["year"].max() <= 2020
    return {int(Y): g.drop(columns="eval_year") for Y, g in T.groupby("eval_year")}


def build() -> tuple[pd.DataFrame, dict, dict]:
    tabs, gmeta = dev_tables()
    rows, params = [], {}
    for Y in YEARS:
        Dy = tabs[Y]
        tr, te = Dy[Dy["year"].between(TRAIN_START, Y - 1)], Dy[Dy["year"] == Y]
        out = te[["race_id", "horse_id", "year", "pos", "umaban_num", "block", "is_debut_race", "class_ord", "nonfinish"]].copy()
        for name, feats in SPECS.items():
            btr = P.RaceBatch(tr, feats)
            for kind in ("pl", "set"):
                t0 = time.time()
                beta, lam = P.fit(btr, kind=kind, fit_lam=(kind == "pl"))
                out[f"u_{name}_{kind}"] = te[feats].fillna(0).to_numpy(float) @ beta
                params[f"{Y}_{name}_{kind}"] = {"beta": beta.round(5).tolist(), "lam": lam}
                log("fit", Y, name, kind, "lam", np.round(lam, 3), round(time.time() - t0), "秒")
        rows.append(out)
    return pd.concat(rows, ignore_index=True), params, gmeta


def metrics(U: pd.DataFrame, pay: dict, lam: dict) -> dict:
    res = {}
    cols = [c for c in U.columns if c.startswith("u_")]
    per = {c: [] for c in cols}
    for rid, g in U.groupby("race_id"):
        if rid not in pay:
            continue
        um = g["umaban_num"].to_numpy(int)
        Y = int(g["year"].iloc[0])
        wins = [(s, v) for s, v in pay[rid] if s <= set(um)]
        if not wins:
            continue
        for c in cols:
            _, name, kind = c.split("_")
            lm = tuple(lam[f"{Y}_{name}_{kind}"]["lam"])
            combos, p = P.triple_probs(g[c].to_numpy(float), kind, lm)
            sets = [frozenset(um[x]) for x in combos]
            idx = {s: i for i, s in enumerate(sets)}
            wi = [idx[s] for s, _ in wins]
            order = np.argsort(-p)
            rank = np.empty(len(p), int)
            rank[order] = np.arange(1, len(p) + 1)
            best = min(rank[i] for i in wi)
            # N 点買いの払戻(当たった当たり組の払戻の合計)
            payN = np.zeros(len(NS))
            for i, (s, v) in zip(wi, wins):
                payN += np.where(rank[i] <= np.array(NS), v, 0)
            per[c].append({"race_id": rid, "logscore": float(np.log(np.mean(p[wi]))), "best_rank": int(best),
                           "n_combos": len(p), "pay": payN, "block": g["block"].iloc[0],
                           "debut_or_maiden": bool(g["is_debut_race"].iloc[0] or g["class_ord"].iloc[0] == 0),
                           "field": len(um)})
    for c, lst in per.items():
        d = pd.DataFrame(lst)
        hit = np.array([[r <= n for n in NS] for r in d["best_rank"]])
        payN = np.vstack(d["pay"].to_numpy())
        res[c] = {"n_races": int(len(d)), "logscore": round(float(d["logscore"].mean()), 5),
                  "uniform_logscore": round(float(-np.log(d["n_combos"]).mean()), 5),
                  "hit_rate": {n: round(float(hit[:, i].mean()), 4) for i, n in enumerate(NS) if n in (1, 5, 10, 15, 20, 30, 50)},
                  "roi": {n: round(float(payN[:, i].sum() / (100 * n * len(d))), 4) for i, n in enumerate(NS) if n in (1, 5, 10, 15, 20, 30, 50)}}
        res[c]["_d"] = d
    return res


def main():
    t0 = time.time()
    U, params, gmeta = build()
    U.to_parquet(OUT_DIR / "dev_utils.parquet", index=False)
    pay = load_sanrenpuku(YEARS)
    M = metrics(U, pay, params)
    out = {"years": YEARS, "train_start": TRAIN_START, "specs": SPECS, "params": params, "nonfinish": gmeta,
           "n_nonfinish_test_rows": int(U["nonfinish"].sum()), "models": {}}
    base = M["u_M1_pl"]["_d"].set_index("race_id")
    for c, m in M.items():
        d = m.pop("_d").set_index("race_id")
        j = d.join(base[["logscore"]], rsuffix="_M1pl", how="inner")
        m["logscore_vs_M1pl"] = P.block_boot((j["logscore"] - j["logscore_M1pl"]).to_numpy(), j["block"].to_numpy(), n=1000)
        out["models"][c] = m
        log(c, m["logscore"], "uniform", m["uniform_logscore"], "hit", m["hit_rate"], "roi", m["roi"])
    out["elapsed_sec"] = round(time.time() - t0, 1)
    (OUT_DIR / "dev_f0_metrics.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    log("wrote dev_f0_metrics.json")


if __name__ == "__main__":
    main()
