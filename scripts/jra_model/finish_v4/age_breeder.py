# -*- coding: utf-8 -*-
"""finish_v4 F7: 生年月日(月齢)と生産者を完成形に足す(2026-10-12、プラン第3版 F2-4・F2-6)。

生年月日の基準(2014〜2020年の2〜3歳の走でカバー率98%以上)を、収集の完了後に 100% で満たしたので着手した。
確認(2021〜2025年)は完成形(FINAL)で1回使用済み。月齢・生産者はそこで一度も使っていないので、足した分だけを
事前登録の追補(PREREG_FINISH_V4.md §12)の新しい仮説として、同じ期間で1回だけ確かめる(finish_v4_eval.py --f7-final)。

- **生産者(breeder_m):** 母の値(dam_m)と同じ作り方。各走の後の馬の値 −(その月の父の値)を、前月末までの生産者ごとに、
  本人を除いて縮めた平均(group_month_value)。生産者と父の交絡は「父の値を引いた残り」で扱う(同時の混合モデルの近似)。
  縮め方 k_b は2017年末までの初出走の能力の予測誤差で選ぶ(k_s・k_d と同じ方法、k_s・k_d は既定値のまま)。
  生産者は data/horse_profile の現在の値。生産者は生まれた時点で決まり、後から変わらないので時点の問題は無い(馬主は使わない)。
- **月齢:** 発走日の日齢(月)の、同じレースの出走馬の平均との差(age_dev)。南半球産も実際の日齢なので、そのまま扱う。
  交差: 2歳・3歳の1〜6月・初出走。レースの出走馬に1頭でも生年月日の無い馬がいれば、そのレースの月齢の値はすべて0
  (欠けの指示変数や平均埋めは使わない、プランの約束)。
- 開発の評価: 2018〜2020年(係数はその前年まで、2014年から)。凍結の規則は F3 と同じ(外すと組の対数スコアの95%区間が
  0より下になる群だけを残す)。

使い方(.venv_dl の python):
  python age_breeder.py --tune     # k_b を選ぶ(2017年末まで) → f7_tune.json
  python age_breeder.py --tables   # 2018〜2020年の評価の表に月齢・生産者を足す → f7_tables.parquet
  python age_breeder.py --eval     # 開発の評価 → f7_eval_dev.json・f7_utils.parquet
"""
import argparse
import json
import sys
import time

import numpy as np
import pandas as pd

from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ability_v2 as V  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import model as MD  # noqa: E402

P, AB = F0.P, F0.AB
OUT = F0.OUT_DIR
PROFILE = OUT / "profile.parquet"
TUNE_OUT = OUT / "f7_tune.json"
TABLES = OUT / "f7_tables.parquet"
EVAL_OUT = OUT / "f7_eval_dev.json"
UTILS_OUT = OUT / "f7_utils.parquet"
K_B_GRID = [10, 20, 40, 80, 160, 320, 640]
log = F0.log

AGE = ["age_dev", "age_dev×2歳", "age_dev×3歳春", "age_dev×初出走"]
BREED = ["breeder_m", "新×breeder_m"]
F7_GROUPS = {"月齢": AGE, "生産者": BREED}
FINAL = MD.SPECS["FULL"]          # 確認で凍結した完成形(7群・70本)
F7_SPECS = {"FINAL": FINAL, "FINAL+月齢": FINAL + AGE, "FINAL+生産者": FINAL + BREED, "FINAL+両方": FINAL + AGE + BREED}


# ------------------------------------------------------------------ 値
def attach_profile(ra: pd.DataFrame) -> pd.DataFrame:
    pf = pd.read_parquet(PROFILE).drop_duplicates("horse_id").set_index("horse_id")
    ra = ra.copy()
    ra["birth_date"] = ra["horse_id"].map(pf["birth_date"])
    ra["breeder_id"] = ra["horse_id"].map(pf["breeder_id"]).replace("", np.nan)
    ra["age_m"] = (ra["date"] - ra["birth_date"]).dt.days / 30.4375
    return ra


def breeder_month(ra: pd.DataFrame, fig: pd.Series, end_year: int, k_s: float, k_b: float) -> pd.Series:
    """前月末までの生産者の値(父の値を引いた残りの縮約平均、本人除外)。"""
    r = V.resid_table(ra, fig, end_year)
    ev = V.horse_events(ra, r)
    ev["breeder_id"] = ra.loc[ev.index, "breeder_id"]
    sire_m = V.group_month_value(ev, "ped_S_horse_id", "x", ra, k_s)
    ev["sire_at"] = sire_m.reindex(ev.index).to_numpy()
    ev["zb"] = ev["x"] - ev["sire_at"]
    return V.group_month_value(ev, "breeder_id", "zb", ra, k_b)


def age_table(ra: pd.DataFrame) -> pd.DataFrame:
    """走ごとの月齢と交差のもと(レース内の平均との差は評価の表で作る。表のレースの候補と同じ馬で平均するため)。"""
    return pd.DataFrame({"age_m": ra["age_m"], "is2": (ra["age"] == 2).astype(float),
                         "is3spring": ((ra["age"] == 3) & (ra["date"].dt.month <= 6)).astype(float)}, index=ra.index)


def add_age(D: pd.DataFrame) -> pd.DataFrame:
    D = D.copy()
    ok = D.groupby("race_id")["age_m"].transform(lambda s: s.notna().all())
    dev = (D["age_m"] - D.groupby("race_id")["age_m"].transform("mean")).where(ok, 0.0).fillna(0.0)
    D["age_dev"] = dev
    D["age_dev×2歳"] = dev * D["is2"]
    D["age_dev×3歳春"] = dev * D["is3spring"]
    D["age_dev×初出走"] = dev * (D["k_n"] == 0).astype(float)
    D["新×breeder_m"] = (D["k_n"] == 0).astype(float) * D["breeder_m"]
    return D


# ------------------------------------------------------------------ k_b の選択(2017年末まで)
def tune() -> dict:
    t0 = time.time()
    ra = attach_profile(V.load_runs(V.TUNE_END))
    par = MD.v2_par()
    fin = ra[~ra["nonfinish"]]
    fig = AB.apply_fig(ra, AB.fit_fig(fin, V.TUNE_END))
    S = AB.sire_S(ra)
    mv = V.monthly_values(ra, fig, V.TUNE_END, par["k_s"], par["k_d"])
    deb = (ra["career"] == 0) & fig.notna() & S.notna() & ~ra["nonfinish"]
    fit_m, tst = deb & (ra["year"] <= 2014), deb & ra["year"].between(2015, 2017)

    def mse(cols):
        X = lambda m: np.column_stack([np.ones(m.sum())] + [c[m] for c in cols])  # noqa: E731
        b = np.linalg.lstsq(X(fit_m), fig[fit_m], rcond=None)[0]
        return float(np.mean((fig[tst] - X(tst) @ b) ** 2)), [float(x) for x in b]

    base_cols = [S, mv["sire_m"], mv["dam_m"]]
    res = {"end": V.TUNE_END, "k_s": par["k_s"], "k_d": par["k_d"], "mse_without_breeder": mse(base_cols)[0], "grid": [],
           "breeder_coverage_debut_test": float(ra.loc[tst, "breeder_id"].notna().mean())}
    for k_b in K_B_GRID:
        bm = breeder_month(ra, fig, V.TUNE_END, par["k_s"], k_b)
        m, b = mse(base_cols + [bm])
        res["grid"].append({"k_b": k_b, "mse": m, "b": b})
        log("k_b", k_b, round(m, 5), "(生産者なし", round(res["mse_without_breeder"], 5), ")")
    res["k_b"] = min(res["grid"], key=lambda r: r["mse"])["k_b"]
    # 月齢: 初出走の能力と日齢(2歳の新馬、2015〜2017年)の関係(記述)
    d2 = tst & (ra["age"] == 2)
    res["debut2_age_corr"] = float(np.corrcoef(ra.loc[d2, "age_m"].fillna(ra.loc[d2, "age_m"].mean()), fig[d2])[0, 1])
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


# ------------------------------------------------------------------ 評価の表
def build_tables(years=F0.YEARS, max_year: int = 2020, base_cache=None, out=None, k_b=None):
    """model の表(base_cache、既定は f3_tables.parquet)に月齢・生産者を足す。生産者の値は各評価年 Y の前年末までの係数で作る。"""
    k_b = k_b or json.loads(TUNE_OUT.read_text(encoding="utf-8"))["k_b"]
    par = MD.v2_par()
    T = pd.read_parquet(base_cache or MD.CACHE)
    assert T["year"].max() <= max_year
    ra = attach_profile(V.load_runs(max_year))
    at = age_table(ra)
    out_parts = []
    for Y in years:
        rY = ra[ra["year"] <= Y]
        fin = rY[~rY["nonfinish"]]
        fig = AB.apply_fig(rY, AB.fit_fig(fin, Y - 1))
        bm = breeder_month(rY, fig, Y - 1, par["k_s"], k_b)
        add = pd.concat([rY[["race_id", "horse_id"]], at.loc[rY.index], bm.rename("breeder_m")], axis=1).set_index(["race_id", "horse_id"])
        add = add[~add.index.duplicated()]
        TY = T[T["eval_year"] == Y].join(add, on=["race_id", "horse_id"])
        log("f7 tables", Y, len(TY), "月齢の欠け", round(float(TY["age_m"].isna().mean()), 5), "生産者の値0", round(float((TY["breeder_m"] == 0).mean()), 4))
        out_parts.append(TY)
    R = pd.concat(out_parts, ignore_index=True)
    R.to_parquet(out or TABLES, index=False)
    return R


def load_tables(cache=None, max_year: int = 2020) -> dict:
    T = pd.read_parquet(cache or TABLES)
    assert T["year"].max() <= max_year
    return {int(Y): add_age(MD.derived(g.drop(columns="eval_year"))) for Y, g in T.groupby("eval_year")}


# ------------------------------------------------------------------ 開発の評価(2018〜2020年)
def layer_ids(te: pd.DataFrame) -> dict:
    r = te.drop_duplicates("race_id").set_index("race_id")
    two = te.groupby("race_id")["is2"].max()
    return {"新馬・未勝利": set(r.index[r["class_ord"] == 0]) if "class_ord" in r else set(), "2歳": set(two.index[two > 0]),
            "新馬戦": set(r.index[r["is_debut_race"].astype(bool)])}


def evaluate() -> dict:
    t0 = time.time()
    tabs = load_tables()
    pay = F0.load_sanrenpuku(F0.YEARS)
    blk = tabs[F0.YEARS[-1]].drop_duplicates("race_id").set_index("race_id")["block"]
    specs = dict(F7_SPECS)
    for g in F7_GROUPS:
        specs[f"FINAL+両方−{g}"] = FINAL + [c for gg, cs in F7_GROUPS.items() if gg != g for c in cs]
    LL, CM, U, params = {k: [] for k in specs}, {k: [] for k in specs}, [], {}
    lay = {}
    for Y in F0.YEARS:
        D = tabs[Y]
        tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
        out = te[["race_id", "horse_id", "year", "pos", "umaban_num", "block", "is_debut_race", "class_ord", "nonfinish"]].copy()
        for k, feats in specs.items():
            beta, lam = MD.fit_std(tr, feats, 0.0)
            ll, cm, u = MD.test_eval(te, feats, beta, lam, pay)
            LL[k].append(ll)
            CM[k].append(cm)
            if k in F7_SPECS:
                out[f"u_{k}"] = u
                params[f"{Y}_{k}"] = {"lam": list(lam), "beta": {f: round(float(b), 6) for f, b in zip(feats, beta)}}
            log(Y, k, "ll", round(float(ll.mean()), 4), "combo", round(float(cm["logscore"].mean()), 4), round(time.time() - t0), "秒")
        for name, ids in layer_ids(te).items():
            lay.setdefault(name, set()).update(ids)
        U.append(out)
    LL = {k: pd.concat(v) for k, v in LL.items()}
    CM = {k: pd.concat(v) for k, v in CM.items()}
    res = {"eval_years": F0.YEARS, "k_b": json.loads(TUNE_OUT.read_text(encoding="utf-8"))["k_b"], "groups": F7_GROUPS,
           "vs_FINAL": MD.summarize({k: LL[k] for k in F7_SPECS}, {k: CM[k] for k in F7_SPECS}, blk, "FINAL")}
    both = "FINAL+両方"
    res["drop_one"] = {k: v for k, v in MD.summarize({k: LL[k] for k in [both] + [f"{both}−{g}" for g in F7_GROUPS]},
                                                      {k: CM[k] for k in [both] + [f"{both}−{g}" for g in F7_GROUPS]}, blk, both).items() if k != both}
    keep = [g for g in F7_GROUPS if res["drop_one"][f"{both}−{g}"][f"combo_vs_{both}"]["ci_bca"][1] < 0]
    res["kept_groups"] = keep
    res["f7_spec_name"] = "FINAL+" + "・".join(keep) if keep else "FINAL"
    res["f7_spec"] = FINAL + [c for g in keep for c in F7_GROUPS[g]]
    # 層ごとの差(記述): 組の対数スコア
    res["by_layer_combo_vs_FINAL"] = {}
    for k in F7_SPECS:
        if k == "FINAL":
            continue
        c = CM[k].join(CM["FINAL"], rsuffix="_b", how="inner")
        d = c["logscore"] - c["logscore_b"]
        res["by_layer_combo_vs_FINAL"][k] = {name: {"n": int(d.index.isin(list(ids)).sum()), "mean": round(float(d[d.index.isin(list(ids))].mean()), 5)}
                                             for name, ids in lay.items()}
    res["coef"] = {k: {f: params[f"{Y}_{k}"]["beta"].get(f) for Y in F0.YEARS for f in (AGE + BREED) if f in params[f"{Y}_{k}"]["beta"]}
                   for k in F7_SPECS if k != "FINAL"}
    res["coef_by_year"] = {f"{Y}_{both}": {f: params[f"{Y}_{both}"]["beta"][f] for f in AGE + BREED} for Y in F0.YEARS}
    pd.concat(U, ignore_index=True).to_parquet(UTILS_OUT, index=False)
    (OUT / "f7_params.json").write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--eval", action="store_true")
    a = ap.parse_args()
    if a.tune:
        r = tune()
        TUNE_OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1), encoding="utf-8")
        log("k_b", r["k_b"], r["elapsed_sec"], "秒")
    if a.tables:
        build_tables()
    if a.eval:
        r = evaluate()
        EVAL_OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log("kept", r["kept_groups"], r["elapsed_sec"], "秒")


if __name__ == "__main__":
    main()
