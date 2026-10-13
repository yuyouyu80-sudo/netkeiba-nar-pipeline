# -*- coding: utf-8 -*-
"""finish_v4 G1(プラン第4版、2026-10-13): F7 の上積みの分解。開発の期間だけを使い、確認期間(2021〜2025年)・2026年の層は見ない。

レビューの指摘: 「月齢」の上積みの多くは年齢(年)の効果かもしれない/生産者の値0(履歴なし)が平均扱い/JRA初出走(移籍)が
未出走の新馬と同じ扱い/生産者は新馬戦の回収率を下げる(人気の牧場に寄る)かもしれない。

- 足す候補(土台は FINAL+生産者):
  - 年齢(年): age_y(レースの年 − 生まれ年)の、同じレースの平均との差。
  - 生まれ月: 月齢の差 − 12×年齢(年)の差(同じ年齢の中で何か月早く生まれたか)と、2歳・3歳春・初出走との交差。
  - 生産者の規模: 前月末までの生産者の重みの合計(本人除外)の log(1+n)と、初出走との交差。0 の生産者を平均と区別する。
  - 移籍初戦: JRA で初めて走る馬のうち、未勝利より上のクラスに出る馬(地方・外国からの移籍が大半)。
- 選び方(Fable の指摘2): 候補は2018〜2019年だけで、1群ずつ足して選ぶ。足す条件は「組の対数スコアの差のブロックの区間の下限が0より上」
  かつ「2018・2019年とも符号が正」。選んだ版(G1*)を2020年で1回だけ確かめる。試した設定は dev_ledger.json に残す。
- 記述: レース内の入れ替えプラセボ(新しい値をレース内で入れ替えると上積みが消えるか)、種類別(新馬戦・未勝利戦・1〜3勝・OP/L・重賞)の
  20点の的中と市場の物差し(開発の期間、評価だけ)。

使い方(.venv_dl の python):
  python g1_decompose.py --tables   # f7_tables.parquet に生産者の規模と年齢(年)を足す → g1_tables.parquet
  python g1_decompose.py --select   # 2018〜2019年で選ぶ → g1_select.json
  python g1_decompose.py --confirm  # 選んだ版を2020年で1回だけ確かめ、プラセボと種類別の記述 → g1_result.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ability_v2 as V  # noqa: E402
import age_breeder as AGB  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import model as MD  # noqa: E402

P, AB, OUT = F0.P, F0.AB, F0.OUT_DIR
TABLES = OUT / "g1_tables.parquet"
SELECT_OUT = OUT / "g1_select.json"
RESULT_OUT = OUT / "g1_result.json"
LEDGER = OUT / "dev_ledger.json"
SEL_YEARS = (2018, 2019)
CHECK_YEAR = 2020
log = F0.log

BASE = AGB.FINAL + AGB.BREED
CAND = {"年齢(年)": ["ageyr_dev"],
        "生まれ月": ["mo_dev", "mo_dev×2歳", "mo_dev×3歳春", "mo_dev×初出走"],
        "生産者の規模": ["breeder_n", "新×breeder_n"],
        "移籍初戦": ["移籍初戦"]}
F7 = AGB.FINAL + AGB.AGE + AGB.BREED


# ------------------------------------------------------------------ 値
def group_weight(ra: pd.DataFrame, fig: pd.Series, end_year: int, key: str) -> pd.Series:
    """前月末までの group(key)の重みの合計(本人を除く)。group_month_value の分母と同じ量(縮めの k は足さない)。"""
    r = V.resid_table(ra, fig, end_year)
    ev = V.horse_events(ra, r)
    ev[key] = ra.loc[ev.index, key]
    ev = ev.dropna(subset=[key]).copy()
    ev["dsw"] = ev["w"] - ev["w_prev"]
    ev = ev.sort_values([key, "date"])
    ev["Sw"] = ev.groupby(key)["dsw"].cumsum()
    qq = ra[["horse_id", "month0", key]].reset_index().rename(columns={"index": "_i"}).dropna(subset=[key]).sort_values("month0")
    a = pd.merge_asof(qq, ev[[key, "date", "Sw"]].sort_values("date"), left_on="month0", right_on="date", by=key, allow_exact_matches=False)
    own = ev[["horse_id", "date", "w"]].sort_values("date").rename(columns={"date": "d_own"})
    a = pd.merge_asof(a.sort_values("month0"), own, left_on="month0", right_on="d_own", by="horse_id", allow_exact_matches=False)
    out = pd.Series(0.0, index=ra.index)
    out[a["_i"].to_numpy()] = (a["Sw"].fillna(0) - a["w"].fillna(0)).clip(lower=0).to_numpy()
    return out


def build_tables(years=F0.YEARS, max_year: int = 2020, base_cache=None, out=None):
    T = pd.read_parquet(base_cache or AGB.TABLES)
    assert T["year"].max() <= max_year
    ra = AGB.attach_profile(V.load_runs(max_year))
    parts = []
    for Y in years:
        rY = ra[ra["year"] <= Y]
        fin = rY[~rY["nonfinish"]]
        fig = AB.apply_fig(rY, AB.fit_fig(fin, Y - 1))
        n = group_weight(rY, fig, Y - 1, "breeder_id").rename("breeder_sw")
        add = pd.concat([rY[["race_id", "horse_id"]], n], axis=1).set_index(["race_id", "horse_id"])
        add = add[~add.index.duplicated()]
        TY = T[T["eval_year"] == Y].join(add, on=["race_id", "horse_id"])
        log("g1 tables", Y, len(TY), "生産者の重み0", round(float((TY["breeder_sw"] == 0).mean()), 4))
        parts.append(TY)
    R = pd.concat(parts, ignore_index=True)
    R.to_parquet(out or TABLES, index=False)
    return R


def add_g1(D: pd.DataFrame) -> pd.DataFrame:
    D = AGB.add_age(D)
    ok = D.groupby("race_id")["age_m"].transform(lambda s: s.notna().all())
    age_y = D["year"] - pd.to_numeric(D["horse_id"].str[:4], errors="coerce")
    ydev = (age_y - age_y.groupby(D["race_id"]).transform("mean")).where(ok, 0.0).fillna(0.0)
    D["ageyr_dev"] = ydev
    mo = (D["age_dev"] - 12.0 * ydev).where(ok, 0.0)
    new = (D["k_n"] == 0).astype(float)
    D["mo_dev"] = mo
    D["mo_dev×2歳"] = mo * D["is2"]
    D["mo_dev×3歳春"] = mo * D["is3spring"]
    D["mo_dev×初出走"] = mo * new
    D["breeder_n"] = np.log1p(D["breeder_sw"].fillna(0.0))
    D["新×breeder_n"] = new * D["breeder_n"]
    D["移籍初戦"] = (new * (D["class_ord"] >= 1).astype(float)).fillna(0.0)
    return D


def load_tables(cache=None, max_year: int = 2020, permute: bool = False, seed: int = 20261013) -> dict:
    T = pd.read_parquet(cache or TABLES)
    assert T["year"].max() <= max_year
    out = {}
    rng = np.random.default_rng(seed)
    for Y, g in T.groupby("eval_year"):
        g = g.drop(columns="eval_year").copy()
        if permute:   # プラセボ: 評価年のレースだけ、新しい値(月齢・生産者の値・重み)を馬ごとまとめてレース内で入れ替える
            sub = g[g["year"] == Y]
            src = np.empty(len(sub), dtype=sub.index.dtype)
            pos = 0
            dst = []
            for _, ix in sub.groupby("race_id").groups.items():
                ix = np.asarray(ix)
                src[pos:pos + len(ix)] = rng.permutation(ix)
                dst.append(ix)
                pos += len(ix)
            dst = np.concatenate(dst)
            for c in ["age_m", "breeder_m", "breeder_sw"]:
                g.loc[dst, c] = g.loc[src, c].to_numpy()
        out[int(Y)] = add_g1(MD.derived(g))
    return out


# ------------------------------------------------------------------ 当てはめ・評価
def eval_specs(tabs: dict, specs: dict, years) -> tuple[dict, dict]:
    pay = F0.load_sanrenpuku(years)
    LL, CM = {k: [] for k in specs}, {k: [] for k in specs}
    for Y in years:
        D = tabs[Y]
        tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
        for k, feats in specs.items():
            beta, lam = MD.fit_std(tr, feats, 0.0)
            ll, cm, _ = MD.test_eval(te, feats, beta, lam, pay)
            LL[k].append(ll)
            CM[k].append(cm)
            log(Y, k, "ll", round(float(ll.mean()), 4), "combo", round(float(cm["logscore"].mean()), 4))
    return {k: pd.concat(v) for k, v in LL.items()}, {k: pd.concat(v) for k, v in CM.items()}


def diff(CM, LL, a: str, b: str, blk) -> dict:
    c = CM[a].join(CM[b], rsuffix="_b", how="inner")
    d = c["logscore"] - c["logscore_b"]
    hit = lambda s: (s <= 20).astype(float)  # noqa: E731
    return {"combo": P.block_boot(d.to_numpy(), blk.loc[d.index].to_numpy(), n=1000),
            "by_year": d.groupby(d.index.str[:4]).mean().round(5).to_dict(),
            "hit20": round(float((hit(c["best_rank"]) - hit(c["best_rank_b"])).mean()), 4),
            "ll": round(float((LL[a] - LL[b]).dropna().mean()), 5)}


def ledger(entry: dict):
    L = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    L.append(entry | {"time": time.strftime("%Y-%m-%d %H:%M:%S")})
    LEDGER.write_text(json.dumps(L, ensure_ascii=False, indent=1, default=float), encoding="utf-8")


def select() -> dict:
    tabs = load_tables()
    blk = tabs[F0.YEARS[-1]].drop_duplicates("race_id").set_index("race_id")["block"]
    specs = {"F7": F7, "B(FINAL+生産者)": BASE}
    for g, cols in CAND.items():
        specs[f"B+{g}"] = BASE + cols
    LL, CM = eval_specs(tabs, specs, SEL_YEARS)
    single = {k: diff(CM, LL, k, "B(FINAL+生産者)", blk) for k in specs if k != "B(FINAL+生産者)"}
    # 1群ずつ前向きに足す(2018〜2019年)。毎回、残りの候補から差のいちばん大きいものを試し、条件を満たせば足す
    chosen, cur_name, cur = [], "B(FINAL+生産者)", list(BASE)
    steps = []
    left = list(CAND)
    while left:
        trial = {f"{cur_name}+{g}": cur + CAND[g] for g in left}
        L2, C2 = eval_specs(tabs, trial | {cur_name: cur}, SEL_YEARS)
        res = {k: diff(C2, L2, k, cur_name, blk) for k in trial}
        best = max(res, key=lambda k: res[k]["combo"]["mean"])
        r = res[best]
        ok = r["combo"]["ci_bca"][0] > 0 and all(v > 0 for v in r["by_year"].values())
        steps.append({"base": cur_name, "tried": {k: v for k, v in res.items()}, "best": best, "added": ok})
        g = best.split("+")[-1]
        if not ok:
            break
        chosen.append(g)
        cur_name, cur = best, trial[best]
        left.remove(g)
    out = {"select_years": SEL_YEARS, "candidates": CAND, "single_vs_B": single, "steps": steps, "chosen": chosen,
           "g1_spec": cur, "g1_name": cur_name, "f7_vs_B": single["F7"]}
    ledger({"stage": "G1 select", "n_specs_tried": len(specs) + sum(len(s["tried"]) for s in steps), "chosen": chosen})
    return out


def confirm() -> dict:
    sel = json.loads(SELECT_OUT.read_text(encoding="utf-8"))
    tabs = load_tables()
    blk = tabs[F0.YEARS[-1]].drop_duplicates("race_id").set_index("race_id")["block"]
    specs = {"FINAL": AGB.FINAL, "F7": F7, "G1*": sel["g1_spec"]}
    LL, CM = eval_specs(tabs, specs, (CHECK_YEAR,))
    res = {"g1_name": sel["g1_name"], "chosen": sel["chosen"], "check_year": CHECK_YEAR,
           "G1_vs_F7_2020": diff(CM, LL, "G1*", "F7", blk), "G1_vs_FINAL_2020": diff(CM, LL, "G1*", "FINAL", blk),
           "F7_vs_FINAL_2020": diff(CM, LL, "F7", "FINAL", blk)}
    # プラセボ(2018〜2020年): 新しい値をレース内で入れ替えた表で、G1* と F7 が FINAL を上回らないこと
    tp = load_tables(permute=True)
    LLp, CMp = eval_specs(tp, {"FINAL": AGB.FINAL, "F7(入れ替え)": F7, "G1*(入れ替え)": sel["g1_spec"]}, F0.YEARS)
    res["placebo"] = {k: diff(CMp, LLp, k, "FINAL", blk) for k in ("F7(入れ替え)", "G1*(入れ替え)")}
    ledger({"stage": "G1 confirm 2020", "g1_name": sel["g1_name"], "combo_vs_F7_2020": res["G1_vs_F7_2020"]["combo"]["mean"]})
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--confirm", action="store_true")
    a = ap.parse_args()
    if a.tables:
        build_tables()
    if a.select:
        r = select()
        SELECT_OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log("G1 chosen", r["chosen"])
    if a.confirm:
        assert not RESULT_OUT.exists(), "2020年の確かめは1回だけ(g1_result.json が既にある)"
        r = confirm()
        RESULT_OUT.write_text(json.dumps(r, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
        log("G1 2020", r["G1_vs_F7_2020"]["combo"])


if __name__ == "__main__":
    main()
