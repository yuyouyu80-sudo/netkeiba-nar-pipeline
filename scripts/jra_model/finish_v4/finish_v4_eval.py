# -*- coding: utf-8 -*-
"""finish_v4 F5: 事前登録した確認の実行(2026-10-11、プラン第3版 F5)。事前登録: data/jra_pipeline/finish_v4/PREREG_FINISH_V4.md

凍結したもの(FROZEN): 完成形の特徴量(model.SPECS["FULL"]、F3 の凍結の規則で7群すべてを残した)、L2=0、MC なし、
主ルール N20(sanrenpuku_rules.MAIN_RULE)、見送りの閾値(副次)、状態・縮約の設定(ability_tune.json・ability_v2_tune.json、2017年末まで)。
係数は評価年 Y ごとに 2014〜Y−1年で推定し直す(ローリング)。市場は評価の物差しだけに使う。

  --final-dry   試運転: 2018〜2020年に本番と同じ処理を当てる(開発の数値が再現することの確認)
  --freeze      入力の sha を prereg_manifest.json に書く(事前登録のコミット・タグの前に1回)
  --final       本番: 2021〜2025年。1回だけ。次のどれかなら実行を拒否する:
                結果が既にある/finish_v4 に未コミットの変更がある/タグ finish_v4_prereg が無いか HEAD と中身が違う/入力の sha が manifest と違う
引数名に --dry は使わない(プランのエンジニアリングの約束)。
"""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import f0_market_eval as FM  # noqa: E402   評価だけ(物差し)
import market_yardstick as MY  # noqa: E402   評価だけ(物差し)
import model as MD  # noqa: E402
import pl_ext as P  # noqa: E402
import sanrenpuku_rules as SR  # noqa: E402

OUT = F0.OUT_DIR
ROOT = F0.C.PROJECT_ROOT
LEDGER = OUT / "run_ledger.json"
MANIFEST = OUT / "prereg_manifest.json"
TAG = "finish_v4_prereg"
MODES = {"final_dry": {"years": (2018, 2019, 2020), "max_year": 2020},
         "final": {"years": (2021, 2022, 2023, 2024, 2025), "max_year": 2025}}
FROZEN = {"spec": MD.SPECS["FULL"], "l2": 0.0, "mc": False, "main_rule": SR.MAIN_RULE, "sub_rules": list(SR.SUB_RULES),
          "skip_pcov_threshold": None}     # 見送りの閾値は freeze で開発の値から決めて manifest に書く
COMPARE = {"M1": F0.SPECS["M1"], "M5": F0.SPECS["M5"], "M6": MD.SPECS["M6"], "FINAL": FROZEN["spec"]}
RULES = ["N10", "N20", "N30"]
INPUTS = ["data/jra_pipeline/pedigree_reports/radar_v2/runs_all.parquet", "data/jra_pipeline/pedigree_reports/radar_v2/dl_table.parquet",
          "data/jra_pipeline/pedigree_reports/radar_v2/runs_ext.parquet", "data/jra_pipeline/pedigree_reports/radar_v2/stamina_race.parquet",
          "data/jra_pipeline/pedigree_reports/radar_v2/race_req.parquet", "data/jra_pipeline/pedigree_reports/radar_v2/stamina_l3.parquet",
          "data/jra_pipeline/pedigree_reports/radar_v2/ability_tune.json", "data/jra_pipeline/finish_v4/ability_v2_tune.json",
          "data/jra_pipeline/finish_v4/dam_ids.parquet"]
DIRS = [("data/race_results", range(2011, 2026)), ("data/payouts", range(2014, 2026)), ("data/pedigree", None)]
PED_MAX_YEAR = 2025      # 血統ファイルは、2025年までに走った馬の分だけを sha に入れる(収集用の会話が毎日足す新馬の分は入れない)
log = F0.log


# ------------------------------------------------------------------ 入力の sha・台帳
def sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def input_manifest() -> dict:
    out = {"files": {}, "dirs": {}}
    for rel in INPUTS:
        p = ROOT / rel
        out["files"][rel] = sha_file(p) if p.exists() else None
    for rel, years in DIRS:
        h = hashlib.sha256()
        base = ROOT / rel
        if years is None:     # 血統は数が多いので、ファイル名と大きさだけ(2025年までに走った馬)
            for hid in ped_horses():
                p = base / f"{hid}.csv"
                h.update(f"{hid}:{p.stat().st_size if p.exists() else -1}".encode())
        else:
            for y in years:
                for p in sorted((base / str(y)).glob("*.csv")):
                    h.update(p.name.encode())
                    h.update(sha_file(p).encode())
        out["dirs"][rel] = h.hexdigest()
    return out


def ped_horses() -> list:
    r = pd.read_parquet(ROOT / "data/jra_pipeline/pedigree_reports/radar_v2/runs_all.parquet", columns=["horse_id", "year"])
    return sorted(r.loc[r["year"] <= PED_MAX_YEAR, "horse_id"].unique())


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, encoding="utf-8").stdout.strip()


def guard_final(result: Path):
    errs = []
    if result.exists():
        errs.append(f"結果が既にある: {result.name}")
    dirty = git("status", "--porcelain", "--", "scripts/jra_model/finish_v4", "data/jra_pipeline/finish_v4/PREREG_FINISH_V4.md",
                "data/jra_pipeline/finish_v4/prereg_manifest.json")
    if dirty:
        errs.append("finish_v4 に未コミットの変更がある:\n" + dirty)
    if not git("tag", "--list", TAG):
        errs.append(f"タグ {TAG} が無い")
    elif git("diff", "--stat", TAG, "HEAD", "--", "scripts/jra_model/finish_v4", "data/jra_pipeline/finish_v4/PREREG_FINISH_V4.md"):
        errs.append(f"タグ {TAG} と HEAD で finish_v4 の中身が違う")
    if not MANIFEST.exists():
        errs.append("prereg_manifest.json が無い(--freeze をしていない)")
    else:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))
        now = input_manifest()
        diff = [k for k in man["inputs"]["files"] if man["inputs"]["files"][k] != now["files"].get(k)] + \
               [k for k in man["inputs"]["dirs"] if man["inputs"]["dirs"][k] != now["dirs"].get(k)]
        if diff:
            errs.append("入力の sha が manifest と違う: " + ", ".join(diff))
    if errs:
        raise SystemExit("本番の実行を拒否:\n- " + "\n- ".join(errs))


def append_ledger(entry: dict):
    L = json.loads(LEDGER.read_text(encoding="utf-8")) if LEDGER.exists() else []
    L.append(entry)
    LEDGER.write_text(json.dumps(L, ensure_ascii=False, indent=1), encoding="utf-8")


# ------------------------------------------------------------------ 実行
def tables(mode: str):
    m = MODES[mode]
    f0c, f3c = OUT / f"{mode}_tables_f0.parquet", OUT / f"{mode}_tables.parquet"
    if not f3c.exists():
        F0.dev_tables(m["years"], m["max_year"], f0c)
        MD.build_tables(m["years"], m["max_year"], f0c, f3c)
    return f3c


def year_table(cache: Path, Y: int, max_year: int) -> pd.DataFrame:
    T = pd.read_parquet(cache, filters=[("eval_year", "=", Y)]).drop(columns="eval_year")
    assert T["year"].max() <= max_year and T["year"].max() == Y
    return MD.derived(T)


def run(mode: str, skip_thr: float | None) -> dict:
    t0 = time.time()
    m = MODES[mode]
    cache = tables(mode)
    years, max_year = m["years"], m["max_year"]
    pay = F0.load_sanrenpuku(range(2014, max_year + 1))
    odds = MY.load_odds(range(2014, max_year + 1))
    assert odds["year"].max() <= max_year
    LL = {k: [] for k in COMPARE}
    CM = {k: [] for k in COMPARE}
    rows, logit = [], []
    for Y in years:
        D = year_table(cache, Y, max_year)
        tr, te = D[D["year"].between(F0.TRAIN_START, Y - 1)], D[D["year"] == Y]
        U = {}
        lams = {}
        for k, feats in COMPARE.items():
            beta, lam = MD.fit_std(tr, feats, FROZEN["l2"] if k == "FINAL" else 0.0)
            ll, cm, u = MD.test_eval(te, feats, beta, lam, pay)
            LL[k].append(ll)
            CM[k].append(cm)
            U[k], lams[k] = u, lam
            log(mode, Y, k, "ll", round(float(ll.mean()), 4), "combo", round(float(cm["logscore"].mean()), 4), round(time.time() - t0), "秒")
        # 市場の物差し(評価だけ)
        trm = odds[odds["year"].between(2014, Y - 1)]
        gamma, mlam = MY.fit_market(trm)
        Tg = MY.build_return_table(trm, pay, gamma, mlam)
        ug = dict(tuple(te.assign(**{f"u_{k}": U[k] for k in COMPARE}).groupby("race_id")))
        for rid, g in odds[odds["year"] == Y].groupby("race_id"):
            if rid not in pay or rid not in ug or len(g) < 3:
                continue
            w = MY.win_keys(pay[rid])
            c_m, p_m, um_all = MY.market_combos(g, gamma, mlam)
            k_m = MY.combo_keys(c_m)
            e_m = Tg.expect(np.log(p_m), len(um_all))
            ex_of = dict(zip(k_m.tolist(), e_m.tolist()))
            pm_of = dict(zip(k_m.tolist(), np.log(p_m).tolist()))
            om = np.argsort(-p_m)
            rk = np.empty(len(p_m), int)
            rk[om] = np.arange(1, len(p_m) + 1)
            rank_of = dict(zip(k_m.tolist(), rk.tolist()))
            u = ug[rid]
            umod = u["umaban_num"].to_numpy(int)
            same = set(umod.tolist()) == set(um_all.tolist())
            n = len(um_all)
            rec = {"race_id": rid, "year": Y, "block": g["block"].iloc[0], "field": n, "band": FM.band_label(n), "same": same,
                   "debut_or_maiden": bool(u["is_debut_race"].iloc[0] or u["class_ord"].iloc[0] == 0)}
            for rule in RULES:
                ii = FM.pick(p_m, rule)
                kk = k_m[ii]
                rec[f"mkt_hit_{rule}"] = int(any(int(x) in w for x in kk))
                rec[f"mkt_act_{rule}"] = sum(w.get(int(x), 0) for x in kk)
                rec[f"mkt_exp_{rule}"] = float(e_m[ii].sum())
                rec[f"mkt_k_{rule}"] = len(kk)
                rec[f"mkt_pcov_{rule}"] = float(p_m[ii].sum())
            for k in COMPARE:
                cc, pp = P.triple_probs(u[f"u_{k}"].to_numpy(float), "pl", tuple(lams[k]))
                kk_all = MY.combo_keys(umod[cc])
                for rule in RULES:
                    ii = FM.pick(pp, rule)
                    kk = kk_all[ii]
                    rec[f"{k}_hit_{rule}"] = int(any(int(x) in w for x in kk))
                    rec[f"{k}_act_{rule}"] = sum(w.get(int(x), 0) for x in kk)
                    rec[f"{k}_exp_{rule}"] = float(sum(ex_of.get(int(x), 0.0) for x in kk))
                    rec[f"{k}_k_{rule}"] = len(kk)
                    rec[f"{k}_pcov_{rule}"] = float(pp[ii].sum())
                    rec[f"{k}_mrank_{rule}"] = float(np.median([rank_of.get(int(x), len(p_m)) for x in kk]))
                if k == "FINAL" and same:
                    lpk = np.array([pm_of.get(int(x), np.nan) for x in kk_all])
                    widx = [i for i, x in enumerate(kk_all.tolist()) if x in w]
                    if widx and np.isfinite(lpk).all():
                        logit.append((np.log(pp), lpk, widx, g["block"].iloc[0]))
            rows.append(rec)
        log(mode, Y, "market", round(time.time() - t0), "秒")
    LL = {k: pd.concat(v) for k, v in LL.items()}
    CM = {k: pd.concat(v) for k, v in CM.items()}
    R = pd.DataFrame(rows)
    R.to_parquet(OUT / f"{mode}_races.parquet", index=False)
    blk_all = pd.concat([R.set_index("race_id")["block"]])
    res = {"mode": mode, "years": years, "frozen": {k: v for k, v in FROZEN.items() if k != "spec"} | {"n_features": len(FROZEN["spec"])},
           "n_races_ll": int(len(LL["FINAL"])), "n_races_market": int(len(R)), "candidates_same_as_market": int(R["same"].sum())}
    res |= hypotheses(LL, CM, R, logit, skip_thr)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def boot_pair(a: pd.Series, b: pd.Series, blk: pd.Series) -> dict:
    d = (a - b).dropna()
    return P.block_boot(d.to_numpy(), blk.reindex(d.index).to_numpy(), n=2000)


def hypotheses(LL, CM, R, logit, skip_thr) -> dict:
    blk = pd.concat([v["block"] for v in [R.set_index("race_id")]])
    # LL・組の指標のブロック(R に無いレースがあれば race_id の日付・場から作る)
    allid = LL["FINAL"].index
    blk = blk.reindex(allid)
    miss = blk.isna()
    if miss.any():
        blk[miss] = ["?" + x[4:6] + x[:4] for x in allid[miss]]
    out = {}
    c = CM["FINAL"].join(CM["M1"], rsuffix="_M1", how="inner")
    h2a = boot_pair(c["logscore"], c["logscore_M1"], blk)
    hit = lambda s, n: (s <= n).astype(float)  # noqa: E731
    h2b = boot_pair(hit(c["best_rank"], 20), hit(c["best_rank_M1"], 20), blk)
    h2 = bool(h2a["ci_bca"][0] > 0 and h2b["ci_bca"][0] > 0)
    out["H2"] = {"combo_logscore_FINAL_minus_M1": h2a, "hit_N20_FINAL_minus_M1": h2b, "passed": h2,
                 "hit_N20": {"FINAL": round(float(hit(c["best_rank"], 20).mean()), 4), "M1": round(float(hit(c["best_rank_M1"], 20).mean()), 4)}}
    h1b = boot_pair(LL["FINAL"], LL["M5"], blk)
    out["H1b"] = {"ll_FINAL_minus_M5": h1b, "tested": h2, "passed": bool(h2 and h1b["ci_bca"][0] > 0)}
    inc = FM.increment_logit(logit)
    out["H4"] = {"increment": inc, "tested": out["H1b"]["passed"], "passed": bool(out["H1b"]["passed"] and inc["z_a"] > 1.96)}
    out["H1a_safety"] = {"ll_M5_minus_M1": boot_pair(LL["M5"], LL["M1"], blk)}
    # H3(推定)と副次(記述)
    b = R["block"].to_numpy()
    est = {}
    for who in ["mkt", "M1", "M5", "M6", "FINAL"]:
        for rule in RULES:
            a, e = R[f"{who}_act_{rule}"].to_numpy(float), R[f"{who}_exp_{rule}"].to_numpy(float)
            st = 100.0 * R[f"{who}_k_{rule}"].to_numpy(float)
            top5 = np.argsort(-a)[:5]
            keep5 = np.ones(len(a), bool)
            keep5[top5] = False
            r_ = {"hit_rate": round(float(R[f"{who}_hit_{rule}"].mean()), 4), "roi": P.ratio_boot(a, st, b, n=1000),
                  "expected_roi_market": round(float(e.sum() / st.sum()), 4), "uplift": P.ratio_boot(a - e, st, b, n=1000),
                  "roi_excl_50k": round(float(np.where(a > 50000, 0, a).sum() / st.sum()), 4),
                  "roi_excl_top5": round(float(a[keep5].sum() / st[keep5].sum()), 4),
                  "median_market_rank": round(float(R[f"{who}_mrank_{rule}"].median()), 1) if who != "mkt" else None,
                  "by_year_uplift": {int(y): round(float(((a - e)[R["year"] == y]).sum() / st[R["year"] == y].sum()), 4) for y in sorted(R["year"].unique())},
                  "by_layer": {}}
            for lab, msk in [(x, (R["band"] == x).to_numpy()) for x in FM.LAYER_NAMES] + \
                    [("新馬・未勝利", R["debut_or_maiden"].to_numpy()), ("それ以外", ~R["debut_or_maiden"].to_numpy())]:
                r_["by_layer"][lab] = {"n": int(msk.sum()), "hit_rate": round(float(R.loc[msk, f"{who}_hit_{rule}"].mean()), 4),
                                       "roi": round(float(a[msk].sum() / st[msk].sum()), 4),
                                       "uplift": round(float((a - e)[msk].sum() / st[msk].sum()), 4)}
            est[f"{who}_{rule}"] = r_
    u = est[f"FINAL_{FROZEN['main_rule']}"]["uplift"]
    lo, hi = u["ci_bca"]
    out["H3_estimate"] = {"uplift": u, "label": "あり" if lo > 0 else ("意味のある差なし" if (lo > -0.03 and hi < 0.03) else "判定できない")}
    out["rules"] = est
    # 見送り(副次): 買った組の確率の合計が閾値未満のレースを見送る。陰性対照は市場の上位20組に同じ割合の見送り
    if skip_thr is not None:
        sk = {}
        for who, thr in [("FINAL", skip_thr), ("mkt", float(np.quantile(R["mkt_pcov_N20"], (R["FINAL_pcov_N20"] < skip_thr).mean())))]:
            keep = R[f"{who}_pcov_N20"] >= thr
            a, e = R[f"{who}_act_N20"].to_numpy(float), R[f"{who}_exp_N20"].to_numpy(float)
            st = 100.0 * R[f"{who}_k_N20"].to_numpy(float)
            sk[who] = {"threshold": thr, "skip_rate": round(float(1 - keep.mean()), 4),
                       "roi_bought": round(float(a[keep].sum() / st[keep].sum()), 4), "hit_bought": round(float(R.loc[keep, f"{who}_hit_N20"].mean()), 4),
                       "uplift_bought": P.ratio_boot((a - e)[keep], st[keep], b[keep.to_numpy()], n=1000)}
        out["skip"] = sk
    return out


def freeze():
    """開発の値から見送りの閾値を決め、入力の sha と一緒に manifest に書く。"""
    import ability_v2 as V
    import nonfinish as NF
    # 母の ID のキャッシュを 2025年までの全馬(中止の馬を含む)で埋めておく(本番の途中で入力の sha が変わらないように)
    nf = NF.started_nonfinishers(PED_MAX_YEAR)
    V.dam_ids(sorted(set(ped_horses()) | set(nf["horse_id"])))
    R = pd.read_parquet(OUT / "final_dry_races.parquet")
    thr = float(np.quantile(R["FINAL_pcov_N20"], 0.2))      # 開発(2018〜2020年)の下位20%
    man = {"made": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": git("rev-parse", "HEAD"), "skip_pcov_threshold": thr,
           "skip_rule": "主ルール N20 の買い目の確率の合計が、開発の下位20%点未満なら見送り(副次・記述)",
           "inputs": input_manifest()}
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log("manifest", MANIFEST, "skip_thr", round(thr, 4))


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--final-dry", action="store_true")
    g.add_argument("--freeze", action="store_true")
    g.add_argument("--final", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
        return
    mode = "final" if a.final else "final_dry"
    result = OUT / f"{mode}_result.json"
    thr = None
    if a.final:
        guard_final(result)
        thr = json.loads(MANIFEST.read_text(encoding="utf-8"))["skip_pcov_threshold"]
    elif MANIFEST.exists():
        thr = json.loads(MANIFEST.read_text(encoding="utf-8"))["skip_pcov_threshold"]
    res = run(mode, thr)
    res["git_head"] = git("rev-parse", "HEAD")
    result.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    append_ledger({"mode": mode, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": res["git_head"], "result": result.name,
                   "H2": res["H2"]["passed"], "H1b": res["H1b"]["passed"], "H4": res["H4"]["passed"]})
    log(mode, "H2", res["H2"]["passed"], "H1b", res["H1b"]["passed"], "H4", res["H4"]["passed"], "H3", res["H3_estimate"]["label"])


if __name__ == "__main__":
    main()
