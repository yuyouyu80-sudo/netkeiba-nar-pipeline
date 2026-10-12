# -*- coding: utf-8 -*-
"""finish_v4 F7 の確認(事前登録の追補 PREREG_FINISH_V4.md §12、2026-10-12)。

完成形 FINAL(確認済み・凍結)に、月齢と生産者(age_breeder.py、開発で2群とも残った)を足した F7 を、FINAL と比べる。
FINAL の土台の表は、確認のときに作った表(final_tables.parquet など)をそのまま使い、月齢・生産者の列だけを足す。
係数は評価年 Y ごとに 2014〜Y−1年で推定し直す(ローリング)。市場は評価の物差しだけに使う。

  --f7-dry     試運転: 2018〜2020年に本番と同じ処理を当てる(開発の数値が再現することの確認)
  --f7-freeze  入力の sha を f7_manifest.json に書く(追補のコミット・タグ finish_v4_f7_prereg の前に1回)
  --f7-final   本番: 2021〜2025年。1回だけ。結果が既にある/未コミットの変更がある/タグが無いか HEAD と中身が違う/
               入力の sha が manifest と違う、のどれかなら拒否する
  --f7-y2026   本番の後に1回だけ: 2026年1〜10月の層(記述のみ)
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
import age_breeder as AGB  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import f0_market_eval as FM  # noqa: E402   評価だけ(物差し)
import finish_v4_eval as EV  # noqa: E402

OUT = F0.OUT_DIR
TAG = "finish_v4_f7_prereg"
MANIFEST = OUT / "f7_manifest.json"
MODES = {"f7_dry": {"years": (2018, 2019, 2020), "max_year": 2020, "base": "final_dry_tables.parquet"},
         "f7_final": {"years": (2021, 2022, 2023, 2024, 2025), "max_year": 2025, "base": "final_tables.parquet"},
         "f7_y2026": {"years": (2026,), "max_year": 2026, "base": "y2026_tables.parquet"}}
F7_SPEC = AGB.FINAL + AGB.AGE + AGB.BREED      # 開発の凍結の規則で月齢・生産者の2群とも残った(f7_eval_dev.json)
COMPARE = {"FINAL": AGB.FINAL, "F7": F7_SPEC}
EXTRA_INPUTS = ["data/jra_pipeline/finish_v4/profile.parquet", "data/jra_pipeline/finish_v4/f7_tune.json"]
log = F0.log


def tables(mode: str) -> Path:
    m = MODES[mode]
    out = OUT / f"{mode}_tables.parquet"
    if not out.exists():
        AGB.build_tables(m["years"], m["max_year"], OUT / m["base"], out)
    return out


def manifest_now(mode: str = "f7_final") -> dict:
    man = EV.input_manifest()
    for rel in EXTRA_INPUTS + [f"data/jra_pipeline/finish_v4/{MODES[mode]['base']}"]:
        p = F0.C.PROJECT_ROOT / rel
        man["files"][rel] = EV.sha_file(p) if p.exists() else None
    return man


def guard(result: Path):
    errs = []
    if result.exists():
        errs.append(f"結果が既にある: {result.name}")
    dirty = EV.git("status", "--porcelain", "--", "scripts/jra_model/finish_v4", "data/jra_pipeline/finish_v4/PREREG_FINISH_V4.md",
                   "data/jra_pipeline/finish_v4/f7_manifest.json")
    if dirty:
        errs.append("finish_v4 に未コミットの変更がある:\n" + dirty)
    if not EV.git("tag", "--list", TAG):
        errs.append(f"タグ {TAG} が無い")
    elif EV.git("diff", "--stat", TAG, "HEAD", "--", "scripts/jra_model/finish_v4", "data/jra_pipeline/finish_v4/PREREG_FINISH_V4.md"):
        errs.append(f"タグ {TAG} と HEAD で finish_v4 の中身が違う")
    if not MANIFEST.exists():
        errs.append("f7_manifest.json が無い(--f7-freeze をしていない)")
    else:
        man = json.loads(MANIFEST.read_text(encoding="utf-8"))["inputs"]
        now = manifest_now()
        diff = [k for k in man["files"] if man["files"][k] != now["files"].get(k)] + [k for k in man["dirs"] if man["dirs"][k] != now["dirs"].get(k)]
        if diff:
            errs.append("入力の sha が manifest と違う: " + ", ".join(diff))
    if errs:
        raise SystemExit("本番の実行を拒否:\n- " + "\n- ".join(errs))


def layer_diff(CM, R, blk) -> dict:
    c = CM["F7"].join(CM["FINAL"], rsuffix="_b", how="inner")
    d = c["logscore"] - c["logscore_b"]
    meta = R.set_index("race_id")
    out = {}
    for lab, ids in [("新馬・未勝利", meta.index[meta["debut_or_maiden"]]), ("それ以外", meta.index[~meta["debut_or_maiden"]])] + \
            [(x, meta.index[meta["band"] == x]) for x in FM.LAYER_NAMES]:
        dd = d[d.index.isin(ids)]
        out[lab] = {"n": int(len(dd)), "mean": round(float(dd.mean()), 5)}
    out["by_year"] = d.groupby(d.index.str[:4]).mean().round(5).to_dict()
    return out


def hypotheses(LL, CM, R, logit) -> dict:
    blk = R.set_index("race_id")["block"].reindex(LL["FINAL"].index)
    miss = blk.isna()
    if miss.any():
        blk[miss] = ["?" + x[4:6] + x[:4] for x in blk.index[miss]]
    c = CM["F7"].join(CM["FINAL"], rsuffix="_F", how="inner")
    hit = lambda s, n: (s <= n).astype(float)  # noqa: E731
    h5 = EV.boot_pair(c["logscore"], c["logscore_F"], blk)
    out = {"H5": {"combo_logscore_F7_minus_FINAL": h5, "passed": bool(h5["ci_bca"][0] > 0)}}
    h5b = EV.boot_pair(hit(c["best_rank"], 20), hit(c["best_rank_F"], 20), blk)
    out["H5b"] = {"hit_N20_F7_minus_FINAL": h5b, "tested": out["H5"]["passed"], "passed": bool(out["H5"]["passed"] and h5b["ci_bca"][0] > 0),
                  "hit_N20": {"F7": round(float(hit(c["best_rank"], 20).mean()), 4), "FINAL": round(float(hit(c["best_rank_F"], 20).mean()), 4)}}
    h5c = EV.boot_pair(LL["F7"], LL["FINAL"], blk)
    out["H5c"] = {"ll_F7_minus_FINAL": h5c, "tested": out["H5b"]["passed"], "passed": bool(out["H5b"]["passed"] and h5c["ci_bca"][0] > 0)}
    out["increment_F7"] = FM.increment_logit(logit)            # 記述(FINAL は H4 で確認済み)
    est = EV.rule_estimates(R, ["mkt", "FINAL", "F7"])
    u = est["F7_N20"]["uplift"]
    a7, e7 = R["F7_act_N20"].to_numpy(float), R["F7_exp_N20"].to_numpy(float)
    aF, eF = R["FINAL_act_N20"].to_numpy(float), R["FINAL_exp_N20"].to_numpy(float)
    st = 100.0 * R["F7_k_N20"].to_numpy(float)
    out["H6_estimate"] = {"uplift_F7": u, "label": EV.uplift_label(*u["ci_bca"]),
                          "uplift_F7_minus_FINAL": F0.P.ratio_boot((a7 - e7) - (aF - eF), st, R["block"].to_numpy(), n=1000)}
    out["rules"] = est
    out["layers_combo_F7_minus_FINAL"] = layer_diff(CM, R, blk)
    return out


def run(mode: str) -> dict:
    t0 = time.time()
    m = MODES[mode]
    LL, CM, R, logit = EV.collect(mode, m["years"], m["max_year"], tables(mode), COMPARE, inc_model="F7", prep=AGB.add_age,
                                  l2_of=lambda k: 0.0)
    res = {"mode": mode, "years": m["years"], "f7_spec_added": AGB.AGE + AGB.BREED, "k_b": json.loads(AGB.TUNE_OUT.read_text(encoding="utf-8"))["k_b"],
           "n_races_ll": int(len(LL["F7"])), "n_races_market": int(len(R)), "candidates_same_as_market": int(R["same"].sum())}
    res |= hypotheses(LL, CM, R, logit)
    res["elapsed_sec"] = round(time.time() - t0, 1)
    return res


def freeze():
    import ability_v2 as V
    import nonfinish as NF
    # 母の ID の控えを 2025年までの全馬で埋めておく(本番の途中で dam_ids.parquet の sha が変わらないように。EV.freeze と同じ)
    nf = NF.started_nonfinishers(EV.PED_MAX_YEAR)
    V.dam_ids(sorted(set(EV.ped_horses()) | set(nf["horse_id"])))
    man ={"made": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": EV.git("rev-parse", "HEAD"), "f7_spec": F7_SPEC, "inputs": manifest_now()}
    MANIFEST.write_text(json.dumps(man, ensure_ascii=False, indent=1), encoding="utf-8")
    log("manifest", MANIFEST)


def main():
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--f7-dry", action="store_true")
    g.add_argument("--f7-freeze", action="store_true")
    g.add_argument("--f7-final", action="store_true")
    g.add_argument("--f7-y2026", action="store_true")
    a = ap.parse_args()
    if a.f7_freeze:
        freeze()
        return
    mode = "f7_final" if a.f7_final else ("f7_y2026" if a.f7_y2026 else "f7_dry")
    result = OUT / f"{mode}_result.json"
    if a.f7_final:
        guard(result)
    if a.f7_y2026:
        assert (OUT / "f7_final_result.json").exists(), "2026年の層は確認(--f7-final)の後にだけ見る"
        assert not result.exists(), "2026年の層は一度だけ見る"
    res = run(mode)
    res["git_head"] = EV.git("rev-parse", "HEAD")
    result.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    EV.append_ledger({"mode": mode, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "git_head": res["git_head"], "result": result.name,
                      "H5": res["H5"]["passed"], "H5b": res["H5b"]["passed"], "H5c": res["H5c"]["passed"]})
    log(mode, "H5", res["H5"]["passed"], "H5b", res["H5b"]["passed"], "H5c", res["H5c"]["passed"], "H6", res["H6_estimate"]["label"])


if __name__ == "__main__":
    main()
