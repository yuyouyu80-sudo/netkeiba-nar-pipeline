# -*- coding: utf-8 -*-
"""finish_v4 G*(プラン第4版の修正、2026-10-13): G1〜G3・G5 で採ったものを1本にまとめて凍結し、2020年で1回だけ確かめる。

G* の中身: G1 は採用なし(2020年で F7 を下回った)、G2 は斤量を今のまま(ユーザー決定)、G3 は g3_select.json の7系統。
よって G* = G3*(F7 + G3 の7系統)。買い方は G5 案B の chosen(g5_select.json)。

手順: --freeze で gstar_manifest.json(仕様・入力の sha・判定の基準)を書く → コミット・タグ finish_v4_gstar_prereg →
--check2020 を1回だけ(結果のファイルがあれば・タグが無ければ・未コミットの変更があれば・sha が違えば拒否)。

判定の基準(--freeze の前に決めた。manifest にも同じ文を書く):
- GS1(予想): 2020年の G* − F7 の組の対数スコアの差の、ブロックの区間(BCa)の下限 > 0。2018・2019年は既に正なので、満たせば3年とも同じ符号。
  合格なら G* を前向きの影の記録に足す(F7 は残す)。不合格なら G* は記録しない。
- GS2(買い方、推定): 2020年の、案B の chosen と w0(G* の上位20組)の人気をそろえた上積みの差の点推定 > 0。
  GS1 合格かつ GS2 が正なら、案B の買い目を前向きの影の記録に足す。区間は書くが、1年(約3,300レース)では区間が広いので合否に使わない。
- 記述: 的中率・回収率の絶対値、FINAL との差、w0.5 と実際のオッズの参考版。
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
import age_breeder as AGB  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import g1_decompose as G1  # noqa: E402
import g3_past5 as G3  # noqa: E402
import g5_popest as G5  # noqa: E402
import pl_ext as P  # noqa: E402

OUT = F0.OUT_DIR
ROOT = F0.C.PROJECT_ROOT
MANIFEST = OUT / "gstar_manifest.json"
RESULT = OUT / "gstar_2020_result.json"
TAG = "finish_v4_gstar_prereg"
CHECK_YEAR = 2020
log = F0.log
CRITERIA = {
    "GS1": "2020年の G* − F7 の組の対数スコアの差のブロック区間(BCa)の下限 > 0。合格なら G* を前向きの影の記録に足す",
    "GS2": "GS1 合格のとき、2020年の案B chosen − w0 の人気をそろえた上積みの差の点推定 > 0 なら、案B の買い目を前向きの影の記録に足す(区間は記述)",
}
INPUTS = {"g3_select": G3.SELECT_OUT, "g5_select": G5.SELECT_OUT, "features_parquet": G3.P5, "g1_tables": G1.TABLES,
          "course_master": ROOT / "data" / "jra_course_master.csv"}
CODE = ["gstar.py", "g5_popest.py", "g3_past5.py", "g1_decompose.py", "model.py", "pl_ext.py", "market_yardstick.py", "age_breeder.py"]


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def spec() -> tuple[list, str]:
    s3 = json.loads(G3.SELECT_OUT.read_text(encoding="utf-8"))
    s5 = json.loads(G5.SELECT_OUT.read_text(encoding="utf-8"))
    assert s5["chosen"], "案B の chosen が無い"
    return s3["g3_spec"], s5["chosen"]


def freeze():
    assert not MANIFEST.exists(), "gstar_manifest.json は既にある"
    feats, rule = spec()
    m = {"date": time.strftime("%Y-%m-%d %H:%M:%S"), "g_star_spec": feats, "n_features": len(feats),
         "spec_sha256": hashlib.sha256(json.dumps(feats, ensure_ascii=False).encode("utf-8")).hexdigest(),
         "buy_rule": rule, "pop_features": G5.POP, "pool": G5.POOL, "l2_pop": G5.L2, "check_year": CHECK_YEAR, "criteria": CRITERIA,
         "inputs_sha256": {k: sha(p) for k, p in INPUTS.items()}}
    MANIFEST.write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
    log("manifest", MANIFEST, m["n_features"], rule)


def guard():
    assert not RESULT.exists(), "2020年の確かめは1回だけ(gstar_2020_result.json が既にある)"
    assert MANIFEST.exists(), "先に --freeze"
    r = subprocess.run(["git", "rev-parse", "-q", "--verify", f"refs/tags/{TAG}"], cwd=ROOT, capture_output=True, text=True)
    assert r.returncode == 0, f"タグ {TAG} が無い"
    paths = [str(HERE / c) for c in CODE] + [str(MANIFEST), str(G3.SELECT_OUT), str(G5.SELECT_OUT)]
    st = subprocess.run(["git", "status", "--porcelain", "--"] + paths, cwd=ROOT, capture_output=True, text=True).stdout.strip()
    assert not st, f"未コミットの変更がある: {st}"
    m = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for k, p in INPUTS.items():
        assert sha(p) == m["inputs_sha256"][k], f"入力の sha が違う: {k}"
    feats, rule = spec()
    assert feats == m["g_star_spec"] and rule == m["buy_rule"]
    return m


def check2020():
    m = guard()
    feats, rule = m["g_star_spec"], m["buy_rule"]
    F = pd.read_parquet(G3.P5, columns=["race_id", "horse_id"] + G3.P5_COLS).drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])
    C = G3.course_table()
    tabs = {CHECK_YEAR: G3.add_g3(G1.load_tables()[CHECK_YEAR], F, C)}
    blk = tabs[CHECK_YEAR].drop_duplicates("race_id").set_index("race_id")["block"]
    LL, CM = G1.eval_specs(tabs, {"FINAL": AGB.FINAL, "F7": G1.F7, "G*": feats}, (CHECK_YEAR,))
    res = {"manifest_date": m["date"], "check_year": CHECK_YEAR, "G*_vs_F7": G1.diff(CM, LL, "G*", "F7", blk),
           "G*_vs_FINAL": G1.diff(CM, LL, "G*", "FINAL", blk), "F7_vs_FINAL": G1.diff(CM, LL, "F7", "FINAL", blk)}
    hit = {k: round(float((CM[k]["best_rank"] <= 20).mean()), 4) for k in CM}
    res["hit20"] = hit
    gs1 = bool(res["G*_vs_F7"]["combo"]["ci_bca"][0] > 0)
    # 買い方(案B)
    rules = ["w0", rule, "w0.5", "ref_mkt_w0.5"]
    rules = list(dict.fromkeys(rules))
    R, qual = G5.run_years((CHECK_YEAR,), feats, rules, races_out=OUT / "gstar_2020_races.parquet")
    res["popularity_quality"] = qual
    res["rules"] = G5.summarize(R, rules, (CHECK_YEAR,))
    st = np.full(len(R), 2000.0)
    d = (R[f"{rule}_act"] - R[f"{rule}_exp"]) - (R["w0_act"] - R["w0_exp"])
    res["uplift_chosen_minus_w0"] = P.ratio_boot(d.to_numpy(float), st, R["block"].to_numpy(), n=1000)
    gs2 = bool(gs1 and res["uplift_chosen_minus_w0"]["ratio"] > 0)
    res["judgement"] = {"GS1": gs1, "GS2": gs2, "criteria": CRITERIA}
    RESULT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    G1.ledger({"stage": "G* check 2020", "GS1": gs1, "GS2": gs2, "combo_vs_F7_2020": res["G*_vs_F7"]["combo"]["mean"]})
    log("GS1", gs1, res["G*_vs_F7"]["combo"]["mean"], res["G*_vs_F7"]["combo"]["ci_bca"], "hit20", hit)
    log("GS2", gs2, res["uplift_chosen_minus_w0"]["ratio"], res["uplift_chosen_minus_w0"]["ci_bca"])
    for r_ in rules:
        v = res["rules"][r_]
        log(r_, v["hit"], round(v["roi"]["ratio"], 4), round(v["uplift"]["ratio"], 4), [round(x, 4) for x in v["uplift"]["ci_bca"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--check2020", action="store_true")
    a = ap.parse_args()
    if a.freeze:
        freeze()
    if a.check2020:
        check2020()


if __name__ == "__main__":
    main()
