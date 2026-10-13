# -*- coding: utf-8 -*-
"""finish_v4 G3(プラン第4版、2026-10-13): 本人の実績を広げる。別の会話の近5走の特徴(analysis/past5_unused_2026_10_12/p1_build.py、
取消「取」「除」の修正後の features.parquet)を読み、オッズ・騎手・馬体重を使わない系統だけを F7 に足す。今回の枠とコース形状×脚質も足す
(2026-10-13 ユーザー了承)。

- 取り込む系統(p1_build の名前): B 前走の馬番、C 通過順の推移、D 上がりのレース内順位、G 間隔・叩き2走目、H 距離・芝ダの変更、
  I 同じ競馬場、K 前走の頭数、L 前走のレースレベル、着順・着差(used_form・used_margin3/5・used_first_corner)。
- 取り込まないもの: A 前走の人気(過去の市場の情報。案B の人気の推定で使う)、J 馬体重、騎手の乗り替わり、
  E 道悪(確定した当日の馬場を使うため発走前に分からない)、F 斤量の増減(d_kin と同じ)。
- 新しく作るもの: 今回の枠(馬番の相対位置と、ダート・距離との交差)、コース形状×脚質(前に行く度合い C_c4_rel3 × 直線の長さ・高低差・左回り)。
  コース形状は data/jra_course_master.csv(競馬場×芝ダ)。
- 選び方: G1 と同じ。2018〜2019年だけで1系統ずつ足し、組の対数スコアの差の区間の下限が0より上かつ2年とも正なら足す。
  2020年はここでは見ない(G* の凍結のときに1回だけ)。
出力: g3_select.json、dev_ledger.json に試した数。
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import f0_dev_utils as F0  # noqa: E402
import g1_decompose as G1  # noqa: E402

OUT = F0.OUT_DIR
ROOT = F0.C.PROJECT_ROOT
P5 = ROOT / "analysis" / "past5_unused_2026_10_12" / "out" / "features.parquet"
SELECT_OUT = OUT / "g3_select.json"
log = F0.log

GROUPS = {
    "着順・着差": ["used_form", "used_margin3", "used_margin5"],
    "通過順の推移": ["used_first_corner", "C_c4_rel3", "C_gain_c1c4", "C_gain_c4fin"],
    "上がりの順位": ["D_ag_rel3", "D_ag_rel_best5"],
    "間隔・叩き2走目": ["G_log_rest", "G_tataki2"],
    "距離・芝ダの変更": ["H_dist_chg", "H_dist_abs", "H_surf_chg", "H_same_cond_best"],
    "同じ競馬場": ["I_same_course", "I_course_change"],
    "前走の頭数": ["K_field1"],
    "前走のレースレベル": ["L_race_level1"],
    "前走の馬番": ["B_uma1_rel", "B_outer_x_beat1"],
    "今回の枠": ["uma_t", "uma_t×ダ", "uma_t×距離"],
    "コース形状×脚質": ["前×直線", "前×高低差", "前×左回り"],
}
P5_COLS = sorted({c for g in GROUPS.values() for c in g if c[:2] in ("us", "B_", "C_", "D_", "G_", "H_", "I_", "K_", "L_")})


def course_table() -> pd.DataFrame:
    m = pd.read_csv(ROOT / "data" / "jra_course_master.csv", dtype=str)
    rows = []
    for _, r in m.iterrows():
        for s, pre in [("芝", "turf"), ("ダ", "dirt")]:
            rows.append({"venue_code": r["venue_code"].zfill(2), "surface": s, "straight": pd.to_numeric(r[f"{pre}_straight_m"], errors="coerce"),
                         "elev": pd.to_numeric(r[f"{pre}_elevation_m"], errors="coerce"), "left": float(r["turn_direction"].startswith("左"))})
    C = pd.DataFrame(rows)
    C["straight_c"] = (C["straight"] - C["straight"].mean()) / C["straight"].std()
    C["elev_c"] = (C["elev"] - C["elev"].mean()) / C["elev"].std()
    return C.set_index(["venue_code", "surface"])


def add_g3(D: pd.DataFrame, F: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    D = D.join(F, on=["race_id", "horse_id"])
    for c in P5_COLS:
        D[c] = D[c].astype(float).fillna(0.0)
    D["uma_t"] = ((D["umaban_num"] - 1) / (D["n_start"] - 1).clip(lower=1) - 0.5).fillna(0.0)
    D["uma_t×ダ"] = D["uma_t"] * D["is_dirt"]
    D["uma_t×距離"] = D["uma_t"] * D["logd_c"]
    key = pd.MultiIndex.from_arrays([D["race_id"].str[4:6], np.where(D["is_dirt"] > 0, "ダ", "芝")])
    cc = C.reindex(key)
    front = D["C_c4_rel3"]            # 大きいほど4角で前(p1_build は −相対順位)
    D["前×直線"] = front * cc["straight_c"].fillna(0).to_numpy()
    D["前×高低差"] = front * cc["elev_c"].fillna(0).to_numpy()
    D["前×左回り"] = front * cc["left"].fillna(0).to_numpy()
    return D


def permute_rows(D: pd.DataFrame, cols: list, Y: int, rng) -> pd.DataFrame:
    """プラセボ: 評価年のレースだけ、cols を馬ごとまとめてレース内で入れ替える。"""
    D = D.copy()
    sub = D[D["year"] == Y]
    src, dst = [], []
    for _, ix in sub.groupby("race_id").groups.items():
        ix = np.asarray(ix)
        src.append(rng.permutation(ix))
        dst.append(ix)
    src, dst = np.concatenate(src), np.concatenate(dst)
    D.loc[dst, cols] = D.loc[src, cols].to_numpy()
    return D


def placebo():
    """採用した系統の値(G3 で足した列)をレース内で入れ替えると、F7 への上積みが消えるかを2018〜2019年で見る。"""
    sel = json.loads(SELECT_OUT.read_text(encoding="utf-8"))
    F = pd.read_parquet(P5, columns=["race_id", "horse_id"] + P5_COLS).drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])
    C = course_table()
    cols = [c for g in sel["chosen"] for c in GROUPS[g]]
    rng = np.random.default_rng(20261013)
    tabs = {Y: permute_rows(add_g3(t, F, C), cols, Y, rng) for Y, t in G1.load_tables().items() if Y in G1.SEL_YEARS}
    blk = pd.concat([t.drop_duplicates("race_id").set_index("race_id")["block"] for t in tabs.values()])
    blk = blk[~blk.index.duplicated()]
    LL, CM = G1.eval_specs(tabs, {"F7": list(G1.F7), "G3*(入れ替え)": sel["g3_spec"]}, G1.SEL_YEARS)
    sel["placebo_vs_F7"] = G1.diff(CM, LL, "G3*(入れ替え)", "F7", blk)
    SELECT_OUT.write_text(json.dumps(sel, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    G1.ledger({"stage": "G3 placebo", "n_specs_tried": 1})
    log("placebo", sel["placebo_vs_F7"]["combo"], sel["placebo_vs_F7"]["by_year"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--placebo", action="store_true")
    if ap.parse_args().placebo:
        return placebo()
    F = pd.read_parquet(P5, columns=["race_id", "horse_id"] + P5_COLS).drop_duplicates(["race_id", "horse_id"]).set_index(["race_id", "horse_id"])
    C = course_table()
    tabs = {Y: add_g3(t, F, C) for Y, t in G1.load_tables().items() if Y in G1.SEL_YEARS}
    blk = pd.concat([t.drop_duplicates("race_id").set_index("race_id")["block"] for t in tabs.values()])
    blk = blk[~blk.index.duplicated()]
    cov = {Y: round(float(t.loc[t["year"] == Y, "used_form"].ne(0).mean()), 4) for Y, t in tabs.items()}
    log("近5走の特徴が付いた割合(評価年の行)", cov)
    base_name, base = "F7", list(G1.F7)
    single_specs = {"F7": base} | {f"F7+{g}": base + cols for g, cols in GROUPS.items()}
    LL, CM = G1.eval_specs(tabs, single_specs, G1.SEL_YEARS)
    single = {k: G1.diff(CM, LL, k, "F7", blk) for k in single_specs if k != "F7"}
    chosen, cur_name, cur, steps = [], base_name, base, []
    left = [g for g in GROUPS if single[f"F7+{g}"]["combo"]["mean"] > 0]
    n_tried = len(single_specs)
    while left:
        trial = {f"{cur_name}+{g}": cur + GROUPS[g] for g in left}
        L2, C2 = G1.eval_specs(tabs, trial | {cur_name: cur}, G1.SEL_YEARS)
        n_tried += len(trial)
        res = {k: G1.diff(C2, L2, k, cur_name, blk) for k in trial}
        best = max(res, key=lambda k: res[k]["combo"]["mean"])
        r = res[best]
        ok = r["combo"]["ci_bca"][0] > 0 and all(v > 0 for v in r["by_year"].values())
        steps.append({"base": cur_name, "tried": res, "best": best, "added": ok})
        log("step", best, round(r["combo"]["mean"], 5), [round(x, 5) for x in r["combo"]["ci_bca"]], "added", ok)
        if not ok:
            break
        g = best[len(cur_name) + 1:]
        chosen.append(g)
        cur_name, cur = best, trial[best]
        left.remove(g)
    total = None
    if chosen:
        LLt, CMt = G1.eval_specs(tabs, {"F7": base, "G3*": cur}, G1.SEL_YEARS)
        total = G1.diff(CMt, LLt, "G3*", "F7", blk)
    out = {"select_years": G1.SEL_YEARS, "groups": GROUPS, "coverage": cov, "single_vs_F7": single, "steps": steps,
           "chosen": chosen, "g3_spec": cur, "g3_name": cur_name, "g3_vs_F7": total,
           "features_parquet_sha256": hashlib.sha256(P5.read_bytes()).hexdigest()}
    SELECT_OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    G1.ledger({"stage": "G3 select", "n_specs_tried": n_tried, "chosen": chosen})
    log("G3 chosen", chosen)


if __name__ == "__main__":
    main()
