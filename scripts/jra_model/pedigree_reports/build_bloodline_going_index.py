# -*- coding: utf-8 -*-
"""系統(父系統・母父系統・父父系統)×馬場状態(良/稍重/重/不良)の「系統適性指数」を、2016年以降の
全JRA芝・ダート戦から算出する(2026-10-03新設)。

## 指数の定義
- 1走ごとの残差 r = (3着以内なら1、でなければ0) − 3/出走頭数(頭数による複勝圏の入りやすさを差し引いた超過)。
- プール = サーフェス(芝/ダ)×距離帯(短距離/マイル/中距離/長距離、PROF.distance_bucketと同じ)。
  距離帯ごとに分けるのは、馬場の影響の出方が短距離とそれ以外で違うため。
- 系統の総合効果 m(系統, プール) = Σr / (N + K)  (K=100の縮約。全馬場合算)
- 馬場別の指数 idx(系統, プール, 馬場) = (Σr_(系統,馬場) + K × m) / (n_(系統,馬場) + K) × 100  [複勝率pt]
  → データの少ない(系統,馬場)セルは、その系統の総合効果へ縮約される(=「道悪でも総合効果と同じ」と仮定)。
- 系統分類はnetkeiba血統ビームの色分け(bloodline_name_map.json)。分類できない馬は集計から除外。
- 父・母父・父父は別々のロールとして集計する(「父系統」「母父系統」「父父系統」)。

## 検証(前後半分割)
前半(〜2020年)の指数が後半(2021年〜)の残差をどれだけ予測するか(回帰の傾き、1に近いほど指数が
そのまま再現)と、前後半の指数の相関を、(a)総合の指数 (b)馬場別の上乗せ分(idx−m) で出力する。
"""
import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build_race_radar_data as B  # noqa: E402

PROJECT_ROOT = B.PROJECT_ROOT
SCRATCH = B.SCRATCH
PROF = B.PROF
NAME_CACHE = SCRATCH / "horse_pedigree_names_cache.pkl"
BLOODLINE_MAP_PATH = B.BLOODLINE_MAP_PATH
OUT_PATH = SCRATCH / "bloodline_going_index.json"

GOINGS = ["良", "稍重", "重", "不良"]
ROLES = [("sire", "S"), ("bms", "DS"), ("ss", "SS")]
K = 100
SPLIT_YEAR = 2021  # 前半=〜2020年、後半=2021年〜


def load_names(horse_ids):
    cache = {}
    if NAME_CACHE.exists():
        cache = pickle.loads(NAME_CACHE.read_bytes())
    todo = [h for h in horse_ids if h not in cache]
    print(f"血統名キャッシュ: 既存{len(cache)}頭、新規読込{len(todo)}頭")
    for i, h in enumerate(todo):
        p = B.pedigree_csv_path(h)
        if not p.exists():
            cache[h] = None
            continue
        with open(p, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            hdr = next(r)
            row = next(r)
        d = dict(zip(hdr, row))
        cache[h] = tuple((d.get(f"ped_{code}_name_ja") or None) for _k, code in ROLES)
    if todo:
        NAME_CACHE.write_bytes(pickle.dumps(cache))
    return cache


def build_long():
    res = B.load_all_results()
    res = res[res["finish_pos_num"].notna() & res["surface"].isin(["芝", "ダ"]) & res["going"].isin(GOINGS)
              & res["distance_num"].notna()].copy()
    res["field"] = res.groupby("race_id")["finish_pos_num"].transform("count")
    res = res[res["field"] >= 5]
    res["hit"] = (res["finish_pos_num"] <= 3).astype(float)
    res["r"] = res["hit"] - np.minimum(3.0 / res["field"], 1.0)
    res["bucket"] = res["distance_num"].map(PROF.distance_bucket)
    res["year"] = res["race_date_dt"].dt.year
    names = load_names(list(res["horse_id"].unique()))
    bmap = json.loads(BLOODLINE_MAP_PATH.read_text(encoding="utf-8"))
    for i, (role, _code) in enumerate(ROLES):
        res[f"cat_{role}"] = res["horse_id"].map(
            lambda h, i=i: bmap.get(names.get(h)[i]) if names.get(h) and names.get(h)[i] else None)
    return res


def index_table(df, role):
    """df(1ロール分、cat_role非NaN)から {(surface,bucket,going,cat): dict} を返す。"""
    col = f"cat_{role}"
    d = df[df[col].notna()]
    main = d.groupby(["surface", "bucket", col])["r"].agg(["sum", "count"])
    main["m"] = main["sum"] / (main["count"] + K)
    cell = d.groupby(["surface", "bucket", "going", col])["r"].agg(["sum", "count"]).reset_index()
    cell = cell.merge(main[["m", "count"]].rename(columns={"count": "n_all"}).reset_index(),
                      on=["surface", "bucket", col], how="left")
    cell["idx"] = (cell["sum"] + K * cell["m"]) / (cell["count"] + K) * 100
    cell["raw"] = cell["sum"] / cell["count"] * 100
    cell["m_pt"] = cell["m"] * 100
    return cell.rename(columns={col: "cat"})


def validate(res):
    """前半の指数で後半の残差を予測できるか。戻り値は辞書(役割合算+役割別)。"""
    out = {}
    pre, post = res[res["year"] < SPLIT_YEAR], res[res["year"] >= SPLIT_YEAR]
    out["n_pre_runs"], out["n_post_runs"] = int(len(pre)), int(len(post))
    allx = {"total": [], "going_part": [], "main": []}
    for role, _c in ROLES:
        t1 = index_table(pre, role)
        t1["going_part"] = t1["idx"] - t1["m_pt"]
        d2 = post[post[f"cat_{role}"].notna()].rename(columns={f"cat_{role}": "cat"})
        d2 = d2.merge(t1[["surface", "bucket", "going", "cat", "idx", "going_part", "m_pt"]],
                      on=["surface", "bucket", "going", "cat"], how="inner")
        d2["role"] = role
        allx["total"].append(d2)
    D = pd.concat(allx["total"], ignore_index=True)
    D["y"] = D["r"] * 100

    def slope(x, y, groups):
        x = x - x.mean()
        b = (x * (y - y.mean())).sum() / (x * x).sum()
        resid = (y - y.mean()) - b * x
        # race単位のクラスター頑健SE
        g = pd.DataFrame({"g": groups, "s": x * resid}).groupby("g")["s"].sum()
        se = np.sqrt((g ** 2).sum()) / (x * x).sum()
        return float(b), float(se)

    res_out = {}
    for name, mask in (("全馬場", D["going"].notna()), ("良のみ", D["going"] == "良"),
                       ("道悪(稍重・重・不良)", D["going"] != "良")):
        sub = D[mask]
        b, se = slope(sub["idx"].to_numpy(), sub["y"].to_numpy(), sub["race_id"].to_numpy())
        b2, se2 = slope(sub["m_pt"].to_numpy(), sub["y"].to_numpy(), sub["race_id"].to_numpy())
        b3, se3 = slope(sub["going_part"].to_numpy(), sub["y"].to_numpy(), sub["race_id"].to_numpy())
        # 馬場別の上乗せ分は、総合効果をコントロールした偏回帰の傾き
        X = np.column_stack([np.ones(len(sub)), sub["m_pt"].to_numpy(), sub["going_part"].to_numpy()])
        coef, *_ = np.linalg.lstsq(X, sub["y"].to_numpy(), rcond=None)
        resid = sub["y"].to_numpy() - X @ coef
        # クラスター頑健共分散
        XtXi = np.linalg.inv(X.T @ X)
        meat = np.zeros((3, 3))
        for _g, idxs in pd.Series(np.arange(len(sub))).groupby(sub["race_id"].to_numpy()):
            u = X[idxs.to_numpy()].T @ resid[idxs.to_numpy()]
            meat += np.outer(u, u)
        cov = XtXi @ meat @ XtXi
        res_out[name] = {
            "n_runs": int(len(sub)),
            "slope_total_idx": [round(b, 3), round(se, 3)],
            "slope_main_only": [round(b2, 3), round(se2, 3)],
            "slope_going_part_alone": [round(b3, 3), round(se3, 3)],
            "partial_slope_main": [round(float(coef[1]), 3), round(float(np.sqrt(cov[1, 1])), 3)],
            "partial_slope_going_part": [round(float(coef[2]), 3), round(float(np.sqrt(cov[2, 2])), 3)],
        }
    out["by_subset"] = res_out

    # 前後半の指数の相関(n>=30の両半分で存在するセル)
    corr = {}
    for role, _c in ROLES:
        t1 = index_table(pre, role).set_index(["surface", "bucket", "going", "cat"])
        t2 = index_table(post, role).set_index(["surface", "bucket", "going", "cat"])
        j = t1.join(t2, lsuffix="1", rsuffix="2", how="inner")
        j = j[(j["count1"] >= 30) & (j["count2"] >= 30)]
        j["gp1"], j["gp2"] = j["idx1"] - j["m_pt1"], j["idx2"] - j["m_pt2"]
        jj = j.reset_index()
        for gname, sub in (("全馬場", jj), ("道悪", jj[jj["going"] != "良"])):
            corr[f"{role}/{gname}"] = {
                "n_cells": int(len(sub)),
                "r_total": round(float(np.corrcoef(sub["idx1"], sub["idx2"])[0, 1]), 3) if len(sub) > 2 else None,
                "r_going_part": round(float(np.corrcoef(sub["gp1"], sub["gp2"])[0, 1]), 3) if len(sub) > 2 else None,
            }
    out["half_corr"] = corr
    return out


def main():
    res = build_long()
    print(f"集計対象: {len(res)}走 / {res['race_id'].nunique()}レース")
    index = {}
    for role, _c in ROLES:
        cell = index_table(res[res[f"cat_{role}"].notna()], role)
        for _, r in cell.iterrows():
            index.setdefault(role, {}).setdefault(r["surface"], {}).setdefault(r["bucket"], {}) \
                .setdefault(r["going"], {})[r["cat"]] = {
                    "idx": round(float(r["idx"]), 2), "n": int(r["count"]), "raw": round(float(r["raw"]), 2),
                    "main": round(float(r["m_pt"]), 2)}
    pool_n = res.groupby(["surface", "bucket", "going"])["race_id"].nunique()
    pool = {f"{a}|{b}|{c}": int(v) for (a, b, c), v in pool_n.items()}
    val = validate(res)
    payload = {"k": K, "split_year": SPLIT_YEAR, "n_runs": int(len(res)), "pool_races": pool,
               "index": index, "validation": val}
    OUT_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(val["by_subset"], ensure_ascii=False, indent=1))
    print(json.dumps(val["half_corr"], ensure_ascii=False, indent=1))
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
