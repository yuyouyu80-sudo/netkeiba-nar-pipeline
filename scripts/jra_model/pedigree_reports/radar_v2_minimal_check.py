# -*- coding: utf-8 -*-
"""血統レーダー v2 の最小検証(R−1、2026-10-06)。

承認済みプラン(C:/Users/yuyou/.claude/plans/wiggly-sniffing-metcalfe.md「血統レーダー v2」)のR−1。
本実装(R0〜R5)の前に、3軸(脚質・キレ・距離)だけを新しい定義で作り、v1(現行)と比べて
  E4: 共通効果(全祖先が同じ向きに偏る現象)が消えるか
  E1: 産駒を2分割したときの祖先値の一致度(信頼性)
  E2簡易: 2020年までの祖先値で、2021年以降にデビューした馬の実際の走りを当てられるか
を測る。結果は data/jra_pipeline/pedigree_reports/radar_v2/r_minus1_result.json。

## v2の定義(この検証で使う範囲)
- 成績 perf = 1 − (着順−1)/(完走頭数−1)(1着=1、最下位=0)。競走中止等(着順が数字でない行)は除外。
- 脚質 front: 第1→第2→第3コーナーの順で最初に順位がある値を (1 − (順位−1)/(頭数−1)) にし、
  コース(競馬場×芝ダ×距離)×馬番位置(頭数で割った5区分)ごとの平均を引いた残差(枠・コースの差を除く)。
- キレ kire: 上がり3Fのレース内順位を同じ尺度にし、着順perfと4角位置front4(とその2次・交差項)で
  芝ダ別に回帰した残差(強さと後方位置で説明できる分を除く)。回帰係数は学習期間(〜2020)のみで推定。
- 距離 dist[b]: 馬内差 c = perf − 同じ馬の他の走の平均(学習期間内、2走以上の馬)。距離帯b×年齢帯×芝ダごとの
  c の平均を引いて中心化した値。
- 祖先値: 馬ごとに平均してから、産駒(馬)の間で平均する。縮約は全体0への経験ベイズ(k=馬間分散/祖先間分散)。

## v1の定義(比較対象、現行 jra_pedigree_commentary_profile と同じ式を同じ期間・同じ分割で再計算)
- 脚質v1 = 4角先頭時の勝率 − 全体勝率、キレv1 = 非先頭時の勝率 − 先頭時の勝率、距離v1[b] = 距離帯bの勝率 − 全体勝率。
  走単位で集計、セル下限N=10。
"""
import csv
import glob
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
import jra_pedigree_commentary_profile_2026_09_25 as PROF  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
PED_DIR = PROJECT_ROOT / "data" / "pedigree"
OUT_DIR = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports" / "radar_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)
PED_CACHE = OUT_DIR / "ped_ids.parquet"
RUNS_CACHE = OUT_DIR / "runs_minimal.parquet"
OUT_JSON = OUT_DIR / "r_minus1_result.json"

TRAIN_END_YEAR = 2020        # 祖先値・回帰係数はここまでのデータで作る
TEST_START_YEAR = 2021       # E2簡易の対象(この年以降にデビューした馬)
ROLES = [("sire", "ped_S_horse_id"), ("bms", "ped_DS_horse_id"), ("ss", "ped_SS_horse_id")]
V1_MIN_N = 10                # v1のセル下限(現行と同じ)
MIN_HORSES_HALF = 10         # E1: 各半分でこの頭数以上の産駒がいる祖先のみ
SEED = 20261006

USECOLS = ["race_id", "race_number", "race_name", "surface", "distance_m", "race_date", "racecourse",
           "finish_pos", "umaban", "horse_id", "sex_age", "last_3f",
           "corner1_rank", "corner2_rank", "corner3_rank", "corner4_rank", "corner4_is_leader"]


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ------------------------------------------------------------------ 読み込み
def load_runs() -> pd.DataFrame:
    frames = []
    for p in sorted(glob.glob(str(RESULTS_DIR / "20*" / "*.csv"))):
        df = pd.read_csv(p, dtype=str, encoding="utf-8", usecols=lambda c: c in USECOLS)
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    for c in USECOLS:
        if c not in df.columns:
            df[c] = np.nan
    n0 = len(df)
    # 事前登録追補1: (馬, 日付)の重複は race_id末尾2桁==race_number の行を残し、無ければ最小race_id
    df["rn2"] = pd.to_numeric(df["race_number"], errors="coerce").fillna(-1).astype(int).map(lambda x: f"{x:02d}")
    df["keep_pri"] = (df["race_id"].str[-2:] != df["rn2"]).astype(int)
    df = df.sort_values(["horse_id", "race_date", "keep_pri", "race_id"])
    dup = df.duplicated(["horse_id", "race_date"], keep="first")
    n_dup = int(dup.sum())
    df = df[~dup].drop(columns=["rn2", "keep_pri"])
    assert not df.duplicated(["horse_id", "race_date"]).any()
    log(f"読込 {n0}行、重複除去 {n_dup}行")

    df = df[df["surface"].isin(["芝", "ダ"])].copy()
    df["pos"] = pd.to_numeric(df["finish_pos"], errors="coerce")
    df = df[df["pos"].notna()].copy()
    df["field"] = df.groupby("race_id")["pos"].transform("count")
    df = df[df["field"] >= 5].copy()
    df["perf"] = 1 - (df["pos"] - 1) / (df["field"] - 1)
    df["is_win"] = (df["pos"] == 1).astype(float)
    df["date"] = pd.to_datetime(df["race_date"])
    df["year"] = df["date"].dt.year
    df["distance"] = pd.to_numeric(df["distance_m"], errors="coerce")
    df["dbucket"] = df["distance"].map(PROF.distance_bucket)
    df["age"] = pd.to_numeric(df["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")
    df["age_band"] = np.where(df["age"] <= 2, "2", np.where(df["age"] == 3, "3", "4+"))
    df["course_key"] = df["racecourse"] + "|" + df["surface"] + "|" + df["distance_m"].astype(str)
    um = pd.to_numeric(df["umaban"], errors="coerce")
    df["um_bin"] = np.minimum(((um - 1) / df["field"] * 5).fillna(0).astype(int), 4)

    # 前半位置(第1→第2→第3コーナーの順で最初にある順位)
    first_rank = np.full(len(df), np.nan)
    for c in ["corner3_rank", "corner2_rank", "corner1_rank"]:
        v = pd.to_numeric(df[c], errors="coerce").to_numpy()
        first_rank = np.where(~np.isnan(v), v, first_rank)
    df["front_raw"] = 1 - (first_rank - 1) / (df["field"] - 1)
    df["front_raw"] = df["front_raw"].clip(0, 1)
    r4 = pd.to_numeric(df["corner4_rank"], errors="coerce")
    df["front4"] = (1 - (r4 - 1) / (df["field"] - 1)).clip(0, 1)
    l3 = pd.to_numeric(df["last_3f"], errors="coerce")
    df["l3"] = l3
    df["l3_rank"] = df.groupby("race_id")["l3"].rank(method="average")
    n_l3 = df.groupby("race_id")["l3"].transform("count")
    df["kire_raw"] = 1 - (df["l3_rank"] - 1) / (n_l3 - 1)
    df["leader"] = df["corner4_is_leader"].map({"True": True, "False": False})
    keep = ["race_id", "horse_id", "date", "year", "surface", "distance", "dbucket", "age_band", "course_key",
            "um_bin", "field", "pos", "perf", "is_win", "front_raw", "front4", "kire_raw", "leader"]
    return df[keep].reset_index(drop=True)


def load_ped_ids(horse_ids) -> pd.DataFrame:
    cache = pd.read_parquet(PED_CACHE) if PED_CACHE.exists() else pd.DataFrame(columns=["horse_id"] + [c for _, c in ROLES])
    have = set(cache["horse_id"])
    todo = [h for h in horse_ids if h not in have]
    log(f"血統IDキャッシュ {len(have)}頭、新規読込 {len(todo)}頭")
    rows = []
    for h in todo:
        p = PED_DIR / f"{h}.csv"
        rec = {"horse_id": h}
        if p.exists():
            with open(p, encoding="utf-8", newline="") as f:
                r = csv.reader(f)
                hdr = next(r)
                row = next(r, None)
            if row:
                d = dict(zip(hdr, row))
                for _, c in ROLES:
                    rec[c] = d.get(c) or None
        rows.append(rec)
    if rows:
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_parquet(PED_CACHE, index=False)
    return cache


# ------------------------------------------------------------------ v2 の1走ごとの値
def add_v2_run_values(df: pd.DataFrame) -> pd.DataFrame:
    train = df["year"] <= TRAIN_END_YEAR
    # 脚質: コース×馬番位置の平均を引く(平均は学習期間のみで推定)
    m = df[train].groupby(["course_key", "um_bin"])["front_raw"].mean().rename("front_mu")
    df = df.join(m, on=["course_key", "um_bin"])
    df["front_mu"] = df["front_mu"].fillna(df.loc[train, "front_raw"].mean())
    df["front_v2"] = df["front_raw"] - df["front_mu"]
    # キレ: 芝ダ別に kire ~ perf + front4 + perf*front4 + perf^2 + front4^2 の残差(係数は学習期間のみ)
    df["kire_v2"] = np.nan
    for s in ["芝", "ダ"]:
        def X(d):
            p, f = d["perf"].to_numpy(), d["front4"].to_numpy()
            return np.column_stack([np.ones(len(d)), p, f, p * f, p * p, f * f])
        fit = df[train & (df["surface"] == s) & df["kire_raw"].notna() & df["front4"].notna()]
        beta, *_ = np.linalg.lstsq(X(fit), fit["kire_raw"].to_numpy(), rcond=None)
        idx = (df["surface"] == s) & df["kire_raw"].notna() & df["front4"].notna()
        df.loc[idx, "kire_v2"] = df.loc[idx, "kire_raw"].to_numpy() - X(df[idx]) @ beta
    return df


def within_horse_contrast(d: pd.DataFrame, center_means=None):
    """d(ある期間の行)に馬内差 c と中心化した c_ctr を付ける。center_means を渡せばそれで中心化。"""
    d = d.copy()
    g = d.groupby("horse_id")["perf"]
    d["n_h"] = g.transform("count")
    d["s_h"] = g.transform("sum")
    d = d[d["n_h"] >= 2].copy()
    d["c"] = d["perf"] - (d["s_h"] - d["perf"]) / (d["n_h"] - 1)
    keys = ["dbucket", "age_band", "surface"]
    if center_means is None:
        center_means = d.groupby(keys)["c"].mean().rename("c_mu")
    d = d.join(center_means, on=keys)
    d["c_ctr"] = d["c"] - d["c_mu"].fillna(0)
    return d, center_means


# ------------------------------------------------------------------ 祖先値
def ancestor_means(horse_vals: pd.DataFrame, value_col: str, role_col: str, min_horses=1) -> pd.DataFrame:
    """horse_vals: 馬1行(horse_id, 祖先ID列, value_col)。祖先ごとの 馬平均の平均・頭数・馬間分散。"""
    d = horse_vals.dropna(subset=[value_col, role_col])
    g = d.groupby(role_col)[value_col]
    out = pd.DataFrame({"mean": g.mean(), "n": g.count(), "var": g.var()})
    return out[out["n"] >= min_horses]


def eb_k(anc: pd.DataFrame) -> float:
    """k = 馬間分散(祖先内) / 祖先間分散(真の値の分散)。モーメント法。"""
    a = anc[anc["n"] >= 5]
    within = np.average(a["var"].fillna(0), weights=a["n"] - 1)
    total_var = np.var(a["mean"])
    tau2 = max(total_var - np.mean(within / a["n"]), 1e-12)
    return float(within / tau2)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return None, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def sb(r):
    return None if r is None else 2 * r / (1 + r)


# ------------------------------------------------------------------ v1(走単位の勝率差)
def v1_values(runs: pd.DataFrame, role_col: str) -> pd.DataFrame:
    d = runs.dropna(subset=[role_col])
    g = d.groupby(role_col)
    overall = g["is_win"].mean()
    out = pd.DataFrame({"overall": overall, "n": g.size()})
    for name, mask in [("lead", d["leader"] == True), ("nonlead", d["leader"] == False)]:  # noqa: E712
        gg = d[mask].groupby(role_col)["is_win"]
        out[f"{name}_wr"] = gg.mean()
        out[f"{name}_n"] = gg.size()
    out["front_v1"] = np.where(out["lead_n"] >= V1_MIN_N, (out["lead_wr"] - out["overall"]) * 100, np.nan)
    out["kire_v1"] = np.where((out["lead_n"] >= V1_MIN_N) & (out["nonlead_n"] >= V1_MIN_N),
                              (out["nonlead_wr"] - out["lead_wr"]) * 100, np.nan)
    for b, gb in d.groupby("dbucket"):
        gg = gb.groupby(role_col)["is_win"]
        wr, n = gg.mean(), gg.size()
        out[f"dist_v1|{b}"] = np.where(n.reindex(out.index).fillna(0) >= V1_MIN_N,
                                       (wr.reindex(out.index) - out["overall"]) * 100, np.nan)
    return out


# ------------------------------------------------------------------ メイン
def main():
    t0 = time.time()
    if RUNS_CACHE.exists():
        runs = pd.read_parquet(RUNS_CACHE)
        log(f"出走行キャッシュ {len(runs)}行")
    else:
        runs = load_runs()
        runs.to_parquet(RUNS_CACHE, index=False)
    ped = load_ped_ids(runs["horse_id"].unique().tolist())
    runs = runs.merge(ped, on="horse_id", how="left")
    runs = add_v2_run_values(runs)
    log(f"出走行 {len(runs)} / レース {runs['race_id'].nunique()} / 年 {runs['year'].min()}〜{runs['year'].max()}")

    train = runs[runs["year"] <= TRAIN_END_YEAR].copy()
    test = runs[runs["year"] >= TEST_START_YEAR].copy()
    first_year = runs.groupby("horse_id")["year"].min()
    new_horses = set(first_year[first_year >= TEST_START_YEAR].index)

    result = {"train_years": f"{runs['year'].min()}-{TRAIN_END_YEAR}", "test_from": TEST_START_YEAR,
              "n_runs": int(len(runs)), "n_train_runs": int(len(train)), "n_test_runs": int(len(test)),
              "n_new_horses_test": len(new_horses)}

    # ---- 欠損率(脚質の元になるコーナー順位、年別)
    result["front_missing_by_year"] = runs.groupby("year")["front_raw"].apply(lambda s: round(float(s.isna().mean()), 3)).to_dict()
    result["kire_missing_by_year"] = runs.groupby("year")["kire_v2"].apply(lambda s: round(float(s.isna().mean()), 3)).to_dict()

    # ---- 学習期間の馬単位の値
    hv = train.groupby("horse_id").agg(front_v2=("front_v2", "mean"), kire_v2=("kire_v2", "mean"))
    tr_c, center_means = within_horse_contrast(train)
    hd = tr_c.groupby(["horse_id", "dbucket"])["c_ctr"].mean().unstack()
    hd.columns = [f"dist_v2|{c}" for c in hd.columns]
    hv = hv.join(hd).join(ped.set_index("horse_id"))
    hv["half"] = [int(hashlib.md5(h.encode()).hexdigest(), 16) % 2 for h in hv.index]
    buckets = [b for _, _, b in PROF.DISTANCE_BUCKETS]
    v2_cols = ["front_v2", "kire_v2"] + [f"dist_v2|{b}" for b in buckets]

    # ---- E1: 産駒を2分割した信頼性(縮約前)、v1とv2を同じ分割で
    train = train.join(hv["half"], on="horse_id")
    e1 = {}
    for role, rc in ROLES:
        h0, h1 = hv[hv["half"] == 0].reset_index(), hv[hv["half"] == 1].reset_index()
        v1a, v1b = v1_values(train[train["half"] == 0], rc), v1_values(train[train["half"] == 1], rc)
        pairs = [("脚質", "front_v2", "front_v1"), ("キレ", "kire_v2", "kire_v1")] + \
                [(f"距離:{b}", f"dist_v2|{b}", f"dist_v1|{b}") for b in buckets]
        for label, c2, c1 in pairs:
            a0 = ancestor_means(h0, c2, rc, MIN_HORSES_HALF)
            a1 = ancestor_means(h1, c2, rc, MIN_HORSES_HALF)
            common = a0.index.intersection(a1.index)
            # 同じ祖先集合で比べる(v1の値がある祖先との共通部分)
            common_v1 = common.intersection(v1a[c1].dropna().index).intersection(v1b[c1].dropna().index)
            r2, n2 = corr(a0.loc[common, "mean"], a1.loc[common, "mean"])
            r2c, n2c = corr(a0.loc[common_v1, "mean"], a1.loc[common_v1, "mean"])
            r1c, n1c = corr(v1a.loc[common_v1, c1], v1b.loc[common_v1, c1])
            e1[f"{role}/{label}"] = {
                "v2_all": {"r": r2, "sb": sb(r2), "n_anc": n2},
                "same_set": {"n_anc": n1c, "v1_r": r1c, "v1_sb": sb(r1c), "v2_r": r2c, "v2_sb": sb(r2c)},
            }
    result["E1"] = e1

    # ---- 祖先値(学習期間全体、縮約)
    anc_tables = {}
    k_used = {}
    for role, rc in ROLES:
        hvr = hv.reset_index()
        for c in v2_cols:
            a = ancestor_means(hvr, c, rc, 1)
            k = eb_k(a)
            a["shrunk"] = a["mean"] * a["n"] / (a["n"] + k)
            anc_tables[(role, c)] = a
            k_used[f"{role}/{c}"] = round(k, 2)
    result["eb_k"] = k_used

    # ---- E4: 共通効果。祖先値の符号の偏り(v1 vs v2、sire、学習期間全体)
    e4 = {}
    v1_all = {role: v1_values(train, rc) for role, rc in ROLES}
    for label, c2, c1 in [("脚質", "front_v2", "front_v1"), ("キレ", "kire_v2", "kire_v1")] + \
            [(f"距離:{b}", f"dist_v2|{b}", f"dist_v1|{b}") for b in buckets]:
        a = anc_tables[("sire", c2)]
        a = a[a["n"] >= 30]
        v1s = v1_all["sire"][c1].dropna()
        e4[label] = {"v1_median": round(float(v1s.median()), 3), "v1_share_positive": round(float((v1s > 0).mean()), 3),
                     "v1_n": int(len(v1s)),
                     "v2_median": round(float(a["mean"].median()), 4), "v2_share_positive": round(float((a["mean"] > 0).mean()), 3),
                     "v2_sd_between": round(float(a["mean"].std()), 4), "v2_n": int(len(a))}
    result["E4_ancestor_sign"] = e4

    # ---- 馬の血統値(父・母父・父父の縮約値の平均、v2)と v1値(3ライン平均)
    def pedigree_value(horses: pd.DataFrame, c2: str, c1: str):
        v2s, v1s = [], []
        for role, rc in ROLES:
            a = anc_tables[(role, c2)]["shrunk"]
            v2s.append(horses[rc].map(a))
            v1s.append(horses[rc].map(v1_all[role][c1]))
        return pd.concat(v2s, axis=1).mean(axis=1), pd.concat(v1s, axis=1).mean(axis=1)

    # ---- E2簡易: 2021年以降にデビューした馬(学習期間に本人の走が無い)の実際の走り
    tnew = test[test["horse_id"].isin(new_horses)].copy()
    hv_test = tnew.groupby("horse_id").agg(front_v2=("front_v2", "mean"), kire_v2=("kire_v2", "mean"),
                                            n_runs=("race_id", "count"))
    hv_test = hv_test.join(ped.set_index("horse_id"))
    e2 = {}
    for label, c2, c1 in [("脚質", "front_v2", "front_v1"), ("キレ", "kire_v2", "kire_v1")]:
        p2, p1 = pedigree_value(hv_test, c2, c1)
        sub = hv_test[hv_test["n_runs"] >= 3]
        r2, n2 = corr(p2.loc[sub.index], sub[c2])
        r1, n1 = corr(p1.loc[sub.index], sub[c2])
        both = p2.loc[sub.index].notna() & p1.loc[sub.index].notna()
        r2b, nb = corr(p2.loc[sub.index][both], sub[c2][both])
        r1b, _ = corr(p1.loc[sub.index][both], sub[c2][both])
        e2[label] = {"target": f"2021年以降デビュー馬(3走以上)の実際の{label}の平均",
                     "v2_r": r2, "v2_n": n2, "v1_r": r1, "v1_n": n1,
                     "same_horses": {"n": nb, "v2_r": r2b, "v1_r": r1b}}
    # 距離: 新規馬の走ごとの馬内差(中心化は学習期間の平均)を、その距離帯の祖先値で当てる
    te_c, _ = within_horse_contrast(tnew, center_means)
    rows2, rows1, ys = [], [], []
    for b in buckets:
        sb_ = te_c[te_c["dbucket"] == b]
        p2, p1 = pedigree_value(sb_, f"dist_v2|{b}", f"dist_v1|{b}")
        rows2.append(p2), rows1.append(p1), ys.append(sb_["c_ctr"])
    P2, P1, Y = pd.concat(rows2), pd.concat(rows1), pd.concat(ys)
    both = P2.notna() & P1.notna()
    r2, n2 = corr(P2, Y)
    r1, n1 = corr(P1, Y)
    r2b, nb = corr(P2[both], Y[both])
    r1b, _ = corr(P1[both], Y[both])
    e2["距離"] = {"target": "2021年以降デビュー馬の各走の馬内差(その距離帯での普段との差、中心化)",
                "v2_r": r2, "v2_n": n2, "v1_r": r1, "v1_n": n1, "same_runs": {"n": nb, "v2_r": r2b, "v1_r": r1b}}
    # 相関の不確かさ(ブートストラップ、馬単位)
    rng = np.random.default_rng(SEED)
    for label, c2, c1 in [("脚質", "front_v2", "front_v1"), ("キレ", "kire_v2", "kire_v1")]:
        p2, p1 = pedigree_value(hv_test, c2, c1)
        sub = hv_test[hv_test["n_runs"] >= 3]
        ok = p2.loc[sub.index].notna() & p1.loc[sub.index].notna() & sub[c2].notna()
        a2, a1, y = p2.loc[sub.index][ok].to_numpy(), p1.loc[sub.index][ok].to_numpy(), sub[c2][ok].to_numpy()
        diffs = []
        for _ in range(1000):
            i = rng.integers(0, len(y), len(y))
            diffs.append(np.corrcoef(a2[i], y[i])[0, 1] - np.corrcoef(a1[i], y[i])[0, 1])
        e2[label]["diff_v2_minus_v1_ci95"] = [round(float(np.percentile(diffs, 2.5)), 4), round(float(np.percentile(diffs, 97.5)), 4)]
    result["E2_simple"] = e2

    # ---- E4b: レース基準。2021年以降のレースで「上位3着の平均−出走馬平均」(v2の血統値)と、
    #      v1の基準(上位3着の祖先値の平均)。共通効果の偏りとコース間の異質性。
    test_h = test.copy()  # 祖先ID列は runs に結合済み
    lift_out = {}
    for label, c2, c1 in [("脚質", "front_v2", "front_v1"), ("キレ", "kire_v2", "kire_v1")]:
        p2, p1 = pedigree_value(test_h, c2, c1)
        test_h["pv2"], test_h["pv1"] = p2, p1
        top = test_h["pos"] <= 3
        g_all = test_h.groupby("race_id")["pv2"].mean()
        g_top = test_h[top].groupby("race_id")["pv2"].mean()
        lift = (g_top - g_all).dropna()
        v1ref = test_h[top].groupby("race_id")["pv1"].mean().dropna()
        sd_between = float(test_h.groupby("race_id")["pv2"].std().mean())
        ck = test_h.groupby("race_id")["course_key"].first()
        L = pd.DataFrame({"lift": lift, "ck": ck.reindex(lift.index)})
        L["year"] = L.index.str[:4].astype(int)
        gm = L.groupby("ck")["lift"].agg(["mean", "count", "std"])
        gm = gm[gm["count"] >= 30]
        se = gm["std"] / np.sqrt(gm["count"])
        grand = float(L["lift"].mean())
        Q = float((((gm["mean"] - grand) / se) ** 2).sum())
        dfree = len(gm) - 1
        early = L[L["year"] <= 2023].groupby("ck")["lift"].agg(["mean", "count"])
        late = L[L["year"] >= 2024].groupby("ck")["lift"].agg(["mean", "count"])
        j = early.join(late, lsuffix="_e", rsuffix="_l", how="inner")
        j = j[(j["count_e"] >= 30) & (j["count_l"] >= 30)]
        r_el, n_el = corr(j["mean_e"], j["mean_l"])
        lift_out[label] = {
            "v1_reference_mean": round(float(v1ref.mean()), 3), "v1_reference_share_positive": round(float((v1ref > 0).mean()), 3),
            "v2_lift_mean_raw": round(grand, 5), "v2_lift_share_positive_raw": round(float((lift > 0).mean()), 3),
            "v2_lift_in_sd_units": round(grand / sd_between, 3) if sd_between else None,
            "v2_centered_share_positive": round(float(((lift - grand) > 0).mean()), 3),
            "course_heterogeneity": {"n_course_keys": int(len(gm)), "Q": round(Q, 1), "df": dfree,
                                     "I2": round(max(0.0, (Q - dfree) / Q), 3) if Q > 0 else None},
            "E3_course_lift_early_vs_late": {"r": r_el, "n_course_keys": n_el},
        }
    result["E4_race_reference"] = lift_out
    result["elapsed_sec"] = round(time.time() - t0, 1)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=lambda o: None), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=1, default=lambda o: None))
    log(f"wrote {OUT_JSON}")


if __name__ == "__main__":
    main()
