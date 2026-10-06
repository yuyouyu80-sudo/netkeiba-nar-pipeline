# -*- coding: utf-8 -*-
"""血統レーダー v2 の共通処理(2026-10-06新設、第2版プランのR−0.5以降で使う)。

プラン: C:/Users/yuyou/.claude/plans/wiggly-sniffing-metcalfe.md(第2版)。R−1(radar_v2_minimal_check.py)で
レビューにより判明した問題(障害戦の混入・降着の誤除外・馬単位平均の選択偏り・縮約目標0 等)を直した版。

## データの規則
- 障害戦: race_name が JUMP_RE(障害|ジャンプ|(JG|JS()に当たるレースを除外。障害戦は surface が「芝」で
  記録されているため surface では除外できない。除外後に芝2700m以上で残る距離が FLAT_LONG_OK だけであることをassert。
- 着順: 先頭の数字を着順とする(降着「3(降)」・再騎乗「12(再)」は残す)。中止・除外・取消・失格は成績から除外。
- 重複: jra_history._dedupe_race_id_drift → 事前登録追補1(race_id末尾2桁==race_numberを優先、無ければ最小race_id)
  → (horse_id, race_date)重複ゼロをassert。
- perf = 1 − (着順−1)/(完走頭数−1)。馬番の区分(um_bin)は最大馬番(≒出走頭数)で割る。

## 集計の規則(全軸共通)
- 馬の値は、その馬の走の平均。祖先の値は「馬の値」を重み w=min(n, W_CAP) で加重平均する(早く引退した弱い馬と
  50走する古馬のどちらにも偏らないように)。
- 中心化は馬単位で集計した後に行う(重み付きの全体平均を引く)。
- 縮約(経験ベイズ)の目標は親の値(全体=中心化後の0、系統を使う場合は系統の値)。k=祖先内分散/祖先間分散。
"""
import glob
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_model"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "jra_race_sim"))
import jra_history as JH  # noqa: E402
import jra_pedigree_commentary_profile_2026_09_25 as PROF  # noqa: E402
from race_potential import class_ordinal  # noqa: E402

RESULTS_DIR = PROJECT_ROOT / "data" / "race_results"
PED_DIR = PROJECT_ROOT / "data" / "pedigree"
OUT_DIR = PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports" / "radar_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

JUMP_RE = re.compile(r"障害|ジャンプ|\(JG|JS\(")
FLAT_LONG_OK = {3000, 3200, 3400, 3600}
ROLES = [("sire", "ped_S_horse_id"), ("bms", "ped_DS_horse_id"), ("ss", "ped_SS_horse_id")]
W_CAP = 5

USECOLS = ["race_id", "race_number", "race_name", "surface", "distance_m", "race_date", "racecourse",
           "finish_pos", "umaban", "horse_id", "sex_age", "last_3f", "time", "odds_final", "jockey_name",
           "corner1_rank", "corner2_rank", "corner3_rank", "corner4_rank", "corner4_is_leader"]


def is_jump(race_name: pd.Series) -> pd.Series:
    return race_name.fillna("").str.contains(JUMP_RE)


def parse_finish(finish_pos: pd.Series) -> pd.Series:
    """先頭の数字を着順に(降着・再騎乗を残す)。中止・除外・取消・失格はNaN。"""
    return pd.to_numeric(finish_pos.fillna("").str.extract(r"^(\d+)")[0], errors="coerce")


def dedupe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(df)
    df = JH._dedupe_race_id_drift(df)
    n1 = len(df)
    rn2 = pd.to_numeric(df["race_number"], errors="coerce").fillna(-1).astype(int).map(lambda x: f"{x:02d}")
    pri = (df["race_id"].str[-2:] != rn2).astype(int)
    df = df.assign(_pri=pri).sort_values(["horse_id", "race_date", "_pri", "race_id"])
    dup = df.duplicated(["horse_id", "race_date"], keep="first")
    df = df[~dup].drop(columns="_pri")
    assert not df.duplicated(["horse_id", "race_date"]).any()
    return df, {"drift_dropped": n0 - n1, "addendum1_dropped": int(dup.sum())}


def first_corner(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """第1→第2→第3コーナーの順で最初にある順位と、そのコーナー番号(無ければNaN/0)。"""
    rank = np.full(len(df), np.nan)
    idx = np.zeros(len(df), dtype=int)
    for k in (3, 2, 1):  # 3→2→1の順に上書きするので、最終的に1が最優先
        v = pd.to_numeric(df[f"corner{k}_rank"], errors="coerce").to_numpy()
        has = ~np.isnan(v)
        rank = np.where(has, v, rank)
        idx = np.where(has, k, idx)
    return rank, idx


def load_runs(max_year: int | None = None) -> tuple[pd.DataFrame, dict]:
    frames = []
    for p in sorted(glob.glob(str(RESULTS_DIR / "20*" / "*.csv"))):
        y = int(Path(p).parent.name)
        if max_year is not None and y > max_year:
            continue
        frames.append(pd.read_csv(p, dtype=str, encoding="utf-8", usecols=lambda c: c in USECOLS))
    df = pd.concat(frames, ignore_index=True)
    for c in USECOLS:
        if c not in df.columns:
            df[c] = np.nan
    meta = {"rows_read": int(len(df))}
    df, dmeta = dedupe(df)
    meta.update(dmeta)
    jump = is_jump(df["race_name"])
    meta["jump_rows_dropped"] = int(jump.sum())
    df = df[~jump & df["surface"].isin(["芝", "ダ"])].copy()
    df["distance"] = pd.to_numeric(df["distance_m"], errors="coerce")
    long_turf = set(df.loc[(df["surface"] == "芝") & (df["distance"] >= 2700), "distance"].dropna().astype(int))
    assert long_turf <= FLAT_LONG_OK, f"障害戦の取りこぼしの疑い: {sorted(long_turf - FLAT_LONG_OK)}"

    df["pos"] = parse_finish(df["finish_pos"])
    meta["non_finish_rows_dropped"] = int(df["pos"].isna().sum())
    um = pd.to_numeric(df["umaban"], errors="coerce")
    df["um_max"] = um.groupby(df["race_id"]).transform("max")
    df = df[df["pos"].notna()].copy()
    df["field"] = df.groupby("race_id")["pos"].transform("count")
    df = df[df["field"] >= 5].copy()
    df["perf"] = 1 - (df["pos"] - 1) / (df["field"] - 1)
    df["perf"] = df["perf"].clip(0, 1)  # 降着で着順>完走頭数になる稀なケース
    df["is_win"] = (df["pos"] == 1).astype(float)
    df["date"] = pd.to_datetime(df["race_date"])
    df["year"] = df["date"].dt.year
    df["logd"] = np.log(df["distance"])
    df["dbucket"] = df["distance"].map(PROF.distance_bucket)
    df["age"] = pd.to_numeric(df["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")
    df["age_band"] = np.where(df["age"] <= 2, "2", np.where(df["age"] == 3, "3", "4+"))
    df["class_ord"] = df["race_name"].map(class_ordinal)
    df["course_key"] = df["racecourse"] + "|" + df["surface"] + "|" + df["distance_m"].astype(str)
    df["um_bin"] = np.minimum(((pd.to_numeric(df["umaban"], errors="coerce") - 1) / df["um_max"] * 5).fillna(0).astype(int), 4)

    rank, cidx = first_corner(df)
    df["first_corner"] = cidx
    df["front_raw"] = (1 - (rank - 1) / (df["field"] - 1)).clip(0, 1)
    r4 = pd.to_numeric(df["corner4_rank"], errors="coerce")
    df["front4"] = (1 - (r4 - 1) / (df["field"] - 1)).clip(0, 1)
    df["l3"] = pd.to_numeric(df["last_3f"], errors="coerce")
    df["l3_rank"] = df.groupby("race_id")["l3"].rank(method="average")
    n_l3 = df.groupby("race_id")["l3"].transform("count")
    df["kire_raw"] = 1 - (df["l3_rank"] - 1) / (n_l3 - 1)
    df["leader"] = df["corner4_is_leader"].map({"True": True, "False": False})

    # 馬の履歴(時系列順): キャリア段階・前走との距離差・クラス変化
    df = df.sort_values(["horse_id", "date"])
    g = df.groupby("horse_id")
    df["career"] = g.cumcount()
    df["career_band"] = pd.cut(df["career"], [-1, 0, 3, 9, 1000], labels=["0", "1-3", "4-9", "10+"]).astype(str)
    df["prev_logd"] = g["logd"].shift(1)
    df["prev_class"] = g["class_ord"].shift(1)
    cc = np.sign(df["class_ord"] - df["prev_class"])
    df["class_change"] = np.where(df["prev_class"].isna() | df["class_ord"].isna(), "na",
                                  np.where(cc > 0, "up", np.where(cc < 0, "down", "same")))
    df["first_in_band"] = ~df.duplicated(["horse_id", "dbucket"])
    keep = ["race_id", "horse_id", "race_name", "date", "year", "surface", "distance", "logd", "dbucket", "age_band",
            "career", "career_band", "class_ord", "class_change", "prev_logd", "first_in_band", "course_key",
            "first_corner", "um_bin", "field", "pos", "perf", "is_win", "front_raw", "front4", "kire_raw", "leader"]
    return df[keep].reset_index(drop=True), meta


def load_ped_ids(horse_ids, cache_path: Path) -> pd.DataFrame:
    """父・母父・父父のID。血統ファイルが無かった馬(全列None)は毎回読み直す(キャッシュ無効化)。"""
    import csv
    cols = [c for _, c in ROLES]
    cache = pd.read_parquet(cache_path) if cache_path.exists() else pd.DataFrame(columns=["horse_id"] + cols)
    complete = set(cache.loc[cache[cols].notna().any(axis=1), "horse_id"])
    todo = [h for h in horse_ids if h not in complete]
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
                for c in cols:
                    rec[c] = d.get(c) or None
        rows.append(rec)
    if rows:
        cache = cache[~cache["horse_id"].isin(todo)]
        cache = pd.concat([cache, pd.DataFrame(rows)], ignore_index=True)
        cache.to_parquet(cache_path, index=False)
    return cache


# ------------------------------------------------------------------ 馬単位の集計と縮約
def horse_values(runs: pd.DataFrame, value_col: str, by=("horse_id",)) -> pd.DataFrame:
    """馬(×by)ごとの値の平均と走数。"""
    g = runs.dropna(subset=[value_col]).groupby(list(by))[value_col]
    return pd.DataFrame({"x": g.mean(), "n": g.count()})


def center_horse_values(hv: pd.DataFrame, strata: list | None = None) -> pd.DataFrame:
    """重み w=min(n, W_CAP) の加重平均を(層ごとに)引く。hv は列 x, n(+strata)。"""
    hv = hv.copy()
    hv["w"] = np.minimum(hv["n"], W_CAP)
    if strata:
        mu = hv.groupby(strata).apply(lambda d: np.average(d["x"], weights=d["w"])).rename("mu")
        hv = hv.join(mu, on=strata)
    else:
        hv["mu"] = np.average(hv["x"], weights=hv["w"])
    hv["xc"] = hv["x"] - hv["mu"]
    return hv


def ancestor_table(hv: pd.DataFrame, ped: pd.DataFrame, role_col: str, value: str = "xc") -> pd.DataFrame:
    """hv(index=horse_id, 列 value, w)と祖先ID列から、祖先ごとの加重平均・重み合計・頭数・加重分散。"""
    d = hv[[value, "w"]].join(ped.set_index("horse_id")[role_col], how="inner").dropna()
    d["wx"] = d["w"] * d[value]
    g = d.groupby(role_col)
    sw = g["w"].sum()
    mean = g["wx"].sum() / sw
    d = d.join(mean.rename("m"), on=role_col)
    d["wr2"] = d["w"] * (d[value] - d["m"]) ** 2
    var = d.groupby(role_col)["wr2"].sum() / sw
    return pd.DataFrame({"mean": mean, "sw": sw, "n_horses": g.size(), "var": var})


def eb_k(anc: pd.DataFrame, min_horses: int = 5) -> float:
    """k = 祖先内分散(馬単位、重み1あたり) / 祖先間分散(真の値)。重み合計 sw を「件数」とみなすモーメント法。"""
    a = anc[anc["n_horses"] >= min_horses]
    within = float(np.average(a["var"], weights=a["sw"]))
    tau2 = float(np.var(a["mean"]) - np.mean(within / a["sw"]))
    return within / max(tau2, 1e-12), within, max(tau2, 0.0)


def shrink(anc: pd.DataFrame, k: float, parent: pd.Series | float = 0.0) -> pd.Series:
    """(sw·mean + k·parent)/(sw + k)。k=0で生の値、k=∞で親の値。"""
    return (anc["sw"] * anc["mean"] + k * parent) / (anc["sw"] + k)


def sb(r):
    return None if r is None else 2 * r / (1 + r)


def corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 10:
        return None, int(ok.sum())
    return float(np.corrcoef(a[ok], b[ok])[0, 1]), int(ok.sum())


def cluster_boot_corr_diff(pred_a, pred_b, y, cluster, n_boot=1000, seed=0):
    """r(pred_a, y) − r(pred_b, y) の95%区間(cluster単位の再標本化)。"""
    d = pd.DataFrame({"a": pred_a, "b": pred_b, "y": y, "c": cluster}).dropna()
    groups = [g.to_numpy() for _, g in d.groupby("c")[["a", "b", "y"]]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        m = np.concatenate([groups[i] for i in pick])
        out.append(np.corrcoef(m[:, 0], m[:, 2])[0, 1] - np.corrcoef(m[:, 1], m[:, 2])[0, 1])
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]


def cluster_boot_corr(pred, y, cluster, n_boot=1000, seed=0):
    d = pd.DataFrame({"a": pred, "y": y, "c": cluster}).dropna()
    groups = [g.to_numpy() for _, g in d.groupby("c")[["a", "y"]]]
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(groups), len(groups))
        m = np.concatenate([groups[i] for i in pick])
        out.append(np.corrcoef(m[:, 0], m[:, 1])[0, 1])
    return [round(float(np.percentile(out, 2.5)), 4), round(float(np.percentile(out, 97.5)), 4)]
