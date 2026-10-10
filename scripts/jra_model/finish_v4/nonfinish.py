# -*- coding: utf-8 -*-
"""finish_v4 F0-1: 競走中止などの「発走したが着順の無い馬」を、買える候補として走の表に足す(2026-10-10、プラン第3版)。

- 発走した = 確定単勝オッズがある。着順の無い発走馬(競走中止・失格)は、3着以内に入らない候補として扱う(その馬を含む組は外れ)。
- 出走取消・競走除外(オッズ無し)は候補に入れない。発走前に分かれば買わず、記録後に分かった場合は精算で返還にする(F4)。
- 足す行(この docstring では「中止の行」)は runs_all.parquet と同じ列を持つ。
  - レースの列(日付・コース・クラスなど)は、同じレースの完走馬の行から写す。
  - 馬の列(馬番・年齢・血統ID)は race_results と、その馬の他の行(無ければ血統ファイル)から作る。
  - 結果の列(着順・perf・位置取り・上がり)は NaN。
  - 経歴の列(キャリア・前走の距離・間隔・クラスの変化・距離帯の初出走)は、その日より前の完走の行から作る(完走馬の定義と同じ)。
- 完走馬の行の値は変えない: 特徴量は「完走馬だけの表」で係数を推定し、中止の行を足した表には当てるだけにする。
  中止の行は結果が NaN なので、累積(本人の過去の走の和)には0として入り、他の馬・その馬の後の走の値を変えない。
  ただしカルマンフィルタは、行を足すとその馬の後の走で成長・分散の加算が1回増えるので、中止の行の値だけを足した表から取り、
  完走馬の値は完走馬だけの表から取る(f0_dev_utils)。
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
sys.path.insert(0, str(HERE.parent))
import radar_v2_common as C  # noqa: E402
import jra_pedigree_commentary_profile_2026_09_25 as PROF  # noqa: E402

RACE_COLS = ["race_name", "date", "year", "surface", "distance", "logd", "dbucket", "class_ord", "course_key", "racecourse",
             "distance_m", "race_number", "going", "going2", "track_index_num", "class_high", "is_debut_race", "field",
             "turn_dir", "circ_tier", "elev_tier", "straight_tier", "turf_type", "turn_type", "course_key_io", "pace"]
RESULT_COLS = ["pos", "perf", "is_win", "front_raw", "front4", "kire_raw", "leader", "first_corner"]
PED_COLS = [c for _, c in C.ROLES]


def started_nonfinishers(max_year: int, race_ids=None) -> pd.DataFrame:
    """race_results から、発走した(オッズあり)が着順の無い馬。列: race_id, horse_id, umaban, sex_age, finish_pos, um_max。"""
    frames = []
    for p in sorted(C.RESULTS_DIR.glob("20*/*.csv")):
        if int(p.parent.name) > max_year:
            continue
        frames.append(pd.read_csv(p, dtype=str, usecols=["race_id", "horse_id", "umaban", "sex_age", "finish_pos", "odds_final"]))
    d = pd.concat(frames, ignore_index=True).drop_duplicates(["race_id", "horse_id"])
    if race_ids is not None:
        d = d[d["race_id"].isin(set(race_ids))]
    um = pd.to_numeric(d["umaban"], errors="coerce")
    d["um_max"] = um.groupby(d["race_id"]).transform("max")
    odds = pd.to_numeric(d["odds_final"], errors="coerce")
    nf = d[(odds > 0) & C.parse_finish(d["finish_pos"]).isna()]
    return nf.drop(columns=["odds_final"]).reset_index(drop=True)


def _ped_ids(horse_ids, runs_all: pd.DataFrame) -> pd.DataFrame:
    """血統ID: runs_all の行 → 既存のキャッシュ(読むだけ)→ 血統ファイル。キャッシュには書かない(他の会話のタスクと競合しないため)。"""
    have = runs_all[["horse_id"] + PED_COLS].drop_duplicates("horse_id").set_index("horse_id")
    out = have.reindex(horse_ids)
    miss = out.index[out[PED_COLS].isna().all(axis=1)]
    cache_p = C.OUT_DIR / "ped_ids_v2.parquet"
    if len(miss) and cache_p.exists():
        cache = pd.read_parquet(cache_p).drop_duplicates("horse_id").set_index("horse_id")
        hit = cache.reindex(miss)[PED_COLS]
        out.loc[miss, PED_COLS] = hit.to_numpy()
    miss = out.index[out[PED_COLS].isna().all(axis=1)]
    for h in miss:
        p = C.PED_DIR / f"{h}.csv"
        if p.exists():
            with open(p, encoding="utf-8", newline="") as f:
                r = csv.reader(f)
                hdr = next(r)
                row = next(r, None)
            if row:
                rec = dict(zip(hdr, row))
                out.loc[h, PED_COLS] = [rec.get(c) or None for c in PED_COLS]
    return out.reset_index().rename(columns={"index": "horse_id"})


def ghost_rows(runs_all: pd.DataFrame, max_year: int) -> pd.DataFrame:
    """runs_all と同じ列の中止の行(runs_all にあるレースだけ)。"""
    runs_all = runs_all[runs_all["year"] <= max_year]
    nf = started_nonfinishers(max_year, runs_all["race_id"].unique())
    first = runs_all.drop_duplicates("race_id").set_index("race_id")[[c for c in RACE_COLS if c in runs_all.columns]]
    g = nf.join(first, on="race_id")
    g["age"] = pd.to_numeric(g["sex_age"].str.extract(r"(\d+)")[0], errors="coerce")
    g["age_band"] = np.where(g["age"] <= 2, "2", np.where(g["age"] == 3, "3", "4+"))
    g["um_bin"] = np.minimum(((pd.to_numeric(g["umaban"], errors="coerce") - 1) / g["um_max"] * 5).fillna(0).astype(int), 4)
    for c in RESULT_COLS:
        g[c] = np.nan
    g = g.merge(_ped_ids(g["horse_id"].unique(), runs_all), on="horse_id", how="left")
    # 経歴: その日より前の完走の行
    hist = runs_all[["horse_id", "date", "logd", "class_ord", "dbucket"]].sort_values(["horse_id", "date"])
    hist_d = hist.sort_values("date")
    g = g.sort_values(["date"]).reset_index(drop=True)
    prev = pd.merge_asof(g[["horse_id", "date"]].reset_index(), hist_d.rename(columns={"date": "prev_date"}),
                         left_on="date", right_on="prev_date", by="horse_id", allow_exact_matches=False).set_index("index")
    cnt = hist.assign(one=1)
    cnt["career_after"] = cnt.groupby("horse_id")["one"].cumsum()
    cnt = cnt.sort_values("date")
    car = pd.merge_asof(g[["horse_id", "date"]].reset_index(), cnt[["horse_id", "date", "career_after"]],
                        on="date", by="horse_id", allow_exact_matches=False).set_index("index")
    g["career"] = car["career_after"].fillna(0).astype(int)
    g["career_band"] = pd.cut(g["career"], [-1, 0, 3, 9, 1000], labels=["0", "1-3", "4-9", "10+"]).astype(str)
    g["prev_logd"] = prev["logd"]
    g["rest_days"] = (g["date"] - prev["prev_date"]).dt.days
    cc = np.sign(g["class_ord"] - prev["class_ord"])
    g["class_change"] = np.where(prev["class_ord"].isna() | g["class_ord"].isna(), "na",
                                 np.where(cc > 0, "up", np.where(cc < 0, "down", "same")))
    seen = runs_all[["horse_id", "date", "dbucket"]].merge(g[["horse_id", "date", "dbucket"]].reset_index(), on=["horse_id", "dbucket"],
                                                           suffixes=("_h", ""))
    seen_before = seen[seen["date_h"] < seen["date"]]["index"].unique()
    g["first_in_band"] = ~g.index.isin(seen_before)
    g["rest_bucket"] = g["rest_days"].map(PROF.rest_bucket)
    g["nonfinish"] = True
    cols = list(runs_all.columns)
    for c in cols:
        if c not in g.columns:
            g[c] = np.nan
    return g[cols + ["nonfinish"]]


def augment(runs_all: pd.DataFrame, max_year: int) -> tuple[pd.DataFrame, dict]:
    """runs_all(max_year まで)+ 中止の行。nonfinish 列で区別できる。"""
    base = runs_all[runs_all["year"] <= max_year].assign(nonfinish=False)
    gh = ghost_rows(runs_all, max_year)
    aug = pd.concat([base, gh], ignore_index=True).sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    meta = {"ghost_rows": int(len(gh)), "ghost_races": int(gh["race_id"].nunique()),
            "ghost_no_ped": int(gh[PED_COLS].isna().all(axis=1).sum()), "ghost_debut": int((gh["career"] == 0).sum())}
    return aug, meta
