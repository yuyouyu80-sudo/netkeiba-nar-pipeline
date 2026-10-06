# -*- coding: utf-8 -*-
"""血統レーダー v2 の R6: 任意の開催日・全競馬場のレースについて、発走前に分かる情報だけで v2/v3 の値を作る(2026-10-07)。

R5(build_race_radar_v2.py)は「結果が出たレース」の行(runs)から値を作っていた(9/26中山の固定)。R6 は出走表だけから作る。
- レース一覧: race_names_{date}.csv(predict_pattern29.py が発走前に作る、レース名・競馬場・芝ダ・距離・発走時刻)。
  無ければその日の race_results(結果が出た過去の日の表示用)。障害戦は除く。
- 出走馬: 馬柱 data/newspaper/{race_id}.csv(無ければ race_results)。
- クラス: レース名から(radar_v2_common.class_ordinal)。特別戦などで読めないときは、馬柱の data_others_slot*_label(今回のクラス条件、
  全角数字を NFKC で正規化)の最頻値。2026-09-13〜10-04 の平地161レースで結果側のクラスと全件一致を確認。
- 血統ID: data/pedigree/{horse_id}.csv。無い馬は、馬柱の父・母父の名前を全血統表から作った名前→ID表で引き、父父は父のIDから引く。
- 値の計算は R4/R5 と同じ関数(radar_v2_eval.features_for_year・add_own、radar_v2_reference の基準の式、v3 は v3_models_2026.pt)を、
  「その日より前の走」+「その日の出走表の行(結果の列は空)」に当てて作る。--check で R5(9/26中山)と一致を確かめる。
- 馬場状態は発走前には分からないので、DL(v3)は 良・稍重・重・不良 の4通りを計算する(v2 は馬場を使わない)。
- --record: 前向きの記録(radar_v2/prospective/{date}.json)。最初の記録が正本で上書きしない。発走時刻より後に作った分は記録に印を付け、評価から除く。

実行は .venv_dl の python(torch が要る):
  .venv_dl/Scripts/python.exe scripts/jra_model/pedigree_reports/radar_v2_live.py --date 20261010 [--record] [--check]
出力: radar_v2/live/race_radar_v2_{date}.json(表示用)、radar_v2/prospective/{date}.json(--record)
"""
import argparse
import csv
import datetime
import glob
import hashlib
import json
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_eval as E  # noqa: E402
import radar_v2_reference as RF  # noqa: E402

LIVE = C.OUT_DIR / "live"
PROS = C.OUT_DIR / "prospective"
LIVE.mkdir(exist_ok=True)
PROS.mkdir(exist_ok=True)
NEWS = C.PROJECT_ROOT / "data" / "newspaper"
# predict_pattern29.py の出力先(scripts/jra_model/plan_2026_10_04/weekend_run.py と同じ場所)
OLD_SCRATCH = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
                   r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad")
PROSPECTIVE_START = pd.Timestamp("2026-10-07")   # 事前登録 追補6: これ以降の開催日だけを前向きの記録にする
GOINGS = ["良", "稍重", "重", "不良"]
TIERS = ["速い", "平均", "遅い"]
COMPS = ["F_front", "F_kire", "F_surf", "F_db", "F_deb"]   # radar_v2_dl_condition_encoder.COMPS と同じ(compute で確認)
JST = datetime.timezone(datetime.timedelta(hours=9))
log = E.log


def sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


def norm_name(s) -> str:
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return ""
    s = unicodedata.normalize("NFKC", str(s))
    s = re.sub(r"\(.*?\)", "", s)
    return re.sub(r"\s+", "", s).lower()


# ------------------------------------------------------------------ 出走表
class NotReady(Exception):
    """開催なし・出走表や馬柱がまだ無い(自動起動では正常終了として扱う)。"""


def race_class_from_labels(news: pd.DataFrame) -> float:
    labs = [c for c in news.columns if c.startswith("data_others_slot") and c.endswith("_label")]
    vals = []
    for _, r in news.iterrows():
        for c in labs:
            v = unicodedata.normalize("NFKC", str(r[c]))
            co = C.class_ordinal(v)
            if v != "nan" and not np.isnan(co):
                vals.append(co)
                break
    return float(pd.Series(vals).mode().iloc[0]) if vals else np.nan


def load_card(D: pd.Timestamp, race_names: Path | None):
    ds = D.strftime("%Y%m%d")
    res_p = C.RESULTS_DIR / str(D.year) / f"{ds}.csv"
    res = pd.read_csv(res_p, dtype=str) if res_p.exists() else None
    rn = race_names or OLD_SCRATCH / f"race_names_{ds}.csv"
    if rn.exists():
        races = pd.read_csv(rn, dtype=str, encoding="utf-8-sig")
        src = f"race_names({rn.name})"
    elif res is not None:
        races = res.drop_duplicates("race_id")[["race_id", "race_name", "racecourse", "race_number", "surface", "distance_m", "start_time"]]
        src = "race_results"
    else:
        raise NotReady(f"{ds}: race_names も race_results もありません(開催なし、または出走表がまだ)")
    races = races.copy()
    races["surface"] = races["surface"].fillna("").map(lambda s: "ダ" if s.startswith("ダ") else ("芝" if s.startswith("芝") else s))
    races["distance"] = pd.to_numeric(races["distance_m"], errors="coerce")
    races["race_number"] = pd.to_numeric(races["race_number"], errors="coerce").astype(int)
    races["skip"] = np.where(C.is_jump(races["race_name"]) | ~races["surface"].isin(["芝", "ダ"]) | races["distance"].isna(),
                             "障害戦のため対象外", None)
    ents, cls = [], {}
    for rid in races["race_id"]:
        p = NEWS / f"{rid}.csv"
        if p.exists():
            n = pd.read_csv(p, dtype=str)
            cls[rid] = race_class_from_labels(n)
            e = n.rename(columns={"bias_sire": "sire", "bias_dam_sire": "bms", "bias_sex_age": "sex_age"})
            e = e[["race_id", "umaban", "horse_id", "horse_name", "sire", "bms", "sex_age"]].assign(ent_src="newspaper")
        elif res is not None and (res["race_id"] == rid).any():
            e = res[res["race_id"] == rid][["race_id", "umaban", "horse_id", "horse_name", "sex_age"]].assign(sire=None, bms=None,
                                                                                                      ent_src="race_results")
        else:
            continue
        ents.append(e)
    if not ents:
        raise NotReady(f"{ds}: 馬柱(data/newspaper)がまだありません")
    ent = pd.concat(ents, ignore_index=True)
    ent = ent.dropna(subset=["horse_id"]).drop_duplicates(["race_id", "horse_id"])
    co = races["race_name"].map(C.class_ordinal)
    races["class_ord"] = co.fillna(races["race_id"].map(cls))
    if res is not None:   # 結果のある過去の日: 表示用に着順・実際の馬場(前向きの記録には使わない)
        races["actual_going"] = races["race_id"].map(res.drop_duplicates("race_id").set_index("race_id")["going"])
        if src == "race_results":
            races["class_ord"] = races["race_name"].map(C.class_ordinal)
    return races, ent, src, res


# ------------------------------------------------------------------ 走の表(その日より前)
def results_hash_upto(D) -> tuple[str, pd.Timestamp]:
    files = sorted(glob.glob(str(C.RESULTS_DIR / "20*" / "*.csv")))
    h = hashlib.sha256()
    last = pd.Timestamp("1900-01-01")
    for p in files:
        d = pd.Timestamp(Path(p).stem)
        if d >= D:
            continue
        last = max(last, d)
        h.update(Path(p).name.encode())
        h.update(hashlib.sha256(Path(p).read_bytes()).digest())
    return h.hexdigest(), last


def runs_before(D: pd.Timestamp) -> tuple[pd.DataFrame, dict]:
    """runs_all.parquet が D の前日までの結果を含んでいればそれを使い、足りなければ同じ手順で作り直す(live/runs_live.parquet)。"""
    runs = pd.read_parquet(A.RUNS)
    rh, last = results_hash_upto(D)
    info = {"results_upto": str(last.date())}
    if runs["date"].max() < last:
        meta_p = LIVE / "runs_live_meta.json"
        cached = json.loads(meta_p.read_text(encoding="utf-8")) if meta_p.exists() else {}
        if cached.get("results_sha256") == rh and (LIVE / "runs_live.parquet").exists():
            runs = pd.read_parquet(LIVE / "runs_live.parquet")
        else:
            log("runs_live を作り直します(結果", last.date(), "まで)")
            runs, _ = C.load_runs(max_year=None)
            runs = runs[runs["date"] < D]
            runs = C.add_course_features(runs)
            runs = runs.merge(C.lap_pace(range(2011, D.year + 1)), on="race_id", how="left")
            ped = C.load_ped_ids(runs["horse_id"].unique().tolist(), C.OUT_DIR / "ped_ids_v2.parquet")
            runs = runs.merge(ped, on="horse_id", how="left")
            runs.to_parquet(LIVE / "runs_live.parquet", index=False)
            meta_p.write_text(json.dumps({"results_sha256": rh, "max_date": str(runs["date"].max().date())}), encoding="utf-8")
        info["runs_source"] = "runs_live"
    else:
        info["runs_source"] = "runs_all"
    runs = runs[runs["date"] < D].copy()
    info["runs_max_date"] = str(runs["date"].max().date())
    return runs, info


# ------------------------------------------------------------------ 血統ID(血統表が無い馬は名前から)
def name_map() -> dict:
    p = LIVE / "ped_name_map.json"
    files = glob.glob(str(C.PED_DIR / "*.csv"))
    if p.exists():
        m = json.loads(p.read_text(encoding="utf-8"))
        if m.get("n_files") == len(files):
            return m
    log("名前→IDの表を作ります(血統表", len(files), "件)")
    names, ss = {}, {}
    for f in files:
        with open(f, encoding="utf-8", newline="") as fh:
            r = csv.reader(fh)
            hdr = next(r, None)
            row = next(r, None)
        if not hdr or not row:
            continue
        d = dict(zip(hdr, row))
        for role in ("S", "DS"):
            i = d.get(f"ped_{role}_horse_id")
            if not i:
                continue
            for k in (f"ped_{role}_name_ja", f"ped_{role}_name_en"):
                n = norm_name(d.get(k))
                if n:
                    names.setdefault(n, {}).setdefault(i, 0)
                    names[n][i] += 1
        s, sss = d.get("ped_S_horse_id"), d.get("ped_SS_horse_id")
        if s and sss:
            ss.setdefault(s, {}).setdefault(sss, 0)
            ss[s][sss] += 1
    m = {"n_files": len(files), "name_to_id": {n: max(v, key=v.get) for n, v in names.items()},
         "sire_to_ss": {s: max(v, key=v.get) for s, v in ss.items()}}
    p.write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    return m


def ped_ids(ent: pd.DataFrame) -> pd.DataFrame:
    cols = [c for _, c in C.ROLES]
    cache = C.load_ped_ids(ent["horse_id"].unique().tolist(), C.OUT_DIR / "ped_ids_v2.parquet")
    df = ent[["horse_id", "sire", "bms"]].drop_duplicates("horse_id").merge(cache, on="horse_id", how="left")
    have = df[cols].notna().any(axis=1)
    df["ped_src"] = np.where(have, "血統表", None)
    if (~have).any():
        m = name_map()
        n2i, s2s = m["name_to_id"], m["sire_to_ss"]
        miss = ~have
        df.loc[miss, "ped_S_horse_id"] = df.loc[miss, "sire"].map(lambda x: n2i.get(norm_name(x)))
        df.loc[miss, "ped_DS_horse_id"] = df.loc[miss, "bms"].map(lambda x: n2i.get(norm_name(x)))
        df.loc[miss, "ped_SS_horse_id"] = df.loc[miss, "ped_S_horse_id"].map(lambda x: s2s.get(x) if x else None)
        got = miss & df["ped_S_horse_id"].notna()
        df.loc[got, "ped_src"] = "名前から"
        df.loc[miss & ~got, "ped_src"] = "不明"
    return df[["horse_id"] + cols + ["ped_src"]]


# ------------------------------------------------------------------ その日の出走表の行(結果の列は空)
def pseudo_rows(races: pd.DataFrame, ent: pd.DataFrame, runs: pd.DataFrame, D: pd.Timestamp, ped: pd.DataFrame) -> pd.DataFrame:
    r = races[races["skip"].isna()]
    df = ent.merge(r[["race_id", "race_name", "racecourse", "race_number", "surface", "distance", "class_ord"]], on="race_id")
    df["date"], df["year"] = D, D.year
    df["distance_m"] = df["distance"].astype(int).astype(str)
    df["logd"] = np.log(df["distance"])
    df["dbucket"] = df["distance"].map(C.PROF.distance_bucket)
    df["age"] = pd.to_numeric(df["sex_age"].fillna("").str.extract(r"(\d+)")[0], errors="coerce")
    df["age_band"] = np.where(df["age"] <= 2, "2", np.where(df["age"] == 3, "3", "4+"))
    df["is_debut_race"] = df["race_name"].fillna("").str.contains("新馬")
    df["course_key"] = df["racecourse"] + "|" + df["surface"] + "|" + df["distance_m"]
    um = pd.to_numeric(df["umaban"], errors="coerce")
    df["umaban_num"] = um
    df["um_max"] = um.groupby(df["race_id"]).transform("max")
    df["um_bin"] = np.minimum(((um - 1) / df["um_max"] * 5).fillna(0).astype(int), 4)
    df["field"] = df.groupby("race_id")["horse_id"].transform("count")
    for c in ["pos", "perf", "is_win", "front_raw", "front4", "kire_raw", "track_index_num", "pace"]:
        df[c] = np.nan
    df["leader"] = None
    df["first_corner"] = 0
    df["going"], df["going2"] = None, None
    prev = runs.sort_values("date").groupby("horse_id")
    last = prev.tail(1).set_index("horse_id")
    df["career"] = df["horse_id"].map(prev.size()).fillna(0).astype(int)
    df["career_band"] = pd.cut(df["career"], [-1, 0, 3, 9, 1000], labels=["0", "1-3", "4-9", "10+"]).astype(str)
    df["prev_logd"] = df["horse_id"].map(last["logd"])
    df["rest_days"] = (D - df["horse_id"].map(last["date"])).dt.days
    df["prev_class"] = df["horse_id"].map(last["class_ord"])
    cc = np.sign(df["class_ord"] - df["prev_class"])
    df["class_change"] = np.where(df["prev_class"].isna() | df["class_ord"].isna(), "na",
                                  np.where(cc > 0, "up", np.where(cc < 0, "down", "same")))
    seen = set(zip(runs["horse_id"], runs["dbucket"]))
    df["first_in_band"] = [(h, b) not in seen for h, b in zip(df["horse_id"], df["dbucket"])]
    df["rest_bucket"] = df["rest_days"].map(C.PROF.rest_bucket)
    df["class_high"] = np.where(df["class_ord"].isna(), None, np.where(df["class_ord"] >= 4, "OP以上", "条件戦"))
    df = C.add_course_features(df)
    df = df.merge(ped, on="horse_id", how="left")
    return df.drop(columns=["prev_class", "um_max"])


# ------------------------------------------------------------------ レースの基準(radar_v2_reference と同じ式、その日より前のレースだけ)
def race_lifts_before(runs: pd.DataFrame) -> pd.DataFrame:
    key = f"{runs['date'].max().date()}_{len(runs)}"
    p = LIVE / "race_lifts.parquet"
    kp = LIVE / "race_lifts_key.txt"
    if p.exists() and kp.exists() and kp.read_text() == key:
        return pd.read_parquet(p)
    th = A.fixed_thresholds(runs)
    rr = RF.class_keys(A.add_tiers(runs, th))
    lifts = []
    for Y in range(2011, int(runs["year"].max()) + 1):
        lifts.append(RF.race_lifts(A.run_values(rr[rr["year"] <= max(Y - 1, 2012)], rr[rr["year"] == Y])))
    R = pd.concat(lifts).sort_values("date")
    R.to_parquet(p)
    kp.write_text(key)
    return R


def req_new(R: pd.DataFrame, X: pd.DataFrame, val: str, k: float) -> pd.DataFrame:
    """RF.build_req と同じ式を、日付が X より前のレース R だけで(X は同じ日のレース)。"""
    P = R.dropna(subset=[val])
    allm = P[val].sum() / len(P)
    st = {}
    for L in ["L1", "L2", "L3", "L4"]:
        g = P.groupby(L)[val].agg(["sum", "count"])
        st[L] = (X[L].map(g["sum"]).fillna(0), X[L].map(g["count"]).fillna(0))
    m4 = (st["L4"][0] / st["L4"][1].replace(0, np.nan)).fillna(allm)
    sh3 = (st["L3"][0] + k * m4) / (st["L3"][1] + k)
    sh2 = (st["L2"][0] + k * sh3) / (st["L2"][1] + k)
    sh1 = (st["L1"][0] + k * sh2) / (st["L1"][1] + k)
    n1, n2, n3, n4 = st["L1"][1], st["L2"][1], st["L3"][1], st["L4"][1]
    req = np.where(n1 >= RF.MIN_L1, sh1, np.where(n2 >= RF.MIN_L2, sh2, np.where(n3 >= RF.MIN_L3, sh3, m4)))
    rule = np.where(n1 >= RF.MIN_L1, "L1", np.where(n2 >= RF.MIN_L2, "L2", np.where(n3 >= RF.MIN_L3, "L3", "L4")))
    nm = np.where(n1 >= RF.MIN_L1, n1, np.where(n2 >= RF.MIN_L2, n2, np.where(n3 >= RF.MIN_L3, n3, n4)))
    return pd.DataFrame({"req": req - allm, "rule": rule, "n_matched": nm}, index=X.index)


def race_req_new(runs: pd.DataFrame, pseudo: pd.DataFrame) -> pd.DataFrame:
    R = race_lifts_before(runs)
    X = RF.class_keys(pseudo.drop_duplicates("race_id")).set_index("race_id")
    k = json.loads((C.OUT_DIR / "race_req_meta.json").read_text(encoding="utf-8"))["k"]
    out = pd.DataFrame(index=X.index)
    for a in RF.AXES:
        val = f"lift_{a}"
        rq = req_new(R, X, val, k[a])
        out[f"req_{a}"], out[f"rule_{a}"], out[f"n_matched_{a}"] = rq["req"], rq["rule"], rq["n_matched"]
        for tier in TIERS:
            Rt = R.copy()
            Rt.loc[Rt["pace_tier"] != tier, val] = np.nan
            out[f"req_{a}_{tier}"] = req_new(Rt, X, val, k[a])["req"]
    return out


# ------------------------------------------------------------------ v3(DL)
def v3_scores(runs_ext: pd.DataFrame, f: pd.DataFrame, new_ids) -> tuple[dict, dict, float]:
    import torch
    import radar_v2_dl_condition_encoder as DL
    ck = torch.load(C.OUT_DIR / "v3_models_2026.pt", weights_only=False)
    ms = []
    for sd in ck["state_dicts"]:
        m = DL.Encoder(True)
        m.load_state_dict(sd)
        m.eval()
        ms.append(m)
    b1 = float(np.mean([m.b1.item() for m in ms]))
    rin = DL.race_inputs(runs_ext.assign(umaban_num=pd.to_numeric(runs_ext["umaban"], errors="coerce")))
    rin = rin.loc[rin.index.isin(new_ids)].drop(columns=["date", "year"])
    base = f[["race_id", "horse_id", "umaban_num", "S"] + DL.ZCOLS].copy()
    base["pos"] = base["umaban_num"]   # 並びのためだけ(結果は使わない)
    base["block"] = "-"
    req_out, g_out = {}, {}
    for going in GOINGS:
        df = base.join(rin.drop(columns=["going"]).assign(going=going), on="race_id")
        b = DL.Batch(df, ck["norm_stats"])
        with torch.no_grad():
            req = torch.stack([m.req(b.all()) for m in ms]).mean(0).numpy()
            u = DL.ens_u(ms, b.all()).numpy()
        G = u - b1 * b.S.numpy()
        dd = df.sort_values(["race_id", "pos", "umaban_num"])
        dd = dd[dd.groupby("race_id")["race_id"].transform("size") >= 3]
        for i, rid in enumerate(b.race_ids):
            req_out.setdefault(rid, {})[going] = {c: round(float(req[i, j]), 4) for j, c in enumerate(DL.ZCOLS)}
            for kk, h in enumerate(dd.loc[dd["race_id"] == rid, "horse_id"]):
                g_out.setdefault((rid, h), {})[going] = round(float(G[i, kk]), 4)
    return req_out, g_out, b1


# ------------------------------------------------------------------ 本体
def compute(D: pd.Timestamp, race_names: Path | None):
    t0 = time.time()
    races, ent, src, res = load_card(D, race_names)
    runs, info = runs_before(D)
    info["card_source"] = src
    log("出走表", src, len(races), "レース", len(ent), "頭 / 走の表", info)
    ped = ped_ids(ent)
    pseudo = pseudo_rows(races, ent, runs, D, ped)
    new_ids = pseudo["race_id"].unique().tolist()
    rq = race_req_new(runs, pseudo)
    req_all = pd.concat([pd.read_parquet(C.OUT_DIR / "race_req.parquet")[["req_front", "req_kire"]], rq[["req_front", "req_kire"]]])
    req_all = req_all[~req_all.index.duplicated(keep="last")]
    runs_ext = pd.concat([runs, pseudo.drop(columns=["ped_src", "sire", "bms", "horse_name", "sex_age", "ent_src", "umaban_num"])],
                         ignore_index=True)
    assert not runs_ext.duplicated(["horse_id", "date"]).any()
    Y = D.year
    kh = E.estimate_kh(runs)
    f = E.features_for_year(runs_ext, req_all, Y)
    f = E.add_own(runs_ext, f, Y, kh)
    f = f[f["race_id"].isin(new_ids)].copy()
    f["umaban_num"] = pd.to_numeric(f["umaban"], errors="coerce")
    import radar_v2_dl_condition_encoder as DL
    assert DL.COMPS == COMPS
    f = DL.components(f)
    v3_req, v3_g, b1 = v3_scores(runs_ext, f, new_ids)
    log("計算", round(time.time() - t0), "秒")
    return {"races": races, "ent": ent, "ped": ped, "f": f, "rq": rq, "v3_req": v3_req, "v3_g": v3_g, "b1": b1, "info": info, "res": res}


def r3(x):
    return None if x is None or (isinstance(x, float) and not np.isfinite(x)) or pd.isna(x) else round(float(x), 3)


def display_json(D: pd.Timestamp, out: dict) -> dict:
    races, ent, f, rq, res = out["races"], out["ent"], out["f"], out["rq"], out["res"]
    sd = {a: E.REQ_META[f"sd_req_{a}_2018_2020"] for a in ["front", "kire"]}
    fin = {}
    if res is not None:
        rr = res.assign(pos=C.parse_finish(res["finish_pos"]))
        fin = {(a, b): (None if np.isnan(c) else int(c), d) for a, b, c, d in zip(rr["race_id"], rr["horse_id"], rr["pos"], rr["finish_pos"])}
    fi = f.set_index(["race_id", "horse_id"])
    eni = ent.set_index(["race_id", "horse_id"])
    pedsrc = out["ped"].set_index("horse_id")["ped_src"]
    day = {"date": str(D.date()), "card_source": out["info"]["card_source"], "results_upto": out["info"]["results_upto"],
           "has_results": res is not None, "races": []}
    for _, r in races.sort_values(["racecourse", "race_number"]).iterrows():
        rid = r["race_id"]
        base = {"race_id": rid, "venue": r["racecourse"], "race_number": int(r["race_number"]), "race_name": r["race_name"],
                "start_time": r.get("start_time") if isinstance(r.get("start_time"), str) else None}
        if isinstance(r["skip"], str) or rid not in rq.index:
            day["races"].append(base | {"skipped": r["skip"] if isinstance(r["skip"], str) else "出走表なし"})
            continue
        q = {a: {"全体": r3(rq.at[rid, f"req_{a}"] / sd[a])} | {t: r3(rq.at[rid, f"req_{a}_{t}"] / sd[a]) for t in TIERS} for a in ["front", "kire"]}
        g = f[f["race_id"] == rid]
        surface, bucket = r["surface"], g["dbucket"].iloc[0] if len(g) else None
        race = base | {"surface": surface, "distance": int(r["distance"]), "dbucket": bucket, "is_debut": bool(g["is_debut_race"].iloc[0]),
                       "class_ord": r3(r["class_ord"]), "q": q,
                       "req_rule": {a: rq.at[rid, f"rule_{a}"] for a in ["front", "kire"]},
                       "req_n": {a: int(rq.at[rid, f"n_matched_{a}"]) for a in ["front", "kire"]},
                       "actual_going": r.get("actual_going") if isinstance(r.get("actual_going"), str) else None,
                       "v3_req": out["v3_req"].get(rid), "horses": []}
        for (_, hid), src in fi.loc[[rid]].iterrows() if rid in fi.index.get_level_values(0) else []:
            e = eni.loc[(rid, hid)]
            fp = fin.get((rid, hid))
            race["horses"].append({
                "umaban": int(src["umaban_num"]), "horse_id": hid, "horse_name": e["horse_name"], "sire": e.get("sire"), "bms": e.get("bms"),
                "ped_src": pedsrc.get(hid),
                "ped": {"S": r3(src["S"]), "front": r3(src["z_front"]), "kire": r3(src["z_kire"]), "surface": r3(src[f"z_s|{surface}"]),
                        "dbucket": r3(src[f"z_b|{bucket}"]) if bucket else None, "debut": r3(src["z_debut"])},
                "own": {"S": r3(src["S_own"]), "front": r3(src["zo_front"]), "kire": r3(src["zo_kire"]), "surface": r3(src["zo_surface"]),
                        "dbucket": r3(src["zo_dbucket"]), "debut": r3(src["z_debut"]),
                        "n_prev": int(src["perf_adj_pn"]) if np.isfinite(src["perf_adj_pn"]) else 0},
                "v3": out["v3_g"].get((rid, hid)),
                "finish": fp[0] if fp else None, "finish_raw": fp[1] if fp else None})
        race["horses"].sort(key=lambda x: x["umaban"])
        day["races"].append(race)
    return day


def record(D: pd.Timestamp, out: dict, race_names: Path | None):
    """前向きの記録。レースごとに最初の記録が正本(既にあるレースは変えず、まだ無いレースだけ足す)。"""
    p = PROS / f"{D.strftime('%Y%m%d')}.json"
    if D < PROSPECTIVE_START:
        sys.exit(f"{D.date()} は前向きの記録の対象外({PROSPECTIVE_START.date()} 以降のみ)。")
    if out["res"] is not None:
        sys.exit(f"{D.date()} は結果が既にあるため、前向きの記録は作りません。")
    now = datetime.datetime.now(JST)
    f, rq, races = out["f"], out["rq"], out["races"]
    ds = D.strftime("%Y%m%d")
    rn = race_names or OLD_SCRATCH / f"race_names_{ds}.csv"
    news = hashlib.sha256()
    for rid in sorted(f["race_id"].unique()):
        news.update(rid.encode())
        news.update((sha(NEWS / f"{rid}.csv") or "-").encode())
    run = {"generated_at": now.isoformat(timespec="seconds"),
           "inputs": out["info"] | {"race_names_sha256": sha(rn), "newspaper_sha256": news.hexdigest(),
                                    "v3_models_sha256": sha(C.OUT_DIR / "v3_models_2026.pt"), "table_year": D.year},
           "code": {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip(),
                    "radar_v2_live_sha256": sha(Path(__file__))}, "added": []}
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "date": str(D.date()), "v3_b1": out["b1"], "runs": [], "races": [],
        "note": "事前登録 追補6 の前向きの記録。レースごとに最初の記録が正本。recorded_before_start が真のレースだけを評価に使う。"}
    have = {r["race_id"] for r in rec["races"]}
    for _, r in races[races["skip"].isna()].iterrows():
        rid = r["race_id"]
        if rid not in rq.index or rid in have:
            continue
        st = r.get("start_time") if isinstance(r.get("start_time"), str) else None
        before = False
        if st:
            hh, mm = st.split(":")
            before = now < datetime.datetime(D.year, D.month, D.day, int(hh), int(mm), tzinfo=JST)
        g = f[f["race_id"] == rid]
        run["added"].append(rid)
        rec["races"].append({
            "race_id": rid, "venue": r["racecourse"], "race_number": int(r["race_number"]), "start_time": st,
            "recorded_at": run["generated_at"], "recorded_before_start": before,
            "entry_source": out["ent"].loc[out["ent"]["race_id"] == rid, "ent_src"].iloc[0],
            "ped_src": out["ped"].set_index("horse_id").loc[g["horse_id"], "ped_src"].value_counts().to_dict(),
            "surface": r["surface"], "distance": int(r["distance"]), "is_debut": bool(g["is_debut_race"].iloc[0]), "class_ord": r3(r["class_ord"]),
            "q_front": r3(g["q_front"].iloc[0]), "q_kire": r3(g["q_kire"].iloc[0]),
            "horses": [{"horse_id": h["horse_id"], "umaban": int(h["umaban_num"]), "S": r3(h["S"]), "F": r3(h["F"]), "S_own": r3(h["S_own"]),
                        "F_own": r3(h["F_own"]), "comps": [r3(h[c]) for c in COMPS],
                        "G3": out["v3_g"].get((rid, h["horse_id"]))} for _, h in g.sort_values("umaban_num").iterrows()]})
    rec["runs"].append(run)
    if run["added"]:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    n_ok = sum(r["recorded_before_start"] for r in rec["races"])
    log("前向きの記録", p.name, "今回追加", len(run["added"]), "/ 合計", len(rec["races"]), "レース(うち発走前", n_ok, ")")
    return run


def check_vs_r5(D: pd.Timestamp, day: dict):
    """R5(build_race_radar_v2.py、結果の行から作った 9/26 中山)と、出走表から作った R6 の値が一致するか。"""
    r5 = json.loads((C.PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports" / "race_radar_v2_data.json").read_text(encoding="utf-8"))
    mine = {r["race_id"]: r for r in day["races"] if not r.get("skipped")}
    diffs = {"ped": [], "own": [], "q": [], "v3": []}
    n = 0
    for r in r5["races"]:
        if r.get("skipped") or r["race_id"] not in mine:
            continue
        m = mine[r["race_id"]]
        for a in ["front", "kire"]:
            for t, v in r["q"][a].items():
                if v is not None and m["q"][a][t] is not None:
                    diffs["q"].append(abs(v - m["q"][a][t]))
        mh = {h["horse_id"]: h for h in m["horses"]}
        for h in r["horses"]:
            x = mh.get(h["horse_id"])
            if not x:
                continue
            n += 1
            for k in ["S", "front", "kire", "surface", "dbucket", "debut"]:
                if h["ped"][k] is not None and x["ped"][k] is not None:
                    diffs["ped"].append(abs(h["ped"][k] - x["ped"][k]))
                if h.get("own") and h["own"][k] is not None and x["own"][k] is not None:
                    diffs["own"].append(abs(h["own"][k] - x["own"][k]))
            if h.get("v3") is not None and x.get("v3") and m.get("actual_going") in x["v3"]:
                diffs["v3"].append(abs(h["v3"] - x["v3"][m["actual_going"]]))
    rep = {k: {"n": len(v), "max_abs_diff": round(max(v), 4) if v else None, "mean_abs_diff": round(float(np.mean(v)), 5) if v else None}
           for k, v in diffs.items()}
    rep["horses_matched"] = n
    log("R5との照合", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--race-names", type=Path)
    ap.add_argument("--record", action="store_true", help="前向きの記録を書く(最初の1回だけ)")
    ap.add_argument("--check", action="store_true", help="R5(9/26中山)との一致を確かめる")
    ap.add_argument("--date-today", action="store_true", help="--date の代わりに今日の日付(タスクスケジューラ用)")
    a = ap.parse_args() if "--date-today" not in sys.argv else ap.parse_args(sys.argv[1:] + ["--date", time.strftime("%Y%m%d")])
    D = pd.Timestamp(a.date)
    if a.date_today:   # タスクスケジューラから: 画面が無いのでログをファイルへ(追記)
        (C.PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(C.PROJECT_ROOT / "logs" / f"radar_v2_live_{a.date}.log", "a", encoding="utf-8", buffering=1)
        if D.weekday() not in (5, 6, 0):
            log(a.date, "は土日月ではないため何もしません")
            return 0
    if a.record and PROS.joinpath(f"{D.strftime('%Y%m%d')}.json").exists() and a.race_names is None:
        rec = json.loads(PROS.joinpath(f"{D.strftime('%Y%m%d')}.json").read_text(encoding="utf-8"))
        log("既存の記録", len(rec["races"]), "レース。まだ無いレースだけ足します")
    try:
        out = compute(D, a.race_names)
    except NotReady as e:
        log("何もしません:", e)
        return 0
    day = display_json(D, out)
    day["ped_src_counts"] = out["ped"]["ped_src"].value_counts().to_dict()
    if a.check:
        day["check_vs_r5"] = check_vs_r5(D, day)
    if a.record:
        day["prospective_record"] = record(D, out, a.race_names)["generated_at"]
    p = LIVE / f"race_radar_v2_{D.strftime('%Y%m%d')}.json"
    p.write_text(json.dumps(day, ensure_ascii=False), encoding="utf-8")
    log("wrote", p, round(p.stat().st_size / 1024), "KB", day["ped_src_counts"])


if __name__ == "__main__":
    main()
