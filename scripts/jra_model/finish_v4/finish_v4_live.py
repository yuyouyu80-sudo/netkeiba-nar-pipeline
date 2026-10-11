# -*- coding: utf-8 -*-
"""finish_v4 P4: 発走前の出走表から完成形(FINAL)の買い目を作り、前向きに記録する(2026-10-11、事前登録 §8)。

- 出走表・その日より前の走・血統ID・その日の出走表の行(結果の列は空)は radar_v2_live と同じ関数で作る
  (load_card・runs_before・ped_ids・pseudo_rows・race_req_new)。
- 血統+実績(S_own・F_own)と血統の値(z)は radar_v2_eval.features_for_year・add_own(kh は確認と同じ 2020年末までの固定値)。
- 能力値(芝ダ別の状態・父の月次・母のきょうだい・主戦距離・前走・斤量・休み)は、その日の出走表の行を「中止の行」と同じ扱い
  (nonfinish=True: 発走前の値を読むだけで状態を進めない)で走の表に足し、確認と同じ関数(ability_v2.features など)で作る。
- 係数は live_params_{年}.json(その年の評価と同じく、2014年〜前年のレースで当てはめた完成形)。オッズ・騎手・馬体重は使わない。
- 買い目: 主ルール20点(sanrenpuku_rules)。印は買い目から作る。
- --record: data/jra_pipeline/finish_v4/prospective/{date}.json。レースごとに最初の記録が正本(上書きしない)。
  発走時刻より後の記録には recorded_before_start=false を付け、評価に使わない。
- --fit-params YEAR: live_params_{YEAR}.json を作る(y2026_tables.parquet など、その年の表の学習行 2014〜YEAR−1 で当てはめ)。
- --check DATE: 結果のある過去の日について、出走表から作った効用と、確認と同じ処理(y2026_utils.parquet)の効用を比べる。

実行: .venv_dl の python
  python scripts/jra_model/finish_v4/finish_v4_live.py --date 20261011 [--record]
"""
import argparse
import datetime
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
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
import ability_v2 as V  # noqa: E402
import f0_dev_utils as F0  # noqa: E402
import model as MD  # noqa: E402
import nonfinish as NF  # noqa: E402
import sanrenpuku_rules as SR  # noqa: E402
import radar_v2_live as RL  # noqa: E402

AB, A, C = F0.AB, F0.A, F0.C
OUT = F0.OUT_DIR
LIVE = OUT / "live"
PROS = OUT / "prospective"
LIVE.mkdir(exist_ok=True)
PROS.mkdir(exist_ok=True)
FEATS = MD.SPECS["FULL"]
JST = datetime.timezone(datetime.timedelta(hours=9))
PROSPECTIVE_START = pd.Timestamp("2026-10-12")      # 確認(2026-10-11)の後の開催日から
log = F0.log


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


# ------------------------------------------------------------------ 係数
def fit_params(year: int):
    cache = OUT / f"y{year}_tables.parquet"
    T = pd.read_parquet(cache, filters=[("eval_year", "=", year)]).drop(columns="eval_year")
    tr = MD.derived(T[T["year"].between(F0.TRAIN_START, year - 1)])
    beta, lam = MD.fit_std(tr, FEATS, 0.0)
    out = {"year": year, "train": [F0.TRAIN_START, year - 1], "features": FEATS, "beta": [float(b) for b in beta], "lam": list(lam),
           "made": time.strftime("%Y-%m-%d %H:%M:%S"), "source": cache.name}
    (OUT / f"live_params_{year}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log("live params", year, "lam", np.round(lam, 3))


# ------------------------------------------------------------------ 特徴量
def kinryo_of(ent: pd.DataFrame, res) -> pd.Series:
    vals = {}
    for rid in ent["race_id"].unique():
        p = RL.NEWS / f"{rid}.csv"
        if p.exists():
            n = pd.read_csv(p, dtype=str)
            col = "bias_weight_carried" if "bias_weight_carried" in n else "speed_weight_carried"
            k = pd.to_numeric(n[col], errors="coerce")
            if k.isna().all() and "speed_weight_carried" in n:
                k = pd.to_numeric(n["speed_weight_carried"], errors="coerce")
            vals.update({(rid, h): v for h, v in zip(n["horse_id"], k)})
        elif res is not None:
            r = res[res["race_id"] == rid]
            vals.update({(rid, h): v for h, v in zip(r["horse_id"], pd.to_numeric(r["kinryo"], errors="coerce"))})
    return pd.Series(vals)


def build(D: pd.Timestamp, race_names: Path | None):
    import radar_v2_eval as E
    import radar_v2_stamina as S
    t0 = time.time()
    Y = D.year
    races, ent, src, res = RL.load_card(D, race_names)
    runs, info = RL.runs_before(D)
    info["card_source"] = src
    ped = RL.ped_ids(ent)
    pseudo = RL.pseudo_rows(races, ent, runs, D, ped)
    new_ids = pseudo["race_id"].unique().tolist()
    # 血統+実績・血統の値
    rq = RL.race_req_new(runs, pseudo)
    req_all = pd.concat([pd.read_parquet(C.OUT_DIR / "race_req.parquet")[["req_front", "req_kire"]], rq[["req_front", "req_kire"]]])
    req_all = req_all[~req_all.index.duplicated(keep="last")]
    keep_cols = list(runs.columns)
    pz = pseudo[[c for c in keep_cols if c in pseudo.columns]].copy()
    for c in keep_cols:
        if c not in pz.columns:
            pz[c] = np.nan
    runs_ext = pd.concat([runs, pz[keep_cols]], ignore_index=True)
    th = A.fixed_thresholds(pd.read_parquet(A.RUNS))
    rx = A.add_tiers(runs_ext, th)
    rx["umaban_num"] = pd.to_numeric(rx["umaban"], errors="coerce")
    f = E.features_for_year(rx, req_all, Y)
    f = E.add_own(rx, f, Y, F0.kh_fixed())
    f = f[f["race_id"].isin(new_ids)].set_index(["race_id", "horse_id"])[["S_own", "F_own"] + F0.PED_Z]
    # 能力値: 出走表の行を「読むだけの行」として足す
    aug, _ = NF.augment(runs, Y)
    aug = aug[aug["date"] < D]
    pz2 = pz[keep_cols].assign(nonfinish=True)
    aug = pd.concat([aug, pz2], ignore_index=True).sort_values(["horse_id", "date", "race_id"]).reset_index(drop=True)
    ra = F0.load_aug_runs(aug)
    kin = kinryo_of(ent, res)
    isnew = ra["race_id"].isin(new_ids)
    ra.loc[isnew, "kinryo"] = [kin.get((r, h), np.nan) for r, h in zip(ra.loc[isnew, "race_id"], ra.loc[isnew, "horse_id"])]
    ra.loc[isnew, "time_sec"] = np.nan
    ra.loc[isnew, "tpk"] = np.nan
    ra["dam_id"] = ra["horse_id"].map(V.dam_ids(ra["horse_id"].unique()))
    ra["is_turf"] = (ra["surface"] == "芝").astype(int)
    ra["month0"] = ra["date"].dt.to_period("M").dt.start_time
    fin = ra[~ra["nonfinish"]]
    Pf = AB.fit_fig(fin, Y - 1)
    fig = AB.apply_fig(ra, Pf)
    g = ra.groupby("horse_id")
    ab = pd.DataFrame({"last_fig": fig.groupby(ra["horse_id"]).shift(1), "kin": ra["kinryo"] - 55.0,
                       "d_kin": (ra["kinryo"] - g["kinryo"].shift(1)).fillna(0), "rest_long": (ra["rest_days"] > 90).astype(float)},
                      index=ra.index)
    R = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")
    flags = S.add_run_cols(ra[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan), R)
    Fv = V.features(ra, Y, MD.v2_par(), AB.sire_S(ra))
    X = pd.concat([ra[["race_id", "horse_id", "umaban_num", "surface", "logd", "class_ord", "is_debut_race"]], ab,
                   flags[["ext", "short", "ext_amt"]], Fv[MD.V2_COLS]], axis=1)[isnew.to_numpy()]
    for c in ["ext", "short"]:
        X[c] = X[c].fillna(False).astype(float)
    X["ext_amt"] = X["ext_amt"].fillna(0.0)
    X["is_dirt"] = (X["surface"] == "ダ").astype(float)
    X["class_n"] = X["class_ord"] / 7.0
    X["debut"] = X["is_debut_race"].astype(float)
    X["n_start"] = X.groupby("race_id")["umaban_num"].transform("max")
    X = X.join(f, on=["race_id", "horse_id"])
    X = MD.derived(X)
    info["elapsed_sec"] = round(time.time() - t0, 1)
    return races, ent, X, info, res


def predict(X: pd.DataFrame, params: dict) -> pd.DataFrame:
    assert params["features"] == FEATS
    X = X.copy()
    X["u"] = X[FEATS].fillna(0).to_numpy(float) @ np.asarray(params["beta"])
    return X


# ------------------------------------------------------------------ 記録・表示
def race_payload(g: pd.DataFrame, lam, names: dict) -> dict:
    g = g.sort_values("umaban_num")
    t = SR.race_tickets(g["umaban_num"].astype(int).tolist(), g["u"].to_numpy(float), lam, SR.MAIN_RULE)
    return {"umaban": g["umaban_num"].astype(int).tolist(), "horse_id": g["horse_id"].tolist(),
            "horse_name": [names.get(h) for h in g["horse_id"]], "u": [round(float(x), 4) for x in g["u"]]} | t


def run(D: pd.Timestamp, race_names: Path | None, do_record: bool):
    params = json.loads((OUT / f"live_params_{D.year}.json").read_text(encoding="utf-8"))
    races, ent, X, info, res = build(D, race_names)
    X = predict(X, params)
    names = dict(zip(ent["horse_id"], ent["horse_name"]))
    rinfo = races.set_index("race_id")
    day = {"date": str(D.date()), "rule": SR.MAIN_RULE, "model": "finish_v4 FINAL", "params_year": params["year"], "info": info, "races": {}}
    for rid, g in X.groupby("race_id"):
        r = rinfo.loc[rid]
        day["races"][rid] = {"race_name": r["race_name"], "venue": r["racecourse"], "race_number": int(r["race_number"]),
                             "start_time": r.get("start_time") if isinstance(r.get("start_time"), str) else None} | race_payload(g, params["lam"], names)
    (LIVE / f"finish_v4_{D.strftime('%Y%m%d')}.json").write_text(json.dumps(day, ensure_ascii=False, indent=1), encoding="utf-8")
    log("live", D.date(), len(day["races"]), "レース", info["elapsed_sec"], "秒")
    if do_record:
        record(D, day, res, race_names)
    return day


def record(D: pd.Timestamp, day: dict, res, race_names: Path | None):
    if D < PROSPECTIVE_START:
        sys.exit(f"{D.date()} は前向きの記録の対象外({PROSPECTIVE_START.date()} 以降のみ)。")
    if res is not None:
        sys.exit(f"{D.date()} は結果が既にあるため、前向きの記録は作りません。")
    p = PROS / f"{D.strftime('%Y%m%d')}.json"
    now = datetime.datetime.now(JST)
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "date": str(D.date()), "rule": SR.MAIN_RULE, "runs": [], "races": {},
        "note": "事前登録 PREREG_FINISH_V4.md §8 の前向きの記録。レースごとに最初の記録が正本。recorded_before_start が真のレースだけを評価に使う。"}
    run_meta = {"generated_at": now.isoformat(timespec="seconds"), "inputs": day["info"],
                "params_sha256": sha(OUT / f"live_params_{D.year}.json"),
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip(),
                "live_sha256": sha(Path(__file__)), "added": []}
    for rid, r in day["races"].items():
        if rid in rec["races"]:
            continue
        before = False
        if r["start_time"]:
            hh, mm = r["start_time"].split(":")
            before = now < datetime.datetime(D.year, D.month, D.day, int(hh), int(mm), tzinfo=JST)
        rec["races"][rid] = r | {"recorded_at": run_meta["generated_at"], "recorded_before_start": before}
        run_meta["added"].append(rid)
    rec["runs"].append(run_meta)
    if run_meta["added"]:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    log("前向きの記録", p.name, "追加", len(run_meta["added"]), "/ 合計", len(rec["races"]),
        "(うち発走前", sum(v["recorded_before_start"] for v in rec["races"].values()), ")")


def check(D: pd.Timestamp):
    """結果のある過去の日: 出走表から作った効用と、確認と同じ処理の効用(y{年}_utils.parquet)の一致を見る。"""
    params = json.loads((OUT / f"live_params_{D.year}.json").read_text(encoding="utf-8"))
    _, _, X, info, _ = build(D, None)
    X = predict(X, params)
    ref = pd.read_parquet(OUT / f"y{D.year}_utils.parquet")
    j = X.merge(ref, on=["race_id", "horse_id"], how="inner")
    out = {"date": str(D.date()), "n_live": int(len(X)), "n_matched": int(len(j)), "corr_u": float(np.corrcoef(j["u"], j["u_FINAL"])[0, 1]),
           "max_abs_diff": float((j["u"] - j["u_FINAL"]).abs().max()), "median_abs_diff": float((j["u"] - j["u_FINAL"]).abs().median())}
    # 確認側の表で血統の値が欠けた馬(血統表が未取得の2歳など。確認側は0で埋まる)。発走前は馬柱の父の名前から血統IDを引くので値が付く
    T = pd.read_parquet(OUT / f"y{D.year}_tables.parquet", filters=[("eval_year", "=", D.year)], columns=["race_id", "horse_id", "z_front", "kin"])
    j = j.merge(T.rename(columns={"z_front": "z_front_ref", "kin": "kin_ref"}), on=["race_id", "horse_id"], how="left")
    gap = set(j.loc[j["z_front_ref"].isna(), "race_id"])
    same = []
    for rid, g in j.groupby("race_id"):
        a = SR.race_tickets(g["umaban_num_x"].astype(int).tolist(), g["u"].to_numpy(float), params["lam"])["tickets"]
        b = SR.race_tickets(g["umaban_num_x"].astype(int).tolist(), g["u_FINAL"].to_numpy(float), params["lam"])["tickets"]
        same.append((rid in gap, len({tuple(x) for x in a} & {tuple(x) for x in b}) / max(len(a), 1), float((g["u"] - g["u_FINAL"]).abs().max())))
    S_ = pd.DataFrame(same, columns=["gap", "overlap", "maxdiff"])
    out["ticket_overlap_mean"] = float(S_["overlap"].mean())
    out["n_horses_ped_missing_in_ref"] = int(j["z_front_ref"].isna().sum())
    out["by_ped_gap"] = {("ref_ped_missing" if k else "ref_ped_complete"): {"n_races": int(len(v)), "ticket_overlap_mean": float(v["overlap"].mean()),
                                                                          "max_abs_diff": float(v["maxdiff"].max())} for k, v in S_.groupby("gap")}
    out["n_kin_diff"] = int(((j["kin"] - j["kin_ref"]).abs() > 1e-9).sum())     # 斤量: 発走前は馬柱、確認側は結果の値
    (LIVE / f"check_{D.strftime('%Y%m%d')}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    log("check", out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date")
    ap.add_argument("--date-today", action="store_true", help="今日(日本時間)。タスクスケジューラ用。開催・出走表が無ければ何もせず終わる")
    ap.add_argument("--race-names-csv")
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--fit-params", type=int)
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.fit_params:
        fit_params(a.fit_params)
    if a.check:
        check(pd.Timestamp(a.check))
    if a.date_today:
        a.date = datetime.datetime.now(JST).strftime("%Y%m%d")
        logs = C.PROJECT_ROOT / "logs"
        logs.mkdir(exist_ok=True)
        fh = open(logs / f"finish_v4_live_{a.date}.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = fh      # pythonw.exe(画面なし)でも記録が残るように
    if a.date:
        try:
            run(pd.Timestamp(a.date), Path(a.race_names_csv) if a.race_names_csv else None, a.record)
        except RL.NotReady as e:
            log("まだ作れません:", e)


if __name__ == "__main__":
    main()
