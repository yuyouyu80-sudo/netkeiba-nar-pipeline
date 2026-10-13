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
- G0(2026-10-13、プラン第4版): M1・FINAL・F7(+月齢・生産者)を同じ実行で並べて記録する(影の記録。FINAL が主、表示・NET登録はしない)。
  レースごとに欠けと遅れ(走の表に無い前走・血統IDの出所・生年月日/生産者の欠け)を数え、入力の控え(snapshot、sha 付き)と
  実行の状態(status_{date}.json)を残す。発走時刻の無いレースは recorded_before_start=None(不明、評価に使わない)。

実行: .venv_dl の python
  python scripts/jra_model/finish_v4/finish_v4_live.py --date 20261011 [--record]
"""
import argparse
import csv
import datetime
import hashlib
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "pedigree_reports"))
import ability_v2 as V  # noqa: E402
import age_breeder as AGB  # noqa: E402
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
MODELS = {"M1": F0.SPECS["M1"], "FINAL": FEATS, "F7": AGB.FINAL + AGB.AGE + AGB.BREED}   # F7 は f7_confirm.F7_SPEC と同じ
SNAP = PROS / "snapshots"
JST = datetime.timezone(datetime.timedelta(hours=9))
PROSPECTIVE_START = pd.Timestamp("2026-10-12")      # 確認(2026-10-11)の後の開催日から
log = F0.log


def sha(p: Path):
    return hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None


# ------------------------------------------------------------------ 係数
def params_path(year: int, model: str) -> Path:
    return OUT / (f"live_params_{year}.json" if model == "FINAL" else f"live_params_{year}_{model}.json")


def fit_params(year: int, force: bool = False):
    """その年の係数(2014〜前年)を M1・FINAL・F7 で当てはめる。既にあるファイルは上書きしない(記録の sha が変わるため)。"""
    cache = OUT / f"y{year}_tables.parquet"
    T = pd.read_parquet(cache, filters=[("eval_year", "=", year)]).drop(columns="eval_year")
    tr = MD.derived(T[T["year"].between(F0.TRAIN_START, year - 1)])
    t7 = OUT / f"f7_y{year}_tables.parquet"
    for model, feats in MODELS.items():
        p = params_path(year, model)
        if p.exists() and not force:
            log("live params", year, model, "既にあるので作りません", p.name)
            continue
        src, trm = cache.name, tr
        if model == "F7":
            if not t7.exists():
                AGB.build_tables((year,), year, cache, t7)
            T7 = pd.read_parquet(t7, filters=[("eval_year", "=", year)]).drop(columns="eval_year")
            trm = AGB.add_age(MD.derived(T7[T7["year"].between(F0.TRAIN_START, year - 1)]))
            src = t7.name
        beta, lam = MD.fit_std(trm, feats, 0.0)
        out = {"year": year, "model": model, "train": [F0.TRAIN_START, year - 1], "features": feats, "beta": [float(b) for b in beta],
               "lam": list(lam), "made": time.strftime("%Y-%m-%d %H:%M:%S"), "source": src}
        p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        log("live params", year, model, "lam", np.round(lam, 3))


def load_params(year: int) -> dict:
    out = {}
    for model in MODELS:
        p = params_path(year, model)
        if not p.exists():
            raise FileNotFoundError(f"{p.name} がありません(--fit-params {year} を先に)")
        out[model] = json.loads(p.read_text(encoding="utf-8"))
    return out


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


def profile_table(ids) -> pd.DataFrame:
    """生年月日・生産者: 確認で凍結した profile.parquet を基本にし、そこに無い・欠けている馬だけ data/horse_profile の CSV から読む。"""
    pf = pd.read_parquet(AGB.PROFILE, columns=["horse_id", "birth_date", "breeder_id"]).drop_duplicates("horse_id").set_index("horse_id")
    pf["profile_src"] = "profile.parquet"
    need = [h for h in ids if h not in pf.index or pd.isna(pf.at[h, "birth_date"]) or pd.isna(pf.at[h, "breeder_id"])]
    rows = []
    for h in need:
        p = C.PROJECT_ROOT / "data" / "horse_profile" / f"{h}.csv"
        if not p.exists():
            continue
        with open(p, encoding="utf-8", newline="") as f:
            r = csv.reader(f)
            hdr, row = next(r, None), next(r, None)
        d = dict(zip(hdr or [], row or []))
        rows.append({"horse_id": h, "birth_date": d.get("birth_date") or None, "breeder_id": d.get("breeder_id") or None, "profile_src": "horse_profile"})
    if rows:
        add = pd.DataFrame(rows).set_index("horse_id")
        add["birth_date"] = pd.to_datetime(add["birth_date"], errors="coerce")
        pf = pd.concat([pf.drop(index=[h for h in add.index if h in pf.index]), add])
    return pf


def parse_date(x):
    if not isinstance(x, str) or not x.strip():
        return pd.NaT
    return pd.to_datetime(x.strip().replace(".", "-").replace("/", "-"), errors="coerce")


def card_quality(ent: pd.DataFrame, runs: pd.DataFrame, ped: pd.DataFrame, prof: pd.DataFrame, runs_max: pd.Timestamp) -> pd.DataFrame:
    """レースごとの欠けと遅れ。stale_lag: 馬柱の前走日が走の表の最終日より後(結果の取り込み待ち)。
    past1_not_jra: 前走日が最終日以前なのに走の表にその日の走が無い(地方・海外の走など)。"""
    have = set(zip(runs["horse_id"], runs["date"]))
    p1 = {}
    for rid in ent["race_id"].unique():
        p = RL.NEWS / f"{rid}.csv"
        if p.exists():
            n = pd.read_csv(p, dtype=str, usecols=lambda c: c in ("horse_id", "past1_date"))
            if "past1_date" in n:
                p1.update({(rid, h): parse_date(d) for h, d in zip(n["horse_id"], n["past1_date"])})
    pe = ped.drop_duplicates("horse_id").set_index("horse_id")["ped_src"] if "ped_src" in ped else pd.Series(dtype=object)
    rows = []
    for rid, g in ent.groupby("race_id"):
        lag = other = 0
        for h in g["horse_id"]:
            d = p1.get((rid, h), pd.NaT)
            if pd.notna(d):
                if d > runs_max:
                    lag += 1
                elif (h, d) not in have:
                    other += 1
        src = g["horse_id"].map(pe)
        pr = prof.reindex(g["horse_id"])
        rows.append({"race_id": rid, "n": int(len(g)), "stale_lag": lag, "past1_not_jra": other,
                     "ped_from_name": int((src == "名前から").sum()), "ped_unknown": int((src == "不明").sum() + src.isna().sum()),
                     "birth_missing": int(pr["birth_date"].isna().sum()), "breeder_missing": int(pr["breeder_id"].isna().sum())})
    return pd.DataFrame(rows).set_index("race_id")


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
    prof = profile_table(ra["horse_id"].unique())
    ra["birth_date"] = ra["horse_id"].map(prof["birth_date"])
    ra["breeder_id"] = ra["horse_id"].map(prof["breeder_id"]).replace("", np.nan)
    ra["age_m"] = (ra["date"] - ra["birth_date"]).dt.days / 30.4375
    fin = ra[~ra["nonfinish"]]
    Pf = AB.fit_fig(fin, Y - 1)
    fig = AB.apply_fig(ra, Pf)
    k_b = json.loads(AGB.TUNE_OUT.read_text(encoding="utf-8"))["k_b"]
    bm = AGB.breeder_month(ra, fig, Y - 1, MD.v2_par()["k_s"], k_b).rename("breeder_m")
    at = AGB.age_table(ra)
    g = ra.groupby("horse_id")
    ab = pd.DataFrame({"last_fig": fig.groupby(ra["horse_id"]).shift(1), "kin": ra["kinryo"] - 55.0,
                       "d_kin": (ra["kinryo"] - g["kinryo"].shift(1)).fillna(0), "rest_long": (ra["rest_days"] > 90).astype(float)},
                      index=ra.index)
    R = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")
    flags = S.add_run_cols(ra[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan), R)
    Fv = V.features(ra, Y, MD.v2_par(), AB.sire_S(ra))
    X = pd.concat([ra[["race_id", "horse_id", "umaban_num", "surface", "logd", "class_ord", "is_debut_race"]], ab,
                   flags[["ext", "short", "ext_amt"]], Fv[MD.V2_COLS], at, bm], axis=1)[isnew.to_numpy()]
    for c in ["ext", "short"]:
        X[c] = X[c].fillna(False).astype(float)
    X["ext_amt"] = X["ext_amt"].fillna(0.0)
    X["is_dirt"] = (X["surface"] == "ダ").astype(float)
    X["class_n"] = X["class_ord"] / 7.0
    X["debut"] = X["is_debut_race"].astype(float)
    X["n_start"] = X.groupby("race_id")["umaban_num"].transform("max")
    X = X.join(f, on=["race_id", "horse_id"])
    X = AGB.add_age(MD.derived(X))
    q = card_quality(ent[ent["race_id"].isin(new_ids)], runs, ped, prof, pd.Timestamp(info["runs_max_date"]))
    info["elapsed_sec"] = round(time.time() - t0, 1)
    info["quality_total"] = {k: int(v) for k, v in q.drop(columns="n").sum().items()}
    info["runs_cut_sha256"] = hashlib.sha256(pd.util.hash_pandas_object(
        runs[["race_id", "horse_id", "date"]].sort_values(["race_id", "horse_id"]), index=False).values.tobytes()).hexdigest()
    snap = {"races": races, "ent": ent, "pseudo": pz, "ped": ped, "profile": prof.reindex(ent["horse_id"].unique()).rename_axis("horse_id").reset_index()}
    return races, ent, X, info, res, q, snap


def predict(X: pd.DataFrame, params: dict, col: str = "u") -> pd.DataFrame:
    feats = params["features"]
    assert feats == MODELS[params.get("model", "FINAL")]
    X = X.copy()
    X[col] = X[feats].fillna(0).to_numpy(float) @ np.asarray(params["beta"])
    return X


# ------------------------------------------------------------------ 記録・表示
def race_payload(g: pd.DataFrame, lam, names: dict) -> dict:
    g = g.sort_values("umaban_num")
    t = SR.race_tickets(g["umaban_num"].astype(int).tolist(), g["u"].to_numpy(float), lam, SR.MAIN_RULE)
    return {"umaban": g["umaban_num"].astype(int).tolist(), "horse_id": g["horse_id"].tolist(),
            "horse_name": [names.get(h) for h in g["horse_id"]], "u": [round(float(x), 4) for x in g["u"]]} | t


def save_snapshot(D: pd.Timestamp, snap: dict, stamp: str) -> dict:
    """入力の控え: 出走表・その日の出走表の行・血統ID・生年月日と生産者(出走馬の分)。走の表は runs_cut_sha256 で同一性を確かめる。"""
    d = SNAP / D.strftime("%Y%m%d") / stamp
    d.mkdir(parents=True, exist_ok=True)
    out = {}
    for k, df in snap.items():
        p = d / f"{k}.parquet"
        df = df.reset_index(drop=True)
        df.astype({c: "string" for c in df.columns if df[c].dtype == object}).to_parquet(p, index=False)
        out[k] = {"path": p.relative_to(C.PROJECT_ROOT).as_posix(), "sha256": sha(p)}
    return out


def run(D: pd.Timestamp, race_names: Path | None, do_record: bool):
    P = load_params(D.year)
    races, ent, X, info, res, q, snap = build(D, race_names)
    for model, par in P.items():
        X = predict(X, par, "u" if model == "FINAL" else f"u_{model}")
    names = dict(zip(ent["horse_id"], ent["horse_name"]))
    rinfo = races.set_index("race_id")
    info["snapshot"] = save_snapshot(D, snap, datetime.datetime.now(JST).strftime("%H%M%S"))
    day = {"date": str(D.date()), "rule": SR.MAIN_RULE, "model": "finish_v4 FINAL(主)+ M1・F7(影)", "params_year": P["FINAL"]["year"],
           "info": info, "races": {}}
    for rid, g in X.groupby("race_id"):
        r = rinfo.loc[rid]
        st = r.get("start_time")
        rec = {"race_name": r["race_name"], "venue": r["racecourse"], "race_number": int(r["race_number"]),
               "start_time": st if isinstance(st, str) and st.strip() else None} | race_payload(g, P["FINAL"]["lam"], names)
        rec["models"] = {}
        gs = g.sort_values("umaban_num")
        for model in ("M1", "F7"):
            t = SR.race_tickets(gs["umaban_num"].astype(int).tolist(), gs[f"u_{model}"].to_numpy(float), P[model]["lam"], SR.MAIN_RULE)
            rec["models"][model] = {"u": [round(float(x), 4) for x in gs[f"u_{model}"]], "tickets": t["tickets"], "p_covered": t["p_covered"]}
        rec["quality"] = {k: int(v) for k, v in q.loc[rid].items()} if rid in q.index else None
        day["races"][rid] = rec
    (LIVE / f"finish_v4_{D.strftime('%Y%m%d')}.json").write_text(json.dumps(day, ensure_ascii=False, indent=1), encoding="utf-8")
    log("live", D.date(), len(day["races"]), "レース", info["elapsed_sec"], "秒", "欠け・遅れ", info["quality_total"])
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
                "params_sha256_by_model": {m: sha(params_path(D.year, m)) for m in MODELS},
                "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip(),
                "live_sha256": sha(Path(__file__)), "added": []}
    for rid, r in day["races"].items():
        if rid in rec["races"]:
            continue
        before = None          # 発走時刻が無いレースは不明(評価に使わない)
        if r["start_time"]:
            hh, mm = r["start_time"].split(":")[:2]
            before = now < datetime.datetime(D.year, D.month, D.day, int(hh), int(mm), tzinfo=JST)
        rec["races"][rid] = r | {"recorded_at": run_meta["generated_at"], "recorded_before_start": before}
        run_meta["added"].append(rid)
    rec["runs"].append(run_meta)
    if run_meta["added"]:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    log("前向きの記録", p.name, "追加", len(run_meta["added"]), "/ 合計", len(rec["races"]),
        "(うち発走前", sum(v["recorded_before_start"] is True for v in rec["races"].values()), ")")


def check(D: pd.Timestamp):
    """結果のある過去の日: 出走表から作った効用と、確認と同じ処理の効用(y{年}_utils.parquet)の一致を見る。"""
    params = json.loads((OUT / f"live_params_{D.year}.json").read_text(encoding="utf-8"))
    _, _, X, info, _, q, _ = build(D, None)
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
    out["quality_total"] = info["quality_total"]
    # F7: 2026年の層の F7 の表(f7_y{年}_tables)に、同じ係数を当てた効用と比べる
    p7, t7 = params_path(D.year, "F7"), OUT / f"f7_y{D.year}_tables.parquet"
    if p7.exists() and t7.exists():
        P7 = json.loads(p7.read_text(encoding="utf-8"))
        X7 = predict(X, P7, "u_F7")
        T7 = pd.read_parquet(t7, filters=[("eval_year", "=", D.year)]).drop(columns="eval_year")
        T7 = AGB.add_age(MD.derived(T7[T7["race_id"].isin(X7["race_id"].unique())]))
        T7["u_F7_ref"] = T7[P7["features"]].fillna(0).to_numpy(float) @ np.asarray(P7["beta"])
        j7 = X7.merge(T7[["race_id", "horse_id", "u_F7_ref", "z_front"]].rename(columns={"z_front": "zr"}), on=["race_id", "horse_id"])
        ok7 = ~j7["race_id"].isin(j7.loc[j7["zr"].isna(), "race_id"])
        d7 = (j7["u_F7"] - j7["u_F7_ref"]).abs()
        out["F7"] = {"n_matched": int(len(j7)), "max_abs_diff_ped_complete": float(d7[ok7].max()), "median_abs_diff": float(d7.median()),
                     "corr": float(np.corrcoef(j7["u_F7"], j7["u_F7_ref"])[0, 1])}
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
    ap.add_argument("--force", action="store_true", help="--fit-params で既にある係数のファイルを作り直す")
    ap.add_argument("--check")
    a = ap.parse_args()
    if a.fit_params:
        fit_params(a.fit_params, a.force)
    if a.check:
        check(pd.Timestamp(a.check))
    if a.date_today:
        a.date = datetime.datetime.now(JST).strftime("%Y%m%d")
        logs = C.PROJECT_ROOT / "logs"
        logs.mkdir(exist_ok=True)
        fh = open(logs / f"finish_v4_live_{a.date}.log", "a", encoding="utf-8", buffering=1)
        sys.stdout = sys.stderr = fh      # pythonw.exe(画面なし)でも記録が残るように
    if a.date:
        status = {"date": a.date, "time": datetime.datetime.now(JST).isoformat(timespec="seconds"), "record": a.record}
        try:
            day = run(pd.Timestamp(a.date), Path(a.race_names_csv) if a.race_names_csv else None, a.record)
            status |= {"status": "ok", "n_races": len(day["races"]), "quality_total": day["info"]["quality_total"]}
        except RL.NotReady as e:
            log("まだ作れません:", e)
            status |= {"status": "not_ready", "reason": str(e)}
        except Exception as e:          # すべての例外を状態として残す(前の年の係数などで代わりに記録はしない)
            log("失敗:", repr(e))
            log(traceback.format_exc())
            status |= {"status": "error", "reason": repr(e)}
        if a.record:
            sp = PROS / f"status_{a.date}.json"
            hist = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else []
            hist.append(status)
            sp.write_text(json.dumps(hist, ensure_ascii=False, indent=1), encoding="utf-8")
        if status["status"] == "error":
            sys.exit(1)


if __name__ == "__main__":
    main()
