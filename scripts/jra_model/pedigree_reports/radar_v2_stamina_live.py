# -*- coding: utf-8 -*-
"""血統レーダー v2: スタミナの軸(A 持続力・B 距離延長への反応)の前向きの記録 P3(事前登録 追補7、2026-10-07)。

**記述のみ**(検出力が足りないため判定はしない)。追補6 の前向きの記録(radar_v2_live.py → radar_v2/prospective/、P1・P2)とは別のファイル
radar_v2/prospective_stamina/{date}.json に書き、既存の記録・判定には触れない。

- 出走表・走の表・血統ID・その日の出走表の行は radar_v2_live と同じ関数(load_card・runs_before・ped_ids・pseudo_rows)。
- z_A・z_B: 時点表 stam_anc_{Y}(Y−1年末まで、本人の走を除く)。延長・短縮・延長の幅: 走の表の前走(学習と同じ定義)。
- q_A: 同等レース(その日より前)の縮約平均は radar_v2_live.req_new(radar_v2_reference.build_req と同じ式)、
  回帰の係数は stamina_fixed.json。馬場は発走前には分からないので 良・稍重・重・不良 の4通り。
- --record: レースごとに最初の記録が正本。発走時刻より後に作った分は recorded_before_start=false。
- --check: 結果のある過去の日について、出走表から作った値と、学習の処理(結果の行)から作った値が一致するか。

使い方: python radar_v2_stamina_live.py --date 20261010 [--record] [--check] [--date-today]
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_live as L  # noqa: E402
import radar_v2_stamina as S  # noqa: E402

PROS_ST = C.OUT_DIR / "prospective_stamina"
PROS_ST.mkdir(exist_ok=True)
START = pd.Timestamp("2026-10-07")
log = A.log


def compute(D: pd.Timestamp, race_names: Path | None) -> dict:
    t0 = time.time()
    races, ent, src, res = L.load_card(D, race_names)
    runs, info = L.runs_before(D)
    info["card_source"] = src
    ped = L.ped_ids(ent)
    pseudo = L.pseudo_rows(races, ent, runs, D, ped)
    new_ids = pseudo["race_id"].unique().tolist()
    runs_ext = pd.concat([runs, pseudo.drop(columns=["ped_src", "sire", "bms", "horse_name", "sex_age", "ent_src", "umaban_num"])],
                         ignore_index=True)
    assert not runs_ext.duplicated(["horse_id", "date"]).any()
    base = S.load_base(runs_ext)
    fx = S.load_fixed()
    R, _ = S.prepare(base, fx)
    # その日のレースの同等レースの平均は、その日より前のレースだけで(build_req は値の空の行を落とすため、当日の空の行に使えない)
    prior = R[R["date"] < D]
    X = R.loc[new_ids]
    for val in ["pace_hi", "c_z"]:
        R.loc[new_ids, f"pri_{val}"] = L.req_new(prior, X, val, fx["q_k"][val])["req"]
    q4 = {g: S.apply_q(R.loc[new_ids], fx, going=g)["qA"] for g in S.GOINGS}
    flags = S.add_run_cols(base[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan), R)
    df = base[base["race_id"].isin(new_ids)].copy()
    df = df.join(flags.set_index(["race_id", "horse_id"])[["ext", "short", "ext_amt"]], on=["race_id", "horse_id"])
    T = S.stam_tables_class()(D.year)
    for ax in S.AXES:
        df[f"z_{ax}"] = T.z(df, ax, "-")
    log("計算", round(time.time() - t0), "秒", len(new_ids), "レース", len(df), "頭")
    return {"races": races, "ent": ent, "ped": ped, "df": df, "q4": q4, "R": R, "info": info, "res": res, "base": base}


def r4(x):
    return None if x is None or pd.isna(x) else round(float(x), 4)


def record(D: pd.Timestamp, out: dict, race_names: Path | None) -> dict:
    p = PROS_ST / f"{D.strftime('%Y%m%d')}.json"
    if D < START:
        sys.exit(f"{D.date()} は前向きの記録の対象外({START.date()} 以降のみ)。")
    if out["res"] is not None:
        sys.exit(f"{D.date()} は結果が既にあるため、前向きの記録は作りません。")
    now = datetime.datetime.now(L.JST)
    df, races = out["df"], out["races"]
    T = A.TABLE_DIR
    run = {"generated_at": now.isoformat(timespec="seconds"),
           "inputs": out["info"] | {"stamina_fixed_sha256": L.sha(S.FIXED), "table_year": D.year,
                                    "stam_tables_sha256": L.sha(T / f"stam_anc_{D.year}.parquet"),
                                    "stam_hv_sha256": L.sha(T / f"stam_hv_{D.year}.parquet")},
           "code": {"git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=C.PROJECT_ROOT).stdout.strip(),
                    "radar_v2_stamina_live_sha256": L.sha(Path(__file__)), "radar_v2_stamina_sha256": L.sha(Path(S.__file__))},
           "added": []}
    rec = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {
        "date": str(D.date()), "runs": [], "races": [],
        "note": "事前登録 追補7 の前向きの記録 P3(スタミナ A・B、記述のみ)。レースごとに最初の記録が正本。追補6 の P1・P2 とは別。"}
    have = {r["race_id"] for r in rec["races"]}
    for _, r in races[races["skip"].isna()].iterrows():
        rid = r["race_id"]
        if rid not in out["q4"]["良"].index or rid in have:
            continue
        st = r.get("start_time") if isinstance(r.get("start_time"), str) else None
        before = False
        if st:
            hh, mm = st.split(":")
            before = now < datetime.datetime(D.year, D.month, D.day, int(hh), int(mm), tzinfo=L.JST)
        g = df[df["race_id"] == rid].sort_values("umaban_num")
        run["added"].append(rid)
        rec["races"].append({
            "race_id": rid, "venue": r["racecourse"], "race_number": int(r["race_number"]), "start_time": st,
            "recorded_at": run["generated_at"], "recorded_before_start": before, "surface": r["surface"], "distance": int(r["distance"]),
            "qA": {gg: r4(out["q4"][gg].get(rid)) for gg in S.GOINGS},
            "horses": [{"horse_id": h["horse_id"], "umaban": int(h["umaban_num"]), "z_stamA": r4(h["z_stamA"]), "z_stamB": r4(h["z_stamB"]),
                        "z_stamC": r4(h["z_stamC"]), "ext": bool(h["ext"]), "short": bool(h["short"]), "ext_amt": r4(h["ext_amt"])}
                       for _, h in g.iterrows()]})
    rec["runs"].append(run)
    if run["added"]:
        tmp = p.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(p)
    log("P3 の記録", p.name, "今回追加", len(run["added"]), "/ 合計", len(rec["races"]), "レース")
    return run


def check(D: pd.Timestamp, out: dict) -> dict:
    """結果のある日: 出走表から作った値 と 学習の処理(stamina_race.parquet の qA、結果の行の前走・時点表)の値の差。"""
    df = out["df"]
    full = S.load_base()
    full = full[full["date"] <= D]
    flags = S.add_run_cols(full[["race_id", "horse_id", "logd", "prev_logd", "dbucket"]].assign(perf_adj=np.nan),
                           pd.DataFrame(columns=["pace_hi", "c_z"]))
    ref = full[full["date"] == D].join(flags.set_index(["race_id", "horse_id"])[["ext", "short", "ext_amt"]], on=["race_id", "horse_id"])
    T = S.stam_tables_class()(D.year)
    for ax in S.AXES:
        ref[f"z_{ax}"] = T.z(ref, ax, "-")
    m = df.merge(ref, on=["race_id", "horse_id"], suffixes=("", "_ref"))
    rep = {"horses_matched": int(len(m)), "horses_live": int(len(df)), "horses_results": int(len(ref))}
    for c in ["z_stamA", "z_stamB", "z_stamC", "ext_amt"]:
        d = (m[c] - m[f"{c}_ref"]).abs()
        rep[c] = {"max_abs_diff": r4(d.max()), "n_both": int(d.notna().sum())}
    for c in ["ext", "short"]:
        rep[c] = {"n_mismatch": int((m[c].astype(bool) != m[f"{c}_ref"].astype(bool)).sum())}
    sr = pd.read_parquet(C.OUT_DIR / "stamina_race.parquet")
    going = full.drop_duplicates("race_id").set_index("race_id")["going"]   # 実際の(発走前に発表された)馬場
    qd = []
    for rid in df["race_id"].unique():
        g = going.get(rid)
        if rid in sr.index and g in S.GOINGS:
            qd.append(abs(out["q4"][g][rid] - sr.at[rid, "qA"]))
    rep["qA_actual_going"] = {"n": len(qd), "max_abs_diff": r4(max(qd)) if qd else None}
    log("照合", rep)
    return rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--race-names", type=Path)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--date-today", action="store_true")
    a = ap.parse_args() if "--date-today" not in sys.argv else ap.parse_args(sys.argv[1:] + ["--date", time.strftime("%Y%m%d")])
    D = pd.Timestamp(a.date)
    if a.date_today:
        (C.PROJECT_ROOT / "logs").mkdir(exist_ok=True)
        sys.stdout = sys.stderr = open(C.PROJECT_ROOT / "logs" / f"radar_v2_stamina_live_{a.date}.log", "a", encoding="utf-8", buffering=1)
        if D.weekday() not in (5, 6, 0):
            log(a.date, "は土日月ではないため何もしません")
            return 0
    try:
        out = compute(D, a.race_names)
    except L.NotReady as e:
        log("何もしません:", e)
        return 0
    if a.check:
        rep = check(D, out)
        (C.OUT_DIR / "live" / f"stamina_check_{D.strftime('%Y%m%d')}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    if a.record:
        record(D, out, a.race_names)


if __name__ == "__main__":
    main()
