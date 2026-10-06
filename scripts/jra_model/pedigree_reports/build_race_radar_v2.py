# -*- coding: utf-8 -*-
"""血統レーダー v2 の R5: 開催日のレースごとに、v2 の血統の値・レースの基準・適合度をレポート用 JSON にする(2026-10-06)。

- 対象: race_fit_score_data.json の各レース(既存 v1 レポートと同じ開催日。障害戦は除く)。
- 血統の値: 時点表 anc_{Y}(Y = 開催年、Y−1年末までのデータ、本人の走を除く)。radar_v2_eval.features_for_year と同じ計算。
- 血統+実績: radar_v2_eval.add_own(その日より前の本人の走だけで更新)。
- レースの基準: race_req.parquet(その日より前の同等レース)。ペース別(速い/平均/遅い)も。
- 採否(事前登録 追補3、2026-10-06 ユーザー了承): 芝ダ・脚質・キレ・距離帯(短〜中)は v2、新馬・長距離帯は参考表示。
  脚質・キレは「予測上の価値は未確認(記述)」と明記する。
出力: data/jra_pipeline/pedigree_reports/race_radar_v2_data.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_aptitude as A  # noqa: E402
import radar_v2_eval as E  # noqa: E402

BASE = C.PROJECT_ROOT / "data" / "jra_pipeline" / "pedigree_reports"
TODAY = BASE / "race_fit_score_data.json"
OUT = BASE / "race_radar_v2_data.json"
TIERS = ["速い", "平均", "遅い"]
LONG = E.BUCKETS[3]


def r3(x):
    return None if x is None or not np.isfinite(x) else round(float(x), 3)


def axis_meta():
    """軸ごとの信頼度(E1、内側検証の父のSB)と表示区分(追補3の採否)。"""
    def e1(axis, lev="-"):
        v = E.INNER["axes"][axis]["levels"].get(lev, {}).get("E1_sire_sb")
        return r3(v) if v is not None else None
    m = {
        "front": {"label": "脚質(前に行く)", "status": "記述", "note": "偏り・信頼性はv1より大きく改善。予測上の価値は未確認", "e1": e1("front")},
        "kire": {"label": "キレ(上がり)", "status": "記述", "note": "偏り・信頼性はv1より大きく改善。予測上の価値は未確認", "e1": e1("kire")},
        "surface": {"label": "芝ダ適性(今回)", "status": "予測", "note": "父の総合力を超える予測上の価値あり(R4の内訳で最大)",
                    "e1": {s: e1("surface", s) for s in ["芝", "ダ"]}},
        "dbucket": {"label": "距離帯適性(今回)", "status": "参考", "note": "偏り・信頼性は合格。予測上の価値ははっきりしない(長距離帯は信頼度低)",
                    "e1": {b: e1("dbucket", b) for b in E.BUCKETS}},
        "debut": {"label": "新馬適性", "status": "参考", "note": "新馬戦の予測には効くが、共通の偏りが一部の時点表で残る", "e1": e1("debut")},
        "S": {"label": "父の総合力", "status": "基準", "note": "適合度とは別の「強さ」の軸(主仮説の基準モデル)", "e1": None},
    }
    return m


def ped_rows(horse_ids):
    rows = []
    for h in horse_ids:
        p = C.PED_DIR / f"{h}.csv"
        if p.exists():
            d = pd.read_csv(p, dtype=str, usecols=["horse_id", "ped_S_horse_id", "ped_DS_horse_id", "ped_SS_horse_id"]).iloc[0].to_dict()
        else:
            d = {"horse_id": h, "ped_S_horse_id": None, "ped_DS_horse_id": None, "ped_SS_horse_id": None}
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    today = json.loads(TODAY.read_text(encoding="utf-8"))
    race_ids = [r["race_id"] for r in today["races"] if r.get("surface") in ("芝", "ダ") and not r.get("skipped_reason")]
    year = int(today["today"][:4])
    runs = pd.read_parquet(A.RUNS)
    runs = A.add_tiers(runs, A.fixed_thresholds(runs))
    runs["umaban_num"] = pd.to_numeric(runs["umaban"], errors="coerce")
    req = pd.read_parquet(C.OUT_DIR / "race_req.parquet")
    kh = E.estimate_kh(runs)
    f = E.add_own(runs, E.features_for_year(runs, req, year), year, kh)
    f = f[f["race_id"].isin(race_ids)]
    T = E.Tables(year)
    sd = {a: E.REQ_META[f"sd_req_{a}_2018_2020"] for a in ["front", "kire"]}
    meta = axis_meta()
    out = {"race_date": today["today"], "table_year": year, "table_window_end": f"{year - 1}年末",
           "axes": meta, "sd_req": sd, "rel": {"front": E.rel("front"), "kire": E.rel("kire"), "debut": E.rel("debut"),
                                                "surface": {s: E.rel("surface", s) for s in ["芝", "ダ"]},
                                                "dbucket": {b: E.rel("dbucket", b) for b in E.BUCKETS}},
           "races": []}
    for tr in today["races"]:
        rid = tr["race_id"]
        if rid not in race_ids:
            out["races"].append({"race_id": rid, "race_number": tr["race_number"], "race_name": tr["race_name"],
                                 "skipped": tr.get("skipped_reason") or "障害戦等のため対象外"})
            continue
        g = f[f["race_id"] == rid]
        rq = req.loc[rid] if rid in req.index else None
        info = runs[runs["race_id"] == rid].iloc[0] if (runs["race_id"] == rid).any() else None
        surface, distance = tr["surface"], int(tr["distance_m"])
        bucket = C.distance_bucket(distance) if hasattr(C, "distance_bucket") else None
        if info is not None:
            bucket = info["dbucket"]
        debut = bool(info["is_debut_race"]) if info is not None else ("新馬" in tr["race_name"])
        q = {}
        for a in ["front", "kire"]:
            q[a] = {"全体": r3(rq[f"req_{a}"] / sd[a]) if rq is not None else None}
            for t in TIERS:
                q[a][t] = r3(rq[f"req_{a}_{t}"] / sd[a]) if rq is not None else None
        race = {"race_id": rid, "race_number": tr["race_number"], "race_name": tr["race_name"], "surface": surface,
                "distance": distance, "dbucket": bucket, "is_debut": debut, "q": q,
                "req_rule": {a: (rq[f"rule_{a}"] if rq is not None else None) for a in ["front", "kire"]},
                "req_n": {a: (int(rq[f"n_matched_{a}"]) if rq is not None else None) for a in ["front", "kire"]},
                "actual_pace": (rq["pace_tier"] if rq is not None else None), "horses": []}
        # 出走表の全馬(完走していない馬は runs に無いので、血統のみを時点表から直接出す)
        ent = pd.DataFrame(tr["horses"])
        miss = ent[~ent["horse_id"].isin(g["horse_id"])]
        extra = None
        if len(miss):
            extra = ped_rows(miss["horse_id"])
            extra["S"] = T.z(extra, "perf_adj", "-")
            for a in ["front", "kire", "debut"]:
                extra[f"z_{a}"] = T.z(extra, a, "-")
            for s in ["芝", "ダ"]:
                extra[f"z_s|{s}"] = T.z(extra, "surface", s)
            for b in E.BUCKETS:
                extra[f"z_b|{b}"] = T.z(extra, "dbucket", b)
            extra = extra.set_index("horse_id")
        gi = g.set_index("horse_id")
        for _, h in ent.iterrows():
            hid = h["horse_id"]
            src = gi.loc[hid] if hid in gi.index else (extra.loc[hid] if extra is not None and hid in extra.index else None)
            if src is None:
                continue
            ran = hid in gi.index
            ped = {"S": r3(src["S"]), "front": r3(src["z_front"]), "kire": r3(src["z_kire"]),
                   "surface": r3(src[f"z_s|{surface}"]), "dbucket": r3(src[f"z_b|{bucket}"]) if bucket else None,
                   "debut": r3(src["z_debut"]),
                   "surf_both": {s: r3(src[f"z_s|{s}"]) for s in ["芝", "ダ"]}}
            own = None
            if ran:
                own = {"S": r3(src["S_own"]), "front": r3(src["zo_front"]), "kire": r3(src["zo_kire"]),
                       "surface": r3(src["zo_surface"]), "dbucket": r3(src["zo_dbucket"]), "debut": r3(src["z_debut"]),
                       "n_prev": int(src["perf_adj_pn"]) if np.isfinite(src["perf_adj_pn"]) else 0}
            race["horses"].append({"umaban": int(h["umaban"]), "horse_name": h["horse_name"], "sire": h.get("sire_name"),
                                   "bms": h.get("bms_name"), "finish": int(src["pos"]) if ran else None, "ran": ran,
                                   "ped": ped, "own": own})
        race["horses"].sort(key=lambda x: x["umaban"])
        out["races"].append(race)
        print(rid, tr["race_name"], len(race["horses"]), "頭", "基準", race["req_rule"], flush=True)
    # 検証の要約(R4 本番と追補)
    R4 = json.loads((C.OUT_DIR / "radar_v2_eval.json").read_text(encoding="utf-8"))
    AD = json.loads((C.OUT_DIR / "radar_v2_eval_addendum.json").read_text(encoding="utf-8"))
    out["validation"] = {
        "main": {"dll": R4["main"]["dll_per_race"], "ci": R4["main"]["ci95_block"], "n_races": R4["n_test_races"]},
        "market": {"dll": R4["secondary"][-1]["dll_per_race"], "ci": R4["secondary"][-1]["ci95_block"]},
        "axis_single": {k: {"dll": v["dll_per_race"], "ci": v["ci95_block"]} for k, v in AD["axis_single"].items()},
        "by_class": AD["main_by_race_class"],
        "top1": {"S": AD["effect_size"]["S"]["top1_accuracy"], "S+F": AD["effect_size"]["S+F"]["top1_accuracy"]}}
    dl = C.OUT_DIR / "radar_v2_dl_eval.json"
    if dl.exists():
        d = json.loads(dl.read_text(encoding="utf-8"))
        out["validation"]["dl"] = {"dll": d["primary_5"]["dll_per_race"], "ci": d["primary_5"]["ci95_block"]}
    OUT.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("wrote", OUT, round(OUT.stat().st_size / 1024), "KB")


if __name__ == "__main__":
    main()
