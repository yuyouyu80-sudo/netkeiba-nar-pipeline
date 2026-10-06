# -*- coding: utf-8 -*-
"""血統レーダー v2 の前向き検証(事前登録 追補6)(2026-10-07)。

radar_v2/prospective/{date}.json(radar_v2_live.py --record、発走前の記録)と、あとで取れた race_results を突き合わせる。
係数はすべて固定(テスト期間で推定し直さない):
  P1(v3 の確認): v3 u = b1·S + G3[発表された馬場]  対  C2 u = β·[S, 5成分](radar_v2_dl_eval.json の 2026年用、学習 2014〜2025年)
  P2(v2 の確認): A1 u = β_full·[S, F]  対  A0 u = β_base·[S](radar_v2_eval.json の main、推定 2018〜2020年)
対象: recorded_before_start が真、平地、完走5頭以上(R4 と同じ)。記録に無い馬・競走中止の馬は除く(R4 と同じく完走馬の上位3頭の PL)。
判定(追補6で固定): P1 は 2027年3月28日までの開催分で1回、P2 は 2027年12月28日までの開催分で1回。各 片側p < 0.0125
(Bonferroni m=2、日付×競馬場のブロックブートストラップ 2,000回)。途中の集計は記述のみ(途中で止めない)。
出力: radar_v2/prospective_eval.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import radar_v2_common as C  # noqa: E402
import radar_v2_eval as E  # noqa: E402

PROS = C.OUT_DIR / "prospective"
OUT = C.OUT_DIR / "prospective_eval.json"
CUTOFF = {"P1": pd.Timestamp("2027-03-28"), "P2": pd.Timestamp("2027-12-28")}
ALPHA = 0.0125   # 片側、Bonferroni m=2
GOINGS = ["良", "稍重", "重", "不良"]


def load() -> tuple[pd.DataFrame, dict]:
    rows, cov = [], {"days": 0, "races_recorded": 0, "races_before_start": 0, "races_with_results": 0}
    for p in sorted(PROS.glob("*.json")):
        rec = json.loads(p.read_text(encoding="utf-8"))
        D = pd.Timestamp(rec["date"])
        cov["days"] += 1
        res_p = C.RESULTS_DIR / str(D.year) / f"{D.strftime('%Y%m%d')}.csv"
        res = pd.read_csv(res_p, dtype=str) if res_p.exists() else None
        for r in rec["races"]:
            cov["races_recorded"] += 1
            if not r["recorded_before_start"]:
                continue
            cov["races_before_start"] += 1
            if res is None or not (res["race_id"] == r["race_id"]).any():
                continue
            rr = res[res["race_id"] == r["race_id"]]
            if C.is_jump(rr["race_name"]).any():
                continue
            cov["races_with_results"] += 1
            pos = dict(zip(rr["horse_id"], C.parse_finish(rr["finish_pos"])))
            going = rr["going"].iloc[0]
            for h in r["horses"]:
                g3 = h.get("G3") or {}
                rows.append({"date": D, "race_id": r["race_id"], "venue": r["venue"], "horse_id": h["horse_id"], "umaban_num": h["umaban"],
                             "pos": pos.get(h["horse_id"], np.nan), "S": h["S"], "F": h["F"],
                             **{f"c{i}": v for i, v in enumerate(h["comps"])},
                             "G3": g3.get(going, np.nan), "G3_良": g3.get("良", np.nan), "going": going})
    return pd.DataFrame(rows), cov


def per_race(d: pd.DataFrame) -> pd.DataFrame:
    d = d[d["pos"].notna()].copy()
    d = d[d.groupby("race_id")["race_id"].transform("size") >= 5]   # 完走5頭以上(R4 の runs と同じ)
    if d.empty:
        return pd.DataFrame()
    dl = json.loads((C.OUT_DIR / "radar_v2_dl_eval.json").read_text(encoding="utf-8"))
    r4 = json.loads((C.OUT_DIR / "radar_v2_eval.json").read_text(encoding="utf-8"))["main"]
    b1 = json.loads((C.OUT_DIR / "v3_export_2026.json").read_text(encoding="utf-8"))["b1"]
    bC2 = np.array(dl["betas"]["2026"]["C2"])
    d["u_C2"] = d[["S", "c0", "c1", "c2", "c3", "c4"]].fillna(0).to_numpy() @ bC2
    d["u_v3"] = b1 * d["S"].fillna(0) + d["G3"].fillna(0)
    d["u_v3_良"] = b1 * d["S"].fillna(0) + d["G3_良"].fillna(0)
    d["u_A0"] = np.array(r4["beta_base_only"])[0] * d["S"].fillna(0)
    bf = np.array(r4["beta_full"])
    d["u_A1"] = d[["S", "F"]].fillna(0).to_numpy() @ bf
    out = {}
    for k in ["u_C2", "u_v3", "u_v3_良", "u_A0", "u_A1"]:
        pl = E.PL(d, [k])
        out[k] = pd.Series(pl.race_ll(np.array([1.0])), index=pl.race_ids)
    R = pd.DataFrame(out)
    first = d.drop_duplicates("race_id").set_index("race_id")
    R["date"], R["venue"] = first.loc[R.index, "date"], first.loc[R.index, "venue"]
    R["block"] = R["date"].dt.strftime("%Y%m%d") + "_" + R.index.str[4:6]
    return R


def summarize(R: pd.DataFrame, a: str, b: str, name: str) -> dict:
    d = (R[a] - R[b]).dropna()
    if len(d) < 2:
        return {"name": name, "n_races": int(len(d))}
    ci, p, se = E.block_boot(d.to_numpy(), R.loc[d.index, "block"].to_numpy(), with_p=True)
    return {"name": name, "n_races": int(len(d)), "dll_per_race": round(float(d.mean()), 5), "ci95_block": ci, "p_one_sided": round(p, 5),
            "by_date": d.groupby(R.loc[d.index, "date"].dt.strftime("%Y-%m-%d")).mean().round(4).to_dict()}


def main():
    d, cov = load()
    res = {"coverage": cov, "cutoff": {k: str(v.date()) for k, v in CUTOFF.items()},
           "note": "途中の集計は記述のみ。判定は追補6の期日に1回(P1: 2027-03-28まで、P2: 2027-12-28まで)、各 片側p < 0.0125。"}
    R = per_race(d) if len(d) else pd.DataFrame()
    if R.empty:
        res["status"] = "結果の出た記録がまだありません"
    else:
        today = pd.Timestamp.today().normalize()
        for key, (a, b, name) in {"P1": ("u_v3", "u_C2", "v3 対 C2(発表された馬場)"),
                                  "P2": ("u_A1", "u_A0", "v2 主仮説 A1 対 A0")}.items():
            sub = R[R["date"] <= CUTOFF[key]]
            s = summarize(sub, a, b, name)
            s["final"] = bool(today > CUTOFF[key] + pd.Timedelta(days=7))   # 期日の開催の結果が揃ってから
            if s["final"] and "p_one_sided" in s:
                s["pass"] = bool(s["p_one_sided"] < ALPHA)
            res[key] = s
        res["P1_sens_going_good"] = summarize(R, "u_v3_良", "u_C2", "感度: v3 を「良」と仮定した値で(発走前の想定)")
    OUT.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=1, default=str)[:3000])


if __name__ == "__main__":
    main()
