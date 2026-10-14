# -*- coding: utf-8 -*-
"""手順3: 馬ごとの「クッション値・含水率への向き不向き」を前日までの成績で作り、確定オッズへの上乗せを測る(2026-10-13、開発)。
事前登録: PREREG_cushion_2026_10_13.md

各走の物差し r = (確定オッズでの人気順位 − 着順)/(頭数−1)(オッズに対する上振れ、+ほど人気より上)。
馬場の偏差 x: 芝 = その日の朝のクッション値 − 競馬場ごとの平均(開発期間)を全体のSDで割ったもの、
            ダート = その日の朝のダートのゴール前含水率で同じもの。
向き不向き(傾き) = Σ x·r / (Σ x² + k)(前日までの、その父/母の父/馬自身の走だけ。k で縮める)。
特徴 = 傾き × 今回の x。F1 父・F2 母の父・F3 馬自身(芝)、F1d〜F3d(ダート)。
評価: 勝ち馬の条件付きロジット [log p] + 特徴。月ごとに抜く交差検証。オッズを使わない参考: 2026年の [u_FINAL] + 特徴。
出力: data/jra_pipeline/cushion_2026_10_13/horse_result.json、horse_rows.pkl
"""
import glob
import json
import sys
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "market_pools_2026_10_12"))
sys.path.insert(0, str(HERE))
import mp_common as M  # noqa: E402
import cushion_forecast as CF  # noqa: E402

M.coef_boot = partial(M.coef_boot, n_boot=200)  # 開発なので200回
OUT = CF.OUT
KS = (20.0, 50.0, 200.0)


def load_runs(years=(2025, 2026)):
    cols = ["race_id", "surface", "umaban", "finish_pos", "odds_final", "horse_id"]
    fr = []
    for y in years:
        for f in sorted(glob.glob(str(ROOT / "data" / "race_results" / str(y) / f"{y}*.csv"))):
            d = pd.read_csv(f, dtype=str, usecols=lambda c: c in cols)
            d["date"] = Path(f).stem
            fr.append(d)
    d = pd.concat(fr, ignore_index=True)
    d = d[~d["surface"].astype(str).str.contains("障")]
    d["fin"] = pd.to_numeric(d["finish_pos"], errors="coerce")
    d["odds"] = pd.to_numeric(d["odds_final"], errors="coerce")
    d = d[d["odds"] > 0].drop_duplicates(["race_id", "umaban"]).copy()
    d["n"] = d.groupby("race_id")["odds"].transform("size")
    d["pop"] = d.groupby("race_id")["odds"].rank(method="first")
    d["r"] = (d["pop"] - d["fin"]) / (d["n"] - 1)
    d["dt"] = pd.to_datetime(d["date"])
    d["vc"] = d["race_id"].str[4:6]
    d["turf"] = d["surface"].astype(str).str.contains("芝")
    ped = pd.read_parquet(ROOT / "data" / "jra_pipeline" / "pedigree_reports" / "radar_v2" / "ped_ids.parquet")
    d = d.merge(ped[["horse_id", "ped_S_horse_id", "ped_DS_horse_id"]], on="horse_id", how="left")
    return d


def attach_track(d, period="dev"):
    # 2026-10-14: 確認期間のデータが届いたので、開発は開発期間だけを読む(追補2)
    b = CF.load_baba(period)
    b = b[b["raceday"]][["vc", "dt", "cv", "md"]]
    d = d.merge(b, on=["vc", "dt"], how="left")
    for c, name in (("cv", "x_turf"), ("md", "x_dirt")):
        dev = d[c] - d.groupby("vc")[c].transform("mean")
        d[name] = dev / dev.std()
    d["x"] = np.where(d["turf"], d["x_turf"], d["x_dirt"])
    return d


def slope_feature(d, key, k):
    """前日までのその主体の走(同じ芝/ダート)で Σx·r と Σx² を累積し、傾き × 今回の x。"""
    out = np.zeros(len(d))
    for turf in (True, False):
        m = (d["turf"] == turf).to_numpy()
        s = d[m & d[key].notna().to_numpy()]
        v = s.dropna(subset=["x", "r"])
        g = v.assign(xr=v["x"] * v["r"], xx=v["x"] ** 2).groupby([key, "dt"])[["xr", "xx"]].sum().reset_index().sort_values([key, "dt"])
        # 前日までの累積: 日ごとの累積和を作り、前の日付の値を引く(当日の走は含めない)
        g["txr"], g["txx"] = g.groupby(key)["xr"].cumsum(), g.groupby(key)["xx"].cumsum()
        q = s[[key, "dt"]].reset_index()
        q = pd.merge_asof(q.sort_values("dt"), g[[key, "dt", "txr", "txx"]].sort_values("dt").rename(columns={"dt": "gdt"}),
                          left_on="dt", right_on="gdt", by=key, allow_exact_matches=False, direction="backward")
        sl = (q["txr"].fillna(0) / (q["txx"].fillna(0) + k)).to_numpy()
        out[q["index"].to_numpy()] = sl * d.loc[q["index"], "x"].fillna(0).to_numpy()
    return out


def race_frame(d, surface_turf):
    x = d[(d["turf"] == surface_turf) & d["x"].notna()].copy()
    w = x.groupby("race_id")["fin"].transform(lambda s: (s == 1).sum())
    x = x[(x["n"] >= 6) & (w >= 1)].copy()
    x["p"] = 1 / x["odds"]
    x["p"] /= x.groupby("race_id")["p"].transform("sum")
    x["logp"] = np.log(x["p"])
    x["win"] = (x["fin"] == 1).astype(float)
    x["venue"] = x["vc"]
    x["day"] = x["date"]
    x["date"] = x["day"].str[:6]  # 交差検証は月ごと(ブロックは月×競馬場)
    return x.reset_index(drop=True)


def std(x, cols):
    for c in cols:
        s = x[c].std()
        x[c] = x[c] / s if s > 0 else x[c]
    return x


def main():
    d = attach_track(load_runs())
    names = {"F1": "ped_S_horse_id", "F2": "ped_DS_horse_id", "F3": "horse_id"}
    for f, key in names.items():
        for k in KS:
            d[f"{f}_k{int(k)}"] = slope_feature(d, key, k)
        print(f, "done", flush=True)
    d.to_pickle(OUT / "horse_rows.pkl")
    res = {"k_screen": {}, "main": {}, "odds_free_2026": {}}
    for turf, tag in ((True, "芝"), (False, "ダ")):
        x = race_frame(d, turf)
        fc = [c for c in x.columns if c[:2] in ("F1", "F2", "F3") and "_k" in c]
        x = std(x, fc)
        res["k_screen"][tag] = {}
        for c in fc:
            dll = M.lodo_dll(x, ["logp"], [c], "win", "clogit")
            tot, lo, hi, _ = M.block_ci(x, dll)
            res["k_screen"][tag][c] = [tot, lo, hi]
            print(tag, c, f"{tot:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
        # 主: k=50(事前に中間の値として決めた。縮めの選択は参考の表で見る)
        for f in ("F1", "F2", "F3"):
            r = M.evaluate(x, ["logp"], [f"{f}_k50"], "win", "clogit", f"{tag} {f} ×馬場(k=50)", n_placebo=20)
            r.pop("placebo_values", None)
            res["main"][f"{tag}_{f}"] = r
            print(f'{r["name"]}: {r["n_races"]}R ΔLL {r["dll"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] プラセボ最大 {r["placebo_max"]:+.2f} '
                  f'順位p {r["perm_p"]:.2f} 合格={r["pass"]}', flush=True)
        r = M.evaluate(x, ["logp"], ["F1_k50", "F2_k50", "F3_k50"], "win", "clogit", f"{tag} F1〜F3 まとめて", n_placebo=20)
        r.pop("placebo_values", None)
        res["main"][f"{tag}_all"] = r
        print(f'{r["name"]}: ΔLL {r["dll"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] 合格={r["pass"]}', flush=True)
        # オッズを使わない参考(2026年、finish_v4 の u_FINAL)
        u = pd.read_parquet(ROOT / "data" / "jra_pipeline" / "finish_v4" / "y2026_utils.parquet", columns=["race_id", "horse_id", "u_FINAL"])
        y = x.merge(u, on=["race_id", "horse_id"], how="inner")
        y = y[y.groupby("race_id")["race_id"].transform("size") == y.groupby("race_id")["n"].transform("first")]
        if y["race_id"].nunique() > 50:
            dll = M.lodo_dll(y, ["u_FINAL"], ["F1_k50", "F2_k50", "F3_k50"], "win", "clogit")
            tot, lo, hi, _ = M.block_ci(y, dll)
            res["odds_free_2026"][tag] = {"n_races": int(y["race_id"].nunique()), "dll": tot, "lo": lo, "hi": hi}
            print(tag, "オッズなし [u_FINAL]+F1〜F3", int(y["race_id"].nunique()), f"{tot:+.2f} [{lo:+.2f}, {hi:+.2f}]", flush=True)
    (OUT / "horse_result.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
