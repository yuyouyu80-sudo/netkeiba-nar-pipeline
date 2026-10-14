# -*- coding: utf-8 -*-
"""手順3の確認(2020-01〜2024-12、1回だけ)。事前登録: PREREG_cushion_2026_10_13.md 追補1・追補2。

定義は開発(horse_features.py)のまま。違いは追補2のとおり:
- 馬場の偏差 = その日の値 − その競馬場の「その日より前の開催日」の平均(前が無ければ偏差0)、SD は開発期間の値で固定。
- 縮め k=200。
仮説: C1 芝 F3(主)、C2 芝 F2、C3 芝 F1〜F3、C4 ダート F1〜F3、C5 オッズを使わない [u_FINAL] + 芝 F1〜F3(2020年)。
使い方:
  python -X utf8 horse_confirm.py --build          # 特徴を作って confirm_rows.pkl に保存
  python -X utf8 horse_confirm.py --hyp C1         # 仮説ごとに評価(並べて実行してよい)→ confirm_C1.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "jra_model" / "market_pools_2026_10_12"))
sys.path.insert(0, str(HERE))
import mp_common as M  # noqa: E402

_COEF_BOOT = M.coef_boot
import horse_features as HF  # noqa: E402  (開発用に coef_boot を200回へ差し替えるので、確認では元に戻す)

M.coef_boot = _COEF_BOOT
OUT = HF.OUT
K = 200.0
ROWS = OUT / "confirm_rows.pkl"


def dev_sds():
    """開発期間(2025-01〜2026)の走ごとの偏差の SD(horse_features.py と同じ計算)。"""
    d = HF.attach_track(HF.load_runs(), "dev")
    sd = {}
    for c in ("cv", "md"):
        dev = d[c] - d.groupby("vc")[c].transform("mean")
        sd[c] = float(dev.std())
    return sd


def build():
    sd = dev_sds()
    d = HF.load_runs(years=range(2020, 2025))
    b = HF.CF.load_baba("confirm")
    b = b[b["raceday"]][["vc", "dt", "cv", "md"]].sort_values(["vc", "dt"]).copy()
    for c in ("cv", "md"):
        # その日より前の開催日の平均(欠損の日は数えない)
        cnt = b.groupby("vc")[c].transform(lambda s: s.notna().cumsum().shift(fill_value=0))
        sm = b.groupby("vc")[c].transform(lambda s: s.fillna(0).cumsum().shift(fill_value=0))
        prior = sm / cnt.replace(0, np.nan)
        b[f"dev_{c}"] = ((b[c] - prior) / sd[c])
    d = d.merge(b, on=["vc", "dt"], how="left")
    d["x_turf"] = d["dev_cv"]
    d["x_dirt"] = d["dev_md"]
    # 測定値はあるが前の開催日が無い(期間の最初)→ 偏差0
    d.loc[d["cv"].notna() & d["x_turf"].isna(), "x_turf"] = 0.0
    d.loc[d["md"].notna() & d["x_dirt"].isna(), "x_dirt"] = 0.0
    d["x"] = np.where(d["turf"], d["x_turf"], d["x_dirt"])
    for f, key in (("F1", "ped_S_horse_id"), ("F2", "ped_DS_horse_id"), ("F3", "horse_id")):
        d[f"{f}_k200"] = HF.slope_feature(d, key, K)
        print(f, "done", flush=True)
    d.to_pickle(ROWS)
    meta = {"dev_sd": sd, "n_runs": int(len(d)), "dates": [str(d["dt"].min().date()), str(d["dt"].max().date())],
            "share_x_known_turf": float(d.loc[d["turf"], "x"].notna().mean()),
            "share_x_known_dirt": float(d.loc[~d["turf"], "x"].notna().mean()),
            "ped_coverage": float(d["ped_S_horse_id"].notna().mean())}
    (OUT / "confirm_build.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(meta)


HYPS = {"C1": (True, ["F3_k200"], "芝 F3 馬自身×クッション値(主)"),
        "C2": (True, ["F2_k200"], "芝 F2 母の父×クッション値"),
        "C3": (True, ["F1_k200", "F2_k200", "F3_k200"], "芝 F1〜F3 まとめて"),
        "C4": (False, ["F1_k200", "F2_k200", "F3_k200"], "ダート F1〜F3(含水率)"),
        "C5": (True, ["F1_k200", "F2_k200", "F3_k200"], "オッズなし [u_FINAL] + 芝 F1〜F3(2020年)")}


def run(h):
    turf, feats, name = HYPS[h]
    d = pd.read_pickle(ROWS)
    x = HF.race_frame(d, turf)
    x = HF.std(x, feats)
    if h == "C5":
        u = pd.read_parquet(ROOT / "data" / "jra_pipeline" / "finish_v4" / "f7_utils.parquet", columns=["race_id", "horse_id", "u_FINAL"])
        y = x.merge(u, on=["race_id", "horse_id"], how="inner")
        y = y[y.groupby("race_id")["race_id"].transform("size") == y.groupby("race_id")["n"].transform("first")].reset_index(drop=True)
        r = M.evaluate(y, ["u_FINAL"], feats, "win", "clogit", name, n_placebo=20)
    else:
        r = M.evaluate(x, ["logp"], feats, "win", "clogit", name, n_placebo=200 if h == "C1" else 20)
    r.pop("placebo_values", None)
    r["hyp"] = h
    (OUT / f"confirm_{h}.json").write_text(json.dumps(r, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f'{h} {name}: {r["n_races"]}R/{r["n_days"]}月 ΔLL {r["dll"]:+.2f} [{r["lo"]:+.2f}, {r["hi"]:+.2f}] '
          f'プラセボ最大 {r["placebo_max"]:+.2f} 順位p {r["perm_p"]:.2f} 合格={r["pass"]}', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--hyp", choices=list(HYPS))
    a = ap.parse_args()
    if a.build:
        build()
    if a.hyp:
        run(a.hyp)


if __name__ == "__main__":
    main()
