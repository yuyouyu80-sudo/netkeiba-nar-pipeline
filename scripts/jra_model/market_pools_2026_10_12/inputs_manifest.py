# -*- coding: utf-8 -*-
"""入力の目録(2026-10-12 2回目レビュー対応、再現性)。

T1・T2・T3 が読んだ入力ファイルの sha256・行数、対象日の一覧、Python とパッケージの版を記録する。
別のPCで再実行するときは、同じ目録になることを先に確かめる(違えば標本が変わっている)。
出力: data/jra_pipeline/market_pools_2026_10_12/inputs_manifest.json
"""
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy

sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent))
import mp_common as M  # noqa: E402


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def entry(p):
    p = Path(p)
    return {"path": p.relative_to(M.ROOT).as_posix(), "sha256": sha(p), "bytes": p.stat().st_size}


def main():
    files = []
    dates = set()
    for t in ("t1", "t2", "t3"):
        d = pd.read_pickle(M.OUT / f"{t}_rows.pkl")
        dates |= set(d["date"].astype(str))
    for d in sorted(dates):
        for p in (M.ROOT / "data" / "race_results" / d[:4] / f"{d}.csv", M.ROOT / "data" / "intraday_results" / f"{d}.csv",
                  M.ROOT / "data" / "odds_history_fuku" / d[:4] / f"{d}.csv", M.ROOT / "data" / "payouts" / d[:4] / f"{d}.csv",
                  M.tansho_hist_path(d)):
            if p is not None and Path(p).exists():
                files.append(entry(p))
    t3 = pd.read_pickle(M.OUT / "t3_rows.pkl")
    for rid in sorted(t3["race_id"].unique()):
        p = M.ROOT / "data" / "newspaper" / f"{rid}.csv"
        if p.exists():
            files.append(entry(p))
    for name in ("comment_labels.csv", "lambda_2017_2025.json", "rater_agreement.json"):
        files.append(entry(M.OUT / name))
    for p in sorted((M.OUT / "inputs").rglob("*")):
        if p.is_file():
            files.append(entry(p))
    out = {"end_date": M.END_DATE, "dates": sorted(dates), "n_files": len(files),
           "python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scipy": scipy.__version__,
           "note": "盲検の並べ替え(key.csv)と別評価者用の200件の抽出は会話の中で行い、スクリプトは残っていない。結果(key.csv・key200.csv)を正本として inputs/ に保存。",
           "files": files}
    (M.OUT / "inputs_manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(files), "files", len(dates), "dates", out["python"], out["numpy"], out["pandas"], out["scipy"])


if __name__ == "__main__":
    main()
