# -*- coding: utf-8 -*-
"""単勝・ワイドの低回収率台帳(jra_ledger_search_low_2026_09_15_result.json)を、
馬連・3連複と同じ方式(既存の回収率10%未満結果から的中率0.0%の行だけを再フィルタ、
新規探索なし)で「的中率0%(一度も的中なし)のみ」へ絞り込む。複勝セクションは対象外
(既にjra_ledger_search_fukusho_hitrate_2026_09_15_result.jsonへ移行済みのため無変更)。
"""
import json
import pathlib

DATA_DIR = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所\data\jra_pipeline")
PATH = DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json"

data = json.loads(PATH.read_text(encoding="utf-8"))

for bt in ("単勝", "ワイド"):
    sec = data["bet_types"][bt]
    rows = sec["rows"]
    n_before = len(rows)
    zero_rows = [r for r in rows if r.get("hit_rate_pct", 0.0) == 0.0]
    for i, r in enumerate(zero_rows, 1):
        r["rank"] = i
    sec["rows"] = zero_rows
    sec["n_patterns"] = len(zero_rows)
    sec["hit_rate_threshold"] = 0.0
    print(f"{bt}: {n_before} -> {len(zero_rows)}件 (的中率0.0%のみ)")

PATH.write_text(json.dumps(data, ensure_ascii=False, indent=None), encoding="utf-8")
print(f"wrote {PATH}")
