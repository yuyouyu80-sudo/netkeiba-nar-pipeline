# -*- coding: utf-8 -*-
"""score_rank_box4/box3(循環参照atom)除外後に再探索した単勝・ワイド・馬連・3連複の各結果を、
的中率0.0%(一度も的中なし)のみへ再フィルタする(複勝は的中率0.0%パターンが0件のため対象外、
既存の的中率5%未満基準242件のまま)。
"""
import json
import pathlib

DATA_DIR = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所\data\jra_pipeline")

TARGETS = [
    (DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "単勝"),
    (DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "ワイド"),
    (DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "馬連"),
    (DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "3連複"),
]

cache = {}
for path, bt in TARGETS:
    if path not in cache:
        cache[path] = json.loads(path.read_text(encoding="utf-8"))
    data = cache[path]
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

for path, data in cache.items():
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {path}")
