# -*- coding: utf-8 -*-
"""低回収率台帳(消し材料)5件(単勝10%未満/複勝的中率5%未満/ワイド10%未満/馬連的中率0%/
3連複的中率0%)を、jra_factor_registry.FACTOR_GROUPSの9tier(basic/radar/radar_detail/
condition/course_baba/reference/candidate_v1/candidate_v2/candidate_v3)別カテゴリ構成比で
検証する。あわせて各台帳の「純度」(的中率0.0%のパターンが何%を占めるか)を集計し、
精度アップの余地(0%未満台帳を0%のみへ絞る等)を診断する。

出力: data/jra_pipeline/jra_ledger_low_radar_analysis_2026_09_16_result.json
"""
import json
import pathlib
import sys
from collections import Counter, defaultdict

PROJECT_ROOT = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
LIB_DIR = PROJECT_ROOT / "scripts" / "jra_model"
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))

import jra_factor_registry as FR  # noqa: E402

OUT_PATH = DATA_DIR / "jra_ledger_low_radar_analysis_2026_09_16_result.json"

TIER_META = {
    "basic": "基本", "radar": "詳細7カテゴリ(レーダー)", "radar_detail": "レーダー細分化",
    "condition": "出走条件・体調フィルター", "course_baba": "コース形態・馬場",
    "reference": "参考指標(本番不使用)", "candidate_v1": "新規ファクターv1",
    "candidate_v2": "新規ファクターv2", "candidate_v3": "騎手厩舎馬主生産者v3",
}
TIER_ORDER = ["basic", "radar", "radar_detail", "condition", "course_baba", "reference",
              "candidate_v1", "candidate_v2", "candidate_v3"]

tier_of_group = {gid: g.get("tier", "unknown") for gid, g in FR.FACTOR_GROUPS.items()}
n_groups_per_tier = Counter(tier_of_group.values())

LEDGERS = [
    ("単勝", DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "回収率10%未満"),
    ("複勝", DATA_DIR / "jra_ledger_search_fukusho_hitrate_2026_09_15_result.json", "的中率5%未満"),
    ("ワイド", DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "回収率10%未満"),
    ("馬連", DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "的中率0%(一度も的中なし)"),
    ("3連複", DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "的中率0%(一度も的中なし)"),
]

result = {"tier_meta": TIER_META, "tier_order": TIER_ORDER, "n_groups_per_tier": dict(n_groups_per_tier),
          "ledgers": {}}

for bt, path, criterion in LEDGERS:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["bet_types"][bt]["rows"]
    n_patterns = len(rows)

    # 純度: 的中率0.0%のパターンが何件・何%か
    zero_rows = [r for r in rows if r.get("hit_rate_pct", 0.0) == 0.0]
    hit_rate_vals = [r.get("hit_rate_pct", 0.0) for r in rows]
    nonzero_vals = sorted(v for v in hit_rate_vals if v > 0.0)

    # tier別 atom使用回数(全パターンのselected内の(gid,oid)をtierへマップして集計)
    tier_atom_count = Counter()
    tier_pattern_count = Counter()  # そのtierのatomを1本でも含むパターン数
    depth_counter = Counter()
    for r in rows:
        sel = r["selected"]
        depth_counter[len(sel)] += 1
        tiers_in_row = set()
        for gid in sel:
            t = tier_of_group.get(gid, "unknown")
            tier_atom_count[t] += 1
            tiers_in_row.add(t)
        for t in tiers_in_row:
            tier_pattern_count[t] += 1

    total_atom_uses = sum(tier_atom_count.values())
    tier_composition_pct = {
        t: round(100.0 * tier_atom_count.get(t, 0) / total_atom_uses, 2) if total_atom_uses else 0.0
        for t in TIER_ORDER
    }
    tier_pattern_share_pct = {
        t: round(100.0 * tier_pattern_count.get(t, 0) / n_patterns, 2) if n_patterns else 0.0
        for t in TIER_ORDER
    }

    result["ledgers"][bt] = {
        "criterion": criterion, "n_patterns": n_patterns,
        "n_zero_hit_rate": len(zero_rows),
        "zero_hit_rate_share_pct": round(100.0 * len(zero_rows) / n_patterns, 2) if n_patterns else None,
        "nonzero_hit_rate_min_pct": nonzero_vals[0] if nonzero_vals else None,
        "nonzero_hit_rate_max_pct": nonzero_vals[-1] if nonzero_vals else None,
        "nonzero_hit_rate_median_pct": nonzero_vals[len(nonzero_vals)//2] if nonzero_vals else None,
        "depth_distribution": {str(k): v for k, v in sorted(depth_counter.items())},
        "tier_composition_pct_of_atom_uses": tier_composition_pct,
        "tier_pattern_share_pct": tier_pattern_share_pct,
    }

    print(f"=== {bt}({criterion}) n={n_patterns} ===")
    print(f"  的中率0.0%: {len(zero_rows)}件 ({result['ledgers'][bt]['zero_hit_rate_share_pct']}%)")
    if nonzero_vals:
        print(f"  的中率>0%の範囲: {nonzero_vals[0]:.2f}%〜{nonzero_vals[-1]:.2f}% (中央値{nonzero_vals[len(nonzero_vals)//2]:.2f}%, {len(nonzero_vals)}件)")
    print("  tier構成比(atom使用回数ベース):", {k: v for k, v in tier_composition_pct.items() if v > 0})
    print()

OUT_PATH.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote {OUT_PATH}")
