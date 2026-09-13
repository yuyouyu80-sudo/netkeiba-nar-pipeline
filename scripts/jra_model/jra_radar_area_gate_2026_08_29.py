# -*- coding: utf-8 -*-
"""レーダーチャート面積着順予想シグナルの検証ゲート(2専門家レビュー反映版)。

設計はjra_lap33_signal_gate_2026_08_28.pyと同型: 単一の事前指定した比較
(幾何平均 vs 線形平均、box4)をblock_bootstrap_diffのCI下限>0で判定する主検定1本に
一本化し、面積(3順序)・最小値・上位3軸和・頭数サブグループは探索的参考報告として
CI付きで並記する(pass/fail判定は付けない、多重比較補正は不要)。

前提: jra_dataset_cache.pkl が最新(race_pace_label列を含むnewspaper CSV反映後)で
あること。古い場合は `python jra_dataset.py` で再構築してから実行すること。

詳細な設計根拠は計画ファイル参照:
C:\\Users\\yuyou\\.claude\\plans\\valiant-cuddling-aho.md
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_radar_categories as RC
import jra_signals as JS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "jra_radar_area_gate_2026_08_29_result.json"
OUT_TXT = OUT_DIR / "jra_radar_area_gate_2026_08_29_report.txt"

BOX_NS = [4, 5, 3]
COVERAGE_THRESHOLD = 0.5  # 33ラップ・pace_fitの事前登録カバレッジ閾値(lap33ゲートと同じ基準)
FIELD_SIZE_SPLIT = 14  # 事前登録した唯一のサブグループ分割点(競馬予想家指摘)

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ データロード
log("データロード中...")
data = jra_dataset.load(rebuild=False)
races_all, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races_all])
log(f"レース数(除外前): {len(races_all)}  日付: {data['dates'][0]}〜{data['dates'][-1]}"
    f"({len(data['dates'])}日)")

lap33_lookup = L33.load_lap33_lookup()
race_meta = L33.load_race_surface_distance()
history_index = L33.build_history_index()

records_all = RC.build_axis_matrix(races_all, priors_all, history_index, lap33_lookup, race_meta)
races, records = RC.filter_min_categories(races_all, records_all)
log(f"N_race>={RC.MIN_CATEGORIES}で残ったレース数: {len(races)}/{len(races_all)}")

# ============================================================ 無料診断(検定前、コスト0)
log("\n" + "=" * 72)
log("無料診断(検定を回す前に必ず確認する5点)")
log("=" * 72)
diag = RC.run_all_diagnostics(races, records, actual)
log(f"1) 勝ち馬の丸さ(平均値、is_winner別): {diag['winner_roundness_summary']}")
pa = diag["pick_agreement_box4"]
log(f"2) picks一致率(box4): 幾何平均vs線形={pa['geometric_vs_linear_exact_match_rate']:.1%}"
    f"  面積(順序1)vs線形={pa['area_order1_vs_linear_exact_match_rate']:.1%}"
    f"  (対象{pa['n_races']}レース)")
cov = diag["axis_coverage"]
log(f"3) N_raceヒストグラム: {cov['n_race_histogram']}")
log(f"   カテゴリ別脱落理由: {cov['category_drop_counts']}")
log(f"   33ラップ型判定カバレッジ: {cov['lap33_type_known_rate']:.1%}"
    f"  pace_fitラベル取得カバレッジ: {cov['pace_fit_known_rate']:.1%}"
    f"  (事前登録閾値={COVERAGE_THRESHOLD:.0%})")
log(f"4) ブロック数: {diag['n_blocks']}(検出力の目安、少なめである点に留意)")
log(f"5) pace_fit と style(既存pace_pressure項込み)の相関: "
    f"{diag['pace_fit_style_correlation']:+.3f}"
    "(高すぎる場合pace_fit追加は情報を増やしていない可能性)")

# --- 事前登録ルール: カバレッジ<50%のカテゴリは軸から落として再実行する。
dropped_for_coverage = []
if cov["lap33_type_known_rate"] < COVERAGE_THRESHOLD:
    dropped_for_coverage.append("33ラップ理論適合度")
if cov["pace_fit_known_rate"] < COVERAGE_THRESHOLD:
    dropped_for_coverage.append("脚質・展開")  # pace_fit単体を落とすことはできないためカテゴリごと除外
if dropped_for_coverage:
    log(f"\n事前登録ルール発動: カバレッジ閾値未満のため {dropped_for_coverage} を軸から除外して再実行")
    records = RC.drop_categories(records, dropped_for_coverage)
    races, records = RC.filter_min_categories(races, records)
    log(f"除外後、N_race>={RC.MIN_CATEGORIES}で残ったレース数: {len(races)}")
else:
    log("\n事前登録ルール: 両カテゴリともカバレッジ閾値を満たすため8軸のまま実行")

# ============================================================ 主検定(唯一の確証的検定)
log("\n" + "=" * 72)
log("主検定: box4, ev.block_bootstrap_diff(幾何平均picks, 線形平均picks)")
log("gate_pass = bool(diff['lo'] > 0)")
log("=" * 72)

gate_results = {}
exploratory = {}
for box_n in BOX_NS:
    log(f"\n--- box_n={box_n} ---")
    ev = JE.Evaluator(races, actual, box_n=box_n)

    lin = RC.linear_picks(records, box_n)
    geo = RC.geometric_mean_picks(records, box_n)
    mn = RC.min_picks(records, box_n)
    top3 = RC.top3sum_picks(records, box_n)
    area1 = RC.radar_area_picks(records, RC.ORDER_1, box_n)
    area2 = RC.radar_area_picks(records, RC.ORDER_2, box_n)
    area_null = RC.radar_area_picks(records, RC.ORDER_NULL, box_n)
    mkt = JE.market_picks(races, box_n)

    eval_lin = ev.evaluate(lin)
    eval_geo = ev.evaluate(geo)
    diff = ev.block_bootstrap_diff(geo, lin)
    gate_pass = bool(diff["lo"] > 0)
    log(f"  線形平均: model={eval_lin['model']:.2f}% market={eval_lin['market']:.2f}% "
        f"excess={eval_lin['excess']:+.2f}pt")
    log(f"  幾何平均: model={eval_geo['model']:.2f}% excess={eval_geo['excess']:+.2f}pt "
        f"(線形比{eval_geo['excess'] - eval_lin['excess']:+.2f}pt)")
    log(f"  ペア差分ブートストラップ95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}]pt  "
        f"G1={'PASS' if gate_pass else 'NO'}")
    gate_results[f"box{box_n}"] = {
        "linear_excess_pt": eval_lin["excess"], "geometric_excess_pt": eval_geo["excess"],
        "excess_diff_pt": eval_geo["excess"] - eval_lin["excess"],
        "paired_ci_lo": diff["lo"], "paired_ci_hi": diff["hi"], "paired_ci_mean": diff["mean"],
        "gate_pass": gate_pass,
    }

    # --- 探索的・参考報告(検定ではない、pass/fail判定を付けない)
    log("  [探索的参考情報、pass/fail判定なし]")
    box_exploratory = {}
    for name, picks in (
        ("area_order1", area1), ("area_order2", area2), ("area_order_null", area_null),
        ("min", mn), ("top3sum", top3),
    ):
        ev_p = ev.evaluate(picks)
        diff_vs_lin = ev.block_bootstrap_diff(picks, lin)
        diff_vs_mkt = ev.block_bootstrap_diff(picks, [np.array(m) for m in mkt]) if name.startswith("area") else None
        log(f"    {name}: excess={ev_p['excess']:+.2f}pt  線形比95%CI=[{diff_vs_lin['lo']:+.2f},"
            f"{diff_vs_lin['hi']:+.2f}]")
        box_exploratory[name] = {
            "excess_pt": ev_p["excess"],
            "vs_linear_ci_lo": diff_vs_lin["lo"], "vs_linear_ci_hi": diff_vs_lin["hi"],
        }
        if diff_vs_mkt is not None:
            log(f"      市場比95%CI=[{diff_vs_mkt['lo']:+.2f},{diff_vs_mkt['hi']:+.2f}]")
            box_exploratory[name]["vs_market_ci_lo"] = diff_vs_mkt["lo"]
            box_exploratory[name]["vs_market_ci_hi"] = diff_vs_mkt["hi"]

    # --- 面積の順序間相互比較(順序感度の例示、CI付きで並記するのみ)
    order_pairs = [("order1_vs_order2", area1, area2), ("order1_vs_null", area1, area_null),
                  ("order2_vs_null", area2, area_null)]
    order_sensitivity = {}
    for pair_name, a, b in order_pairs:
        d = ev.block_bootstrap_diff(a, b)
        log(f"    順序感度 {pair_name}: 95%CI=[{d['lo']:+.2f},{d['hi']:+.2f}]")
        order_sensitivity[pair_name] = {"ci_lo": d["lo"], "ci_hi": d["hi"]}
    box_exploratory["order_sensitivity"] = order_sensitivity

    # --- 頭数サブグループ(事前登録した唯一のサブグループ、競馬予想家指摘)
    field_sizes = np.array([len(r["df"]) for r in races])
    idx_large = np.where(field_sizes >= FIELD_SIZE_SPLIT)[0]
    idx_small = np.where(field_sizes < FIELD_SIZE_SPLIT)[0]
    sub_large = ev.evaluate(geo, idx=idx_large)
    sub_small = ev.evaluate(geo, idx=idx_small)
    log(f"    頭数サブグループ(幾何平均、{FIELD_SIZE_SPLIT}頭以上 n={len(idx_large)}): "
        f"excess={sub_large['excess']:+.2f}pt")
    log(f"    頭数サブグループ(幾何平均、{FIELD_SIZE_SPLIT}頭未満 n={len(idx_small)}): "
        f"excess={sub_small['excess']:+.2f}pt")
    box_exploratory["field_size_subgroup"] = {
        "large_n": int(len(idx_large)), "large_excess_pt": sub_large["excess"],
        "small_n": int(len(idx_small)), "small_excess_pt": sub_small["excess"],
    }

    exploratory[f"box{box_n}"] = box_exploratory

# ============================================================ まとめ
log("\n" + "=" * 72)
log("まとめ(box4主指標、G1=幾何平均vs線形平均のペア差分ブートストラップ95%CI下限>0)")
log("=" * 72)
g4 = gate_results["box4"]
consistent = all(gate_results[f"box{b}"]["gate_pass"] == g4["gate_pass"] for b in BOX_NS)
verdict = "採用検討" if g4["gate_pass"] else "不採用"
log(f"box4 線形比{g4['excess_diff_pt']:+.2f}pt  95%CI=[{g4['paired_ci_lo']:+.2f},"
    f"{g4['paired_ci_hi']:+.2f}]  G1={'PASS' if g4['gate_pass'] else 'NO'}  "
    f"(box5/3一貫: {consistent})")
log(f"判定: {verdict}")
log("\n(探索的参考報告は採否判断には使わない。上記まとめ・主検定のみが確証的結論)")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "n_races_before_filter": len(races_all),
    "dropped_for_coverage": dropped_for_coverage,
    "diagnostics": diag, "gate_results": gate_results, "exploratory": exploratory,
    "verdict": verdict, "box5_3_consistent": consistent,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
