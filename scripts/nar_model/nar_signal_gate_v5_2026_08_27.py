# -*- coding: utf-8 -*-
"""Phase0診断 + Phase1個別シグナルゲート(2026-08-27、全ファクター統合・高確信度選抜計画)。

既存 nar_signals.py / nar_eval.py / nar_dataset.py はすべて無改造で参照するのみ(2026-08-27の
Phase0で追加した後方互換メソッド・シグナルを使うが、既存メソッド・シグナルの挙動には触れない)。

対象:
  トラックA(V4、5本): hold_just/hold_wide/jockey_change/class_drop/weight_trend。
    個別ゲートを経ずに2026-08-20の500パターン一括探索(measurement artifactで全box不採用)に
    投入されたため、今も個別の真の価値が未検証。
  トラックB(V5、予想印・展開、Phase0で11本新設): Spearman相関スクリーニングで
    ~8本に絞ってからPhase1に進む(多重比較を過大にしないため)。

Phase1の設計(計画書からの具体化、docstring内で理由を明記):
  各候補cについて、baseline(POOL_TRUE_PROD=17本、winner_box*_nar.jsonの現行本番と同一構成の
  等重み)と candidate(baseline+c、18本の等重み)を比較する。これは「多数の重みパターンから
  最良を選ぶ」探索ではなく単一の事前指定した比較(cを均等な1票として追加したら変わるか)
  なので、selection_optimism/true_edge_pt(選択バイアス補正用)は本質的に不要と判断し、
  paired_block_bootstrap(同一レース上の対比較、天候等の共通ノイズを相殺)のCI下限>0を
  ゲートG1とする(計画書のG1の2条件のうち後者を単独採用、前者は複数パターンから選ぶ探索
  向けのため今回の設計には直接適用できない — この設計判断はPhase4の統計専門家レビューで
  検証対象とする)。box_n=4を主指標、5/3で一貫性確認。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nar_dataset
import nar_eval as NE
import nar_signals as NS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "nar_signal_gate_v5_2026_08_27_result.json"
OUT_TXT = OUT_DIR / "nar_signal_gate_v5_2026_08_27_report.txt"

BOX_NS = [4, 5, 3]
CORR_DROP_THRESHOLD = 0.70  # V5内部スクリーニング用(高相関ペアの片方を落とす閾値)
G1_CI_METRIC = "excess"  # 市場超過ptの差分で判定

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ データロード
data = nar_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}")

# ============================================================ Phase0(3): V5充足率・NaN安全性
log("\n" + "=" * 72)
log("Phase0: V5(予想印・展開)候補の充足率診断")
log("=" * 72)
mats_v5_all = NS.signal_matrices(races, priors_all, NS.CANDIDATE_SIGNALS_V5)
A_v5 = np.vstack([m["A"] for m in mats_v5_all])
fill_rates = {}
for j, n in enumerate(NS.CANDIDATE_SIGNALS_V5):
    fill = float(A_v5[:, j].mean())
    fill_rates[n] = fill
    log(f"  {n}: 充足率{fill:.1%}")

# hi==lo NaN安全性の実データ確認: mark_honshi列が丸ごと欠測(または全行空文字)のレースを
# 2-3件抽出し、mark_honshi_scoreが実際に全NaN化されることを確認する。
log("\n--- hi==lo NaN安全性の実データ確認(mark_honshi欠測レース) ---")
missing_mark_races = []
for i, r in enumerate(races):
    # 2026-08-28修正(レビュー1指摘): pandas 3.0系のArrow裏付け文字列dtypeでは、NaNセルへの
    # .astype(str)が文字列"nan"にならず.isin(["","nan","None"])が常にFalseを返す(実測確認
    # 済み)。旧実装は欠損レースを常に0件と誤検出していた(=fill_rate=0%という別経路の
    # 集計は正しかったが、このNaN安全性の実データ確認自体は機能していなかった)。
    # .isna()(空文字はまた別に判定)で修正する。
    col = r["df"]["mark_honshi"] if "mark_honshi" in r["df"].columns else None
    if col is None or (col.isna() | (col.astype(str) == "")).all():
        missing_mark_races.append(i)
missing_check = []
mh_idx = NS.CANDIDATE_SIGNALS_V5.index("mark_honshi_score")
for i in missing_mark_races[:3]:
    all_nan = bool(np.isnan(mats_v5_all[i]["S"][:, mh_idx]).all() and
                   (mats_v5_all[i]["A"][:, mh_idx] == 0).all())
    missing_check.append({"race_id": races[i]["race_id"], "confirmed_all_nan": all_nan})
    log(f"  race_id={races[i]['race_id']}: mark_honshi欠測 -> mark_honshi_score全NaN化: {all_nan}")
log(f"  mark_honshi欠測レース総数: {len(missing_mark_races)}/{len(races)}")

# ============================================================ Phase0: V5相関スクリーニング
log("\n" + "=" * 72)
log("Phase0: V5候補間のSpearman相関スクリーニング(多重比較対策、~8本に絞る)")
log("=" * 72)
S_v5 = np.vstack([m["S"] for m in mats_v5_all])
S_v5_masked = np.where(A_v5 > 0, S_v5, np.nan)
df_v5 = pd.DataFrame(S_v5_masked, columns=NS.CANDIDATE_SIGNALS_V5)
corr_v5 = df_v5.corr(method="spearman")
log(corr_v5.round(3).to_string())

active_v5 = list(NS.CANDIDATE_SIGNALS_V5)
dropped_v5 = []
pairs = []
for i, a in enumerate(NS.CANDIDATE_SIGNALS_V5):
    for b in NS.CANDIDATE_SIGNALS_V5[i + 1:]:
        pairs.append((abs(corr_v5.loc[a, b]), a, b))
pairs.sort(key=lambda x: -x[0])
for rho, a, b in pairs:
    # 2026-08-28レビュー2指摘: 元は `rho < CORR_DROP_THRESHOLD` だった。rhoがNaN
    # (mark_honshi_scoreのように片方が全欠損で相関が定義不能な場合)だと `NaN < 0.70` は
    # Falseになり、"高相関ペアとして除外処理に入る"という誤った分岐に進んでいた
    # (今回はfill_rateが低い側=mark_honshi_score側が結果的に落ちるため実害はなかったが、
    # 根拠が誤っていた)。`not (rho >= threshold)` はNaNを明示的に「除外判定不能→continue」
    # として扱う。
    if not (rho >= CORR_DROP_THRESHOLD) or a not in active_v5 or b not in active_v5:
        continue
    # 充足率が低い方を落とす。同点ならgap系(近似換算定数に依存)よりposition/rank系を残す。
    if fill_rates[a] != fill_rates[b]:
        drop = a if fill_rates[a] < fill_rates[b] else b
    else:
        drop = a if a.endswith("_gap") else b
    active_v5.remove(drop)
    dropped_v5.append(drop)
    log(f"  |rho|={rho:.3f} {a} vs {b} -> 除外: {drop}")
log(f"\nスクリーニング後トラックB候補({len(active_v5)}本): {active_v5}")
log(f"除外({len(dropped_v5)}本、高相関のため): {dropped_v5}")

# ============================================================ Phase0: 死にシグナル確認(V4+V5)
log("\n" + "=" * 72)
log("Phase0: V4+V5(スクリーニング後)の死にシグナル確認")
log("=" * 72)
dead_v4v5 = NS.detect_dead(races, priors_all, names=NS.CANDIDATE_SIGNALS_V4 + active_v5)
log(f"死にシグナル: {dead_v4v5}")
track_a = [n for n in NS.CANDIDATE_SIGNALS_V4 if n not in dead_v4v5]
track_b = [n for n in active_v5 if n not in dead_v4v5]
log(f"トラックA(V4、生存{len(track_a)}本): {track_a}")
log(f"トラックB(V5、生存{len(track_b)}本): {track_b}")

# ============================================================ Phase1: 個別シグナルゲート
log("\n" + "=" * 72)
log("Phase1: 個別シグナルゲート(baseline=POOL_TRUE_PROD 17本 vs baseline+候補1本)")
log("=" * 72)

dead_base = NS.detect_dead(races, priors_all, names=NS.LEGACY_SIGNALS + NS.NEW_SIGNALS)
alive_base = [n for n in NS.LEGACY_SIGNALS + NS.NEW_SIGNALS if n not in dead_base]
v2_alive = [n for n in NS.CANDIDATE_SIGNALS_V2
            if n not in NS.detect_dead(races, priors_all, names=NS.CANDIDATE_SIGNALS_V2)]
POOL_TRUE_PROD = alive_base + v2_alive
log(f"POOL_TRUE_PROD(現行本番と同一構成、{len(POOL_TRUE_PROD)}本): {POOL_TRUE_PROD}")

candidates = track_a + track_b
NAMES = NS.ALL_SIGNALS_V5  # V5込みの全名前空間(mats計算の列順、既存ALL_SIGNALSは不使用=無改造)
mats_all = NS.signal_matrices(races, priors_all, NAMES)


def equal_w(names_subset) -> np.ndarray:
    d = {n: 1.0 / len(names_subset) for n in names_subset}
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


gate_results = {}
for box_n in BOX_NS:
    log(f"\n--- box_n={box_n} ---")
    ev = NE.Evaluator(races, actual, box_n=box_n)
    w_base = equal_w(POOL_TRUE_PROD)
    picks_base = NE.score_picks(mats_all, w_base, box_n)
    eval_base = ev.evaluate(picks_base)
    log(f"  baseline({len(POOL_TRUE_PROD)}本等重み): model={eval_base['model']:.2f}% "
        f"market={eval_base['market']:.2f}% excess={eval_base['excess']:+.2f}pt")
    for c in candidates:
        w_cand = equal_w(POOL_TRUE_PROD + [c])
        picks_cand = NE.score_picks(mats_all, w_cand, box_n)
        eval_cand = ev.evaluate(picks_cand)
        diff = ev.paired_block_bootstrap(picks_cand, picks_base)
        gate_pass = diff["lo"] > 0
        excess_diff_pt = eval_cand["excess"] - eval_base["excess"]
        log(f"  [{c}] excess={eval_cand['excess']:+.2f}pt (基準比{excess_diff_pt:+.2f}pt) "
            f"ペア差分95%CI=[{diff['lo']:+.2f},{diff['hi']:+.2f}] G1={'PASS' if gate_pass else 'NO'}")
        gate_results.setdefault(c, {})[f"box{box_n}"] = {
            "excess_pt": eval_cand["excess"], "baseline_excess_pt": eval_base["excess"],
            "excess_diff_pt": excess_diff_pt, "paired_ci_lo": diff["lo"], "paired_ci_hi": diff["hi"],
            "paired_ci_mean": diff["mean"], "gate_pass": gate_pass,
        }

log("\n" + "=" * 72)
log("Phase1 まとめ(box_n=4主指標、G1=ペア差分ブートストラップ95%CI下限>0)")
log("=" * 72)
for c in candidates:
    g4 = gate_results[c]["box4"]
    consistent = all(gate_results[c][f"box{b}"]["gate_pass"] == g4["gate_pass"] for b in BOX_NS)
    track = "A(V4)" if c in track_a else "B(V5)"
    log(f"  [{track}] {c}: box4基準比{g4['excess_diff_pt']:+.2f}pt "
        f"G1={'PASS' if g4['gate_pass'] else 'NO'} (box5/3一貫: {consistent})")

# ============================================================ 保存
OUT_DIR.mkdir(parents=True, exist_ok=True)
result = {
    "n_races": len(races), "dates": data["dates"],
    "fill_rates_v5": fill_rates,
    "missing_mark_nan_check": missing_check,
    "corr_v5": corr_v5.round(4).to_dict(),
    "dropped_v5_by_correlation": dropped_v5,
    "dead_v4v5": dead_v4v5,
    "track_a": track_a, "track_b": track_b,
    "pool_true_prod": POOL_TRUE_PROD,
    "gate_results": gate_results,
}
OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
