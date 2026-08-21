# -*- coding: utf-8 -*-
"""シグナル別「勝率最大%」の個別探索→合成パターンの回収率検証 + 最下位選択の逆張りチェック
(2026-08-22新設)。

**位置づけ: 探索的診断。in-sample最適化+Nested LOBO OOF併記。本番非採用。**
これまでのbox5/4/3重み探索(jra_search500_2026_08_21_v3signals.py等)は「全10〜30シグナルを
同時にDirichlet探索し、Nested LOBO OOF+選択バイアス診断で採否ゲート(true_edge/sd>=2.0)を
通過するか」という枠組みだった。今回はユーザーの明示依頼により、それとは異なる手法――
「シグナルを1本ずつ個別に(現行重みを基準点として)勝率(単勝的中率)最大化スイープし、
10個の個別最適値をそのまま1つのパターンに組み上げる」――を実施する。この方式はシグナル間の
交互作用を無視するため、全シグナル同時探索より過学習リスクが高いことをユーザー自身が
理解した上で選択している(2026-08-21のAskUserQuestionで明示選択)。

統計専門家ペルソナのサブエージェントによる手法レビュー(2026-08-21実施)を反映し、以下の
安全策を組み込む:
  A. グリッド設計: 現行重み基準点を明示的にグリッドへ挿入、タイブレークは基準点に最も近い
     tを採用(縮小推定の思想)、プラトー幅を診断値として記録。
  B. 目的関数整合性ゲート: box_n=1で単勝的中率を最大化する「Win-Pickプロファイル」と、
     実際のbox_n(5/4/3)で「勝ち馬がBOXに入っているか」を最大化する「Box-Hitプロファイル」の
     コサイン類似度が0.9以上なら両者は実質同じとしてWin-Pickを採用、0.9未満なら実際に賭ける
     目的に合わせてBox-Hitを採用する(jra_ensemble.pyの多様性判定閾値0.9をそのまま転用)。
  C. 「10本スイープ→積む→ゲートで採否」という手続き全体をfit_fn(train_idx)->(w,pattern_idx)
     としてjra_eval.Evaluator.lobo_oofにそのまま渡し、Nested LOBO OOFで退化・楽観バイアスを
     検知する(既存500パターン探索・アンサンブル探索と全く同じ契約)。
  D. 採用パターン・現行モデル・等重み(10本均等)をjra_eval.selection_optimismのW_small列に
     並べ、Nested LOBO OOFのheld-out picksに対してblock_bootstrap_diffで市場・現行・等重みとの
     差分95%CIを算出する。
  E. 積み上げ健全性チェック(合成後の勝率 vs 単体改善の合計)・1回リファイン・box5/4/3間の
     t_i*シフトのスピアマン相関を診断値として記録する。
  F. 最下位選択の逆張りチェック: 有効スコア馬のみから下位box_n頭を選び(組合せキー不足の
     レースは除外してサブセットEvaluatorで評価)、(a)同一パターンの上位選択 (b)市場人気上位
     (c)市場人気下位、の3系統とblock_bootstrap_diffで比較する(9件比較・多重検定補正なしの
     記述的95%CIである旨を明記)。

本番ファイル(winner_v3.json/winner_box4.json/winner_box3.json)は一切変更しない。
既存の公開レポート(prediction_report_jra_axis.html)にも追記しない。
"""
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_backtest as JB  # noqa: E402
import jra_dataset  # noqa: E402
import jra_ensemble as JEN  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402

BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
NAMES = JS.LEGACY_SIGNALS  # 現行本番10シグナル(speed/form/style/jt/waku/apt/train/distance/sire/bms)
COS_GATE = 0.9
GRID_STEP = 0.01
GRID_STEP_COARSE = 0.05
N_REP_SELECTION_OPTIMISM = 200
WIN_BET_COL = JB.BET_TYPES.index("単勝")
OUT_JSON = DATA_DIR / "jra_signal_univariate_search_2026_08_22_result.json"

data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
N_RACES = len(races)
print(f"races={N_RACES}  signals={NAMES}")


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


# --------------------------------------------------------------------------- A. グリッド
def others_ratio(baseline_w: np.ndarray, i: int) -> np.ndarray:
    r = baseline_w.copy()
    r[i] = 0.0
    s = r.sum()
    return r / s if s > 0 else np.full_like(r, 1.0 / (len(r) - 1))


def sweep_weight(baseline_w: np.ndarray, i: int, t: float) -> np.ndarray:
    r = others_ratio(baseline_w, i)
    w = (1.0 - t) * r
    w[i] = t
    return w


def make_grid(baseline_w: np.ndarray, i: int, step: float) -> np.ndarray:
    base_grid = np.round(np.arange(0.0, 1.0 + 1e-9, step), 6)
    return np.unique(np.round(np.append(base_grid, baseline_w[i]), 6))


def pick_t_star(t_grid: np.ndarray, vals: np.ndarray, base_t: float, atol: float = 1e-9):
    max_v = vals.max()
    tie_idx = np.where(np.isclose(vals, max_v, atol=atol))[0]
    tie_t = t_grid[tie_idx]
    best_local = int(np.argmin(np.abs(tie_t - base_t)))
    best = int(tie_idx[best_local])
    return float(t_grid[best]), float(max_v), int(len(tie_idx))


# --------------------------------------------------------------------------- 勝率/BOX的中率
# 性能上の理由(2026-08-22実測: 素朴にJE.score_picks+BoxSettler.returns_forをグリッド点ごとに
# 呼ぶ実装は36fold×グリッド101点×シグナル10×プロファイル2種×box_n3種で20分超かかり実用外)、
# 実際の勝ち馬の行indexさえ分かればBoxSettlerの全payout計算を経由せずに「勝ち馬がbox_n_for_score
# 頭中に入っているか」を判定できることを利用し、グリッド全点をnumpyブロードキャストで一括評価する
# (計画書「実装上の留意点(パフォーマンス)」節の設計をそのまま実装)。最終的な合成パターンの
# 回収率テーブル・ブートストラップ等、金額を伴う評価は引き続き通常通りEvaluator/BoxSettlerを使う。
def extract_winner_idx(ev1: JE.Evaluator) -> list:
    """box_n=1のEvaluatorのsettler.tablesから、各レースの実際の勝ち馬の行indexを抽出する。
    (単勝の払戻>0になる行を勝ち馬とみなす。既にテスト済みのBoxSettlerの決済ロジックをそのまま
    再利用し、独自の的中判定を新規に書かない)。複数該当(同着)は最初の1頭、該当なし
    (結果データ欠損)はNone。"""
    out = []
    for tbl in ev1.settler.tables:
        winners = [key[0] for key, (st, rt) in tbl.items() if rt[WIN_BET_COL] > 0]
        out.append(winners[0] if winners else None)
    return out


def hit_rate_fast(w: np.ndarray, box_n_for_score: int, winner_idx_list: list, idx=None) -> float:
    """実際の勝ち馬がbox_n_for_score頭中に入っているか(単勝的中/BOX的中と数学的に同値)の%。
    1点だけの評価(グリッド一括ではない)。"""
    race_range = range(N_RACES) if idx is None else idx
    hit, n_eval = 0, 0
    for ridx in race_range:
        w_idx = winner_idx_list[ridx]
        if w_idx is None:
            continue
        m = mats_all[ridx]
        num, den = m["S"] @ w, m["A"] @ w
        valid = den > 0
        score = np.where(valid, num / den, -1e18)
        n = len(score)
        k = min(box_n_for_score, n)
        rank = int(((score > score[w_idx]) & valid).sum())
        hit += 1 if rank < k else 0
        n_eval += 1
    return (hit / n_eval * 100) if n_eval else 0.0


def sweep_curve_fast(baseline_w: np.ndarray, i: int, box_n_for_score: int, t_grid: np.ndarray,
                     winner_idx_list: list, idx=None) -> np.ndarray:
    """シグナルiのグリッド全点(t_grid)の勝率/BOX的中率を、レースループ1回でまとめて計算する
    (グリッド軸はnumpyブロードキャストで処理し、グリッド点ごとのPython関数呼び出しを避ける)。
    w(t) = t*e_i + (1-t)*r_other が t の1次式であることを利用し、num(t)/den(t)も1次式で表す。"""
    r_other = others_ratio(baseline_w, i)
    race_range = range(N_RACES) if idx is None else idx
    G = len(t_grid)
    hits = np.zeros(G)
    n_eval = 0
    for ridx in race_range:
        w_idx = winner_idx_list[ridx]
        if w_idx is None:
            continue
        m = mats_all[ridx]
        S, A = m["S"], m["A"]
        n = S.shape[0]
        base_num, base_den = S @ r_other, A @ r_other
        sig_num, sig_den = S[:, i], A[:, i]
        num = base_num[:, None] + t_grid[None, :] * (sig_num - base_num)[:, None]
        den = base_den[:, None] + t_grid[None, :] * (sig_den - base_den)[:, None]
        valid = den > 0
        score = np.where(valid, num / den, -1e18)
        winner_score = score[w_idx, :]
        k = min(box_n_for_score, n)
        rank = ((score > winner_score[None, :]) & valid).sum(axis=0)
        hits += (rank < k)
        n_eval += 1
    return (hits / n_eval * 100) if n_eval else np.zeros(G)


def build_profile(baseline_w: np.ndarray, box_n_for_score: int, winner_idx_list: list,
                  step=GRID_STEP, idx=None):
    """10シグナル分の個別最適化(t_i*)を求める。"""
    n_sig = len(NAMES)
    t_star = np.empty(n_sig)
    peak = np.empty(n_sig)
    plateau = np.empty(n_sig, dtype=int)
    for i in range(n_sig):
        grid = make_grid(baseline_w, i, step)
        vals = sweep_curve_fast(baseline_w, i, box_n_for_score, grid, winner_idx_list, idx=idx)
        t_star[i], peak[i], plateau[i] = pick_t_star(grid, vals, baseline_w[i])
    total = t_star.sum()
    composite = t_star / total if total > 0 else np.full(n_sig, 1.0 / n_sig)
    return composite, t_star, peak, plateau


# --------------------------------------------------------------------------- F. 最下位選択
def worst_picks(w: np.ndarray, box_n: int):
    """有効スコア馬の中から下位box_n頭を選ぶ。組合せキー不足(有効馬<k=min(box_n,n))の
    レースはNoneを返す(呼び出し側で除外・サブセットEvaluator再構築)。"""
    picks = []
    for m, r in zip(mats_all, races):
        n = len(r["df"])
        k = min(box_n, n)
        den = m["A"] @ w
        valid = np.where(den > 0)[0]
        if len(valid) < k:
            picks.append(None)
            continue
        score = (m["S"][valid] @ w) / den[valid]
        order = valid[np.argsort(score, kind="stable")]  # 昇順=下位から
        picks.append(order[:k])
    return picks


def market_worst_picks(box_n: int):
    picks = []
    for r in races:
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        key = np.where(np.isnan(ninki), -1e18, ninki)  # 欠損は「最下位候補」から除外側に扱う
        picks.append(np.argsort(-key, kind="stable")[:box_n])
    return picks


# --------------------------------------------------------------------------- メイン
results_by_box = {}
t_star_wp_by_box = {}  # box5/4/3間の整合性診断用(Win-Pickプロファイルのt_i*)

for BOX_N in BOX_NS:
    t0 = time.time()
    print(f"\n{'=' * 70}\nbox_n={BOX_N}\n{'=' * 70}")
    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    baseline_w = wvec(winner["weights"])
    baseline_w = baseline_w / baseline_w.sum()

    ev1 = JE.Evaluator(races, actual, box_n=1)
    ev_box = JE.Evaluator(races, actual, box_n=BOX_N)
    winner_idx_list = extract_winner_idx(ev1)

    # --- B. Win-PickプロファイルとBox-Hitプロファイル(全211レース、フルグリッド)
    W_wp, t_wp, peak_wp, plat_wp = build_profile(baseline_w, 1, winner_idx_list, step=GRID_STEP)
    W_bh, t_bh, peak_bh, plat_bh = build_profile(baseline_w, BOX_N, winner_idx_list, step=GRID_STEP)
    cos_wp_bh = JEN.cosine_similarity(W_wp, W_bh)
    adopted_profile = "win_pick" if cos_wp_bh >= COS_GATE else "box_hit"
    W_adopted = W_wp if adopted_profile == "win_pick" else W_bh
    print(f"  Win-Pick t*: {dict(zip(NAMES, np.round(t_wp, 3)))}")
    print(f"  Box-Hit  t*: {dict(zip(NAMES, np.round(t_bh, 3)))}")
    print(f"  cos(Win-Pick, Box-Hit)={cos_wp_bh:.4f} → 採用プロファイル: {adopted_profile}")

    # --- A. グリッド解像度感度チェック(粗グリッド0.05刻みで同じ手続きを再実行)
    W_wp_c, t_wp_c, _, _ = build_profile(baseline_w, 1, winner_idx_list, step=GRID_STEP_COARSE)
    W_bh_c, t_bh_c, _, _ = build_profile(baseline_w, BOX_N, winner_idx_list, step=GRID_STEP_COARSE)
    grid_sensitivity_wp = float(np.max(np.abs(t_wp - t_wp_c)))
    grid_sensitivity_bh = float(np.max(np.abs(t_bh - t_bh_c)))
    print(f"  グリッド感度(0.01 vs 0.05の最大差): Win-Pick={grid_sensitivity_wp:.3f}  "
          f"Box-Hit={grid_sensitivity_bh:.3f}")

    # --- E-1. 積み上げ健全性チェック(合成後の勝率 vs 単体改善の合計)
    baseline_winrate = hit_rate_fast(baseline_w, 1, winner_idx_list)
    composite_winrate_wp = hit_rate_fast(W_wp, 1, winner_idx_list)
    sum_of_individual_gains = baseline_winrate + float(np.sum(peak_wp - baseline_winrate))
    print(f"  勝率: 現行={baseline_winrate:.2f}%  合成(Win-Pick)={composite_winrate_wp:.2f}%  "
          f"単体改善の相加的合計(上限参考値)={sum_of_individual_gains:.2f}%")

    # --- E-2. 1回リファイン(採用プロファイルの合成パターンを新しい基準点として再スイープ)
    if adopted_profile == "win_pick":
        W_refined, t_refined, _, _ = build_profile(W_adopted, 1, winner_idx_list, step=GRID_STEP)
        refine_shift = float(np.max(np.abs(t_refined - t_wp)))
    else:
        W_refined, t_refined, _, _ = build_profile(W_adopted, BOX_N, winner_idx_list, step=GRID_STEP)
        refine_shift = float(np.max(np.abs(t_refined - t_bh)))
    print(f"  1回リファイン後のt*最大変化量: {refine_shift:.3f}")

    t_star_wp_by_box[BOX_N] = t_wp

    # --- C. 手続き全体をNested LOBO OOF化(fit_fnはB(A)の手続きをtrain_idxに限定して再実行)
    def fit_fn(train_idx, _box_n=BOX_N, _baseline_w=baseline_w, _winner_idx_list=winner_idx_list):
        w_wp, t_wp_f, _, _ = build_profile(_baseline_w, 1, _winner_idx_list, step=GRID_STEP, idx=train_idx)
        w_bh, t_bh_f, _, _ = build_profile(_baseline_w, _box_n, _winner_idx_list, step=GRID_STEP,
                                           idx=train_idx)
        cos = JEN.cosine_similarity(w_wp, w_bh)
        if cos >= COS_GATE:
            return w_wp, ("wp",) + tuple(np.round(t_wp_f, 2))
        return w_bh, ("bh",) + tuple(np.round(t_bh_f, 2))

    nested_oof = ev_box.lobo_oof(fit_fn, mats_all)
    in_sample_eval = ev_box.evaluate(JE.score_picks(mats_all, W_adopted, BOX_N))
    print(f"  Nested LOBO OOF: n_unique_patterns={nested_oof['n_unique_patterns']}/"
          f"{nested_oof['n_folds']}  市場超過(OOF)={nested_oof['excess']:+.2f}pt  "
          f"市場超過(in-sample)={in_sample_eval['excess']:+.2f}pt  "
          f"楽観バイアス={in_sample_eval['excess'] - nested_oof['excess']:+.2f}pt")

    # --- D. 選択バイアス診断 + ブートストラップ(Nested LOBO OOFのheld-out picksに対して)
    W_EQUAL10 = np.full(len(NAMES), 1.0 / len(NAMES))
    current_picks = JE.score_picks(mats_all, baseline_w, BOX_N)
    equal10_picks = JE.score_picks(mats_all, W_EQUAL10, BOX_N)
    market_picks_top = JE.market_picks(races, BOX_N)

    W_small = np.column_stack([W_adopted, baseline_w, W_EQUAL10])
    opt = JE.selection_optimism(ev_box, mats_all, W_small, n_rep=N_REP_SELECTION_OPTIMISM, seed=8100 + BOX_N)
    edge_ratio = opt["true_edge_pt"] / opt["true_edge_sd"] if opt["true_edge_sd"] else 0.0
    print(f"  選択バイアス診断(参考値): true_edge={opt['true_edge_pt']:+.2f}pt "
          f"(sd={opt['true_edge_sd']:.2f}, 比={edge_ratio:+.3f})")

    boot_vs_market = ev_box.block_bootstrap_diff(nested_oof["picks"], market_picks_top, n=2000, seed=8200 + BOX_N)
    boot_vs_current = ev_box.block_bootstrap_diff(nested_oof["picks"], current_picks, n=2000, seed=8300 + BOX_N)
    boot_vs_equal10 = ev_box.block_bootstrap_diff(nested_oof["picks"], equal10_picks, n=2000, seed=8400 + BOX_N)
    print(f"  OOF vs 市場: 差分95%CI=[{boot_vs_market['lo']:+.2f}, {boot_vs_market['hi']:+.2f}]pt")
    print(f"  OOF vs 現行: 差分95%CI=[{boot_vs_current['lo']:+.2f}, {boot_vs_current['hi']:+.2f}]pt")
    print(f"  OOF vs 等重み: 差分95%CI=[{boot_vs_equal10['lo']:+.2f}, {boot_vs_equal10['hi']:+.2f}]pt")

    # --- 合成パターン(in-sample)の全券種回収率テーブル
    full_tbl_composite = ev_box.full_table(JE.score_picks(mats_all, W_adopted, BOX_N))

    # --- F. 最下位選択の逆張りチェック
    picks_worst_raw = worst_picks(W_adopted, BOX_N)
    excluded_idx = [i for i, p in enumerate(picks_worst_raw) if p is None]
    n_excluded = len(excluded_idx)
    if n_excluded:
        kept_idx = [i for i in range(N_RACES) if i not in excluded_idx]
        races_kept = [races[i] for i in kept_idx]
        ev_box_worst = JE.Evaluator(races_kept, actual, box_n=BOX_N)
        picks_worst = [picks_worst_raw[i] for i in kept_idx]
        picks_top_kept = [JE.score_picks(mats_all, W_adopted, BOX_N)[i] for i in kept_idx]
        picks_mkt_top_kept = [market_picks_top[i] for i in kept_idx]
        picks_mkt_worst_kept = [market_worst_picks(BOX_N)[i] for i in kept_idx]
    else:
        ev_box_worst = ev_box
        picks_worst = picks_worst_raw
        picks_top_kept = JE.score_picks(mats_all, W_adopted, BOX_N)
        picks_mkt_top_kept = market_picks_top
        picks_mkt_worst_kept = market_worst_picks(BOX_N)
    print(f"  最下位選択: 除外レース数={n_excluded}/{N_RACES}")

    diff_worst_vs_top = ev_box_worst.block_bootstrap_diff(picks_worst, picks_top_kept, n=2000, seed=8500 + BOX_N)
    diff_worst_vs_mkt_top = ev_box_worst.block_bootstrap_diff(picks_worst, picks_mkt_top_kept, n=2000,
                                                              seed=8600 + BOX_N)
    diff_worst_vs_mkt_worst = ev_box_worst.block_bootstrap_diff(picks_worst, picks_mkt_worst_kept, n=2000,
                                                                seed=8700 + BOX_N)
    full_tbl_worst = ev_box_worst.full_table(picks_worst)
    print(f"  逆張り(最下位) vs 上位選択: 差分95%CI=[{diff_worst_vs_top['lo']:+.2f}, "
          f"{diff_worst_vs_top['hi']:+.2f}]pt")
    print(f"  逆張り(最下位) vs 市場人気上位: 差分95%CI=[{diff_worst_vs_mkt_top['lo']:+.2f}, "
          f"{diff_worst_vs_mkt_top['hi']:+.2f}]pt")
    print(f"  逆張り(最下位) vs 市場人気下位: 差分95%CI=[{diff_worst_vs_mkt_worst['lo']:+.2f}, "
          f"{diff_worst_vs_mkt_worst['hi']:+.2f}]pt")

    results_by_box[BOX_N] = {
        "model_file": WINNER_FILES[BOX_N],
        "baseline_weights": dict(zip(NAMES, baseline_w.tolist())),
        "signal_profile_table": [
            {"signal": n, "baseline_w": float(baseline_w[i]),
             "t_star_win_pick": float(t_wp[i]), "peak_win_pick_pct": float(peak_wp[i]),
             "plateau_width_win_pick": int(plat_wp[i]),
             "t_star_box_hit": float(t_bh[i]), "peak_box_hit_pct": float(peak_bh[i]),
             "plateau_width_box_hit": int(plat_bh[i])}
            for i, n in enumerate(NAMES)
        ],
        "gate": {"cosine_sim_wp_bh": float(cos_wp_bh), "adopted_profile": adopted_profile,
                 "adopted_weights": dict(zip(NAMES, W_adopted.tolist())),
                 "grid_sensitivity_max_delta_win_pick": grid_sensitivity_wp,
                 "grid_sensitivity_max_delta_box_hit": grid_sensitivity_bh},
        "stacking_sanity_check": {"baseline_winrate_pct": baseline_winrate,
                                  "composite_winrate_win_pick_pct": composite_winrate_wp,
                                  "sum_of_individual_gains_upper_bound_pct": sum_of_individual_gains,
                                  "refine_max_t_shift": refine_shift},
        "nested_lobo_oof": {"model": nested_oof["model"], "market": nested_oof["market"],
                            "excess": nested_oof["excess"],
                            "n_unique_patterns": nested_oof["n_unique_patterns"],
                            "n_folds": nested_oof["n_folds"]},
        "in_sample": {"model": in_sample_eval["model"], "market": in_sample_eval["market"],
                     "excess": in_sample_eval["excess"],
                     "optimism_pt": in_sample_eval["excess"] - nested_oof["excess"]},
        "selection_optimism_3candidates": {k: v for k, v in opt.items()},
        "bootstrap_oof_vs_market": boot_vs_market,
        "bootstrap_oof_vs_current": boot_vs_current,
        "bootstrap_oof_vs_equal10": boot_vs_equal10,
        "full_table_composite_in_sample": full_tbl_composite.to_dict(orient="records"),
        "worst_n_check": {
            "n_excluded_races": n_excluded,
            "diff_worst_vs_top": diff_worst_vs_top,
            "diff_worst_vs_market_top": diff_worst_vs_mkt_top,
            "diff_worst_vs_market_worst": diff_worst_vs_mkt_worst,
            "full_table_model_worst": full_tbl_worst.to_dict(orient="records"),
        },
    }
    print(f"  (box_n={BOX_N} 所要時間: {time.time() - t0:.1f}秒)")

# --- E-3. box5/4/3間の整合性診断(Win-Pickプロファイルのt_i*シフトのスピアマン相関)
shift_df = pd.DataFrame({
    f"box{b}": t_star_wp_by_box[b] - wvec(json.loads((DATA_DIR / WINNER_FILES[b]).read_text(
        encoding="utf-8"))["weights"]) / wvec(json.loads((DATA_DIR / WINNER_FILES[b]).read_text(
            encoding="utf-8"))["weights"]).sum()
    for b in BOX_NS
}, index=NAMES)
cross_box = {}
for a, b in itertools.combinations(BOX_NS, 2):
    cross_box[f"box{a}_vs_box{b}"] = float(shift_df[f"box{a}"].corr(shift_df[f"box{b}"], method="spearman"))
print(f"\nbox5/4/3間のt_i*シフト・スピアマン相関: {cross_box}")

OUT_JSON.write_text(json.dumps({
    "n_races": N_RACES, "names": NAMES, "cos_gate": COS_GATE, "grid_step": GRID_STEP,
    "grid_step_coarse": GRID_STEP_COARSE,
    "note": "探索的診断・in-sample最適化+Nested LOBO OOF併記・本番非採用。"
            "シグナル個別最適化→結合という手法自体が全シグナル同時探索より過学習リスクが高い"
            "ことを前提に、統計専門家レビューを反映した安全策(グリッド設計・目的関数整合性"
            "ゲート・Nested LOBO OOF化・選択バイアス診断・逆張りチェック)を組み込んでいる。",
    "results_by_box": results_by_box,
    "cross_box_consistency_spearman": cross_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {OUT_JSON}")
