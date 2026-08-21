# -*- coding: utf-8 -*-
"""Step1(2026-08-22): 市場アンカー型条件付きロジット + 期待値ベッティングの実装・検証。

2026-08-21のOpus 5サブエージェント調査を受け、「レース内で順位付けして上位N頭を買う」設計
(box5/4/3・軸流し)を離れ、「市場オッズに対する期待値(EV)が閾値を超えた馬だけ単勝/複勝で
賭ける」設計を検証する。既存box探索が非負制約のDirichlet単体上だったため表現できなかった
「市場が過大評価しているシグナルを負の係数でフェードする」戦略を、符号制約なしの条件付き
ロジット(自由パラメータ3個: beta0=市場アンカー, beta1=近走合成, beta2=適性合成)で試す。

採否ゲート(全て満たして初めて「採用候補」、1つでも未達なら不採用として結果を透明に記録):
  1. Nested LOBO OOFの対数尤度が市場のみを上回り、block_bootstrap_diff_nllの95%CI下限>0。
  2. EV選抜picksの単勝回収率、block_bootstrapの95%CI下限>100%。
  3. 閾値グリッドのselection_optimism型診断でtrue_edge_pt/true_edge_sd>=2.0。
  4. chronological_oof(前半→後半・後半→前半の両方向)の符号がNested LOBO OOFと一致。
  5. オッズ分布マッチのランダム選抜2000回との比較でp<0.05。

Opusの予備計算(EV≥1.2かつオッズ≤20)ではゲート2のCI下限が99.0%、ゲート5がp=0.0595と、
いずれも僅かに届いていない。本スクリプトはこの結果を独立に再現・検証する位置づけであり、
同じ結論(ゲート未達・不採用)になったとしても「探索して不採用という結論に達すること自体が
正しい成果」という本プロジェクトの一貫した方針に沿う。

出力: data/jra_pipeline/jra_market_model_search_2026_08_22_result.json
"""
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_dataset  # noqa: E402
import jra_market_model as MM  # noqa: E402
import jra_signals as JS  # noqa: E402
import jra_singles_eval as SE  # noqa: E402

EV_THRESHOLD = MM.DEFAULT_EV_THRESHOLD
ODDS_CAP = MM.DEFAULT_ODDS_CAP
THRESHOLD_GRID = [(t, c) for t in (1.0, 1.1, 1.2, 1.3, 1.4, 1.5) for c in (10.0, 20.0, 30.0, np.inf)]
OUT_JSON = DATA_DIR / "jra_market_model_search_2026_08_22_result.json"

data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
feats = MM.build_composite_features(races, actual, priors_all, JS.CLASS_ORDINAL)
n_valid_winner = sum(1 for f in feats if f["winner_idx"] is not None)
print(f"races={len(races)}  winner判定可能={n_valid_winner}")

ev = SE.Evaluator(races, actual)


def fit_fn_full(train_idx):
    return MM.fit_conditional_logit(feats, idx=train_idx)


def fit_fn_market_only(train_idx):
    return MM.fit_beta0_only(feats, idx=train_idx)


# --- in-sample参考値(全211レースでfit、比較用の参考値として残すのみ、採否判定には使わない)
beta_full_insample = MM.fit_conditional_logit(feats)
beta_market_insample = MM.fit_beta0_only(feats)
nll_full_insample = MM.race_nll(beta_full_insample, feats)
nll_market_insample = MM.race_nll(beta_market_insample, feats)
print(f"\nin-sample(全211レース、参考値):")
print(f"  beta_full = {np.round(beta_full_insample, 4).tolist()}  NLL/race={nll_full_insample:.4f}")
print(f"  beta_market_only = {np.round(beta_market_insample, 4).tolist()}  NLL/race={nll_market_insample:.4f}")

# --- ゲート1: Nested LOBO OOFの対数尤度(モデル vs 市場のみ)
print(f"\n{'=' * 60}\nゲート1: Nested LOBO OOF 対数尤度\n{'=' * 60}")
oof_nll_full = ev.lobo_oof_nll(fit_fn_full, feats)
oof_nll_market = ev.lobo_oof_nll(fit_fn_market_only, feats)
n_unique_full = len(set(oof_nll_full["chosen_params"].values()))
diff_nll = ev.block_bootstrap_diff_nll(oof_nll_full["nll_per_race"], oof_nll_market["nll_per_race"],
                                       n=2000, seed=9201)
gate1_pass = diff_nll["lo"] > 0
print(f"  fold毎パラメータのユニーク数: {n_unique_full}/36")
print(f"  OOF平均NLL: モデル={oof_nll_full['mean_nll']:.4f}  市場のみ={oof_nll_market['mean_nll']:.4f}")
print(f"  改善量(市場のみ-モデル)の95%CI=[{diff_nll['lo']:+.4f}, {diff_nll['hi']:+.4f}]  "
      f"→ {'PASS' if gate1_pass else 'FAIL'}")

# --- ゲート2: EV選抜picksの単勝回収率(Nested LOBO OOF held-out)
print(f"\n{'=' * 60}\nゲート2: EV選抜picksの単勝回収率(Nested LOBO OOF)\n{'=' * 60}")
oof_full = ev.lobo_oof(fit_fn_full, feats, ev_threshold=EV_THRESHOLD, odds_cap=ODDS_CAP)
boot_singles = ev.block_bootstrap(oof_full["picks"], bet="単勝", n=2000, seed=9202)
gate2_pass = boot_singles["lo"] > 100.0
print(f"  賭けたレース数={oof_full['n_bet_races']}/{len(races)}")
print(f"  単勝回収率(OOF)={oof_full['model']:.2f}%  市場(1番人気)={oof_full['market']:.2f}%  "
      f"95%CI=[{boot_singles['lo']:.2f}, {boot_singles['hi']:.2f}]  "
      f"→ {'PASS' if gate2_pass else 'FAIL'}")
full_table_oof = ev.full_table(oof_full["picks"])
print(full_table_oof.to_string(index=False))

# --- ゲート3: 閾値グリッドのselection_optimism型診断
print(f"\n{'=' * 60}\nゲート3: EV閾値グリッドの選択バイアス診断\n{'=' * 60}")
opt_thresholds = SE.selection_optimism_thresholds(ev, feats, beta_full_insample, THRESHOLD_GRID,
                                                  bet="単勝", n_rep=200, seed=9203)
edge_ratio = (opt_thresholds["true_edge_pt"] / opt_thresholds["true_edge_sd"]
             if opt_thresholds["true_edge_sd"] else 0.0)
gate3_pass = edge_ratio >= 2.0
print(f"  true_edge={opt_thresholds['true_edge_pt']:+.2f}pt (sd={opt_thresholds['true_edge_sd']:.2f}, "
      f"比={edge_ratio:+.3f})  → {'PASS' if gate3_pass else 'FAIL'}")

# --- ゲート4: chronological_oof(前半→後半・後半→前半の両方向)
print(f"\n{'=' * 60}\nゲート4: 時系列分割(前半→後半・後半→前半)\n{'=' * 60}")
dates = sorted({r["kaisai_date"] for r in races})
mid = len(dates) // 2
first_half_dates, second_half_dates = set(dates[:mid]), set(dates[mid:])
race_date = np.array([r["kaisai_date"] for r in races])
first_idx = np.where(np.isin(race_date, list(first_half_dates)))[0]
second_idx = np.where(np.isin(race_date, list(second_half_dates)))[0]

beta_fwd = MM.fit_conditional_logit(feats, idx=first_idx)
picks_fwd_full = MM.ev_picks(beta_fwd, feats, ev_threshold=EV_THRESHOLD, odds_cap=ODDS_CAP)
r_fwd = ev.evaluate(picks_fwd_full, idx=second_idx, bet="単勝")

beta_bwd = MM.fit_conditional_logit(feats, idx=second_idx)
picks_bwd_full = MM.ev_picks(beta_bwd, feats, ev_threshold=EV_THRESHOLD, odds_cap=ODDS_CAP)
r_bwd = ev.evaluate(picks_bwd_full, idx=first_idx, bet="単勝")

print(f"  前半{len(first_idx)}Rでfit→後半{len(second_idx)}Rで検証: "
      f"beta={np.round(beta_fwd, 3).tolist()}  単勝回収率={r_fwd['model']:.2f}%  "
      f"市場差={r_fwd['excess']:+.2f}pt")
print(f"  後半{len(second_idx)}Rでfit→前半{len(first_idx)}Rで検証: "
      f"beta={np.round(beta_bwd, 3).tolist()}  単勝回収率={r_bwd['model']:.2f}%  "
      f"市場差={r_bwd['excess']:+.2f}pt")
gate4_pass = (r_fwd["excess"] > 0) and (r_bwd["excess"] > 0) and (oof_full["excess"] > 0)
print(f"  → OOF・前半→後半・後半→前半の3方向とも市場差が同符号(プラス)か: "
      f"{'PASS' if gate4_pass else 'FAIL'}")

chrono = ev.chronological_oof(fit_fn_full, feats, min_train_blocks=3,
                              ev_threshold=EV_THRESHOLD, odds_cap=ODDS_CAP)
print(f"  (参考)expanding-window chronological_oof: n_folds={chrono['n_folds']}  "
      f"賭けたレース数={chrono['n_bet_races']}  単勝回収率={chrono['model']:.2f}%  "
      f"市場差={chrono['excess']:+.2f}pt")

# --- ゲート5: オッズ分布マッチのランダム選抜との比較
print(f"\n{'=' * 60}\nゲート5: オッズ分布マッチのランダム選抜との比較\n{'=' * 60}")
perm_test = SE.odds_matched_permutation_test(ev, races, oof_full["picks"], bet="単勝",
                                             n_perm=2000, seed=9205)
gate5_pass = perm_test["p_value_ge_real"] < 0.05
print(f"  実測回収率={perm_test['real_rate']:.2f}%  ランダム選抜2000回: "
      f"平均={perm_test['sim_mean']:.2f}%  中央値={perm_test['sim_median']:.2f}%  "
      f"95%点={perm_test['sim_p95']:.2f}%")
print(f"  p値(ランダム選抜が実測以上になる割合)={perm_test['p_value_ge_real']:.4f}  "
      f"→ {'PASS' if gate5_pass else 'FAIL'}")

# --- 総合判定
gates = {"gate1_nll": gate1_pass, "gate2_return_ci": gate2_pass, "gate3_selection_bias": gate3_pass,
         "gate4_chronological": gate4_pass, "gate5_permutation": gate5_pass}
n_pass = sum(gates.values())
decision = "採用候補(5ゲート全通過)" if n_pass == 5 else f"不採用(5ゲート中{n_pass}個のみ通過)"
print(f"\n{'=' * 60}\n総合判定: {decision}\n{'=' * 60}")
for k, v in gates.items():
    print(f"  {k}: {'PASS' if v else 'FAIL'}")

OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "ev_threshold": EV_THRESHOLD, "odds_cap": ODDS_CAP,
    "recent_form_signals": MM.RECENT_FORM_SIGNALS, "aptitude_signals": MM.APTITUDE_SIGNALS,
    "note": "Step1(2026-08-22): 市場アンカー型条件付きロジット+期待値ベッティングの検証。"
            "5ゲート全て満たして初めて採用候補、1つでも未達なら不採用として結果を透明に記録する。",
    "in_sample_reference": {
        "beta_full": beta_full_insample.tolist(), "beta_market_only": beta_market_insample.tolist(),
        "nll_full": nll_full_insample, "nll_market_only": nll_market_insample,
    },
    "gate1_nll": {"n_unique_patterns": n_unique_full, "n_folds": 36,
                 "oof_mean_nll_model": oof_nll_full["mean_nll"],
                 "oof_mean_nll_market_only": oof_nll_market["mean_nll"],
                 "bootstrap_diff_ci": diff_nll, "pass": gate1_pass},
    "gate2_return_ci": {"n_bet_races": oof_full["n_bet_races"], "model_return_pct": oof_full["model"],
                       "market_return_pct": oof_full["market"], "bootstrap_ci": boot_singles,
                       "full_table": full_table_oof.to_dict(orient="records"), "pass": gate2_pass},
    "gate3_selection_bias": {**opt_thresholds, "edge_ratio": edge_ratio, "pass": gate3_pass},
    "gate4_chronological": {
        "forward_fit_first_half": {"beta": beta_fwd.tolist(), "excess": r_fwd["excess"],
                                   "model": r_fwd["model"]},
        "backward_fit_second_half": {"beta": beta_bwd.tolist(), "excess": r_bwd["excess"],
                                     "model": r_bwd["model"]},
        "expanding_window": {"n_folds": chrono["n_folds"], "n_bet_races": chrono["n_bet_races"],
                             "model": chrono["model"], "excess": chrono["excess"]},
        "oof_excess": oof_full["excess"], "pass": gate4_pass,
    },
    "gate5_permutation": {**perm_test, "pass": gate5_pass},
    "gates_summary": gates, "n_gates_passed": n_pass, "decision": decision,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {OUT_JSON}")
