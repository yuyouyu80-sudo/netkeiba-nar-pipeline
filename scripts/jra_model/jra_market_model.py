# -*- coding: utf-8 -*-
"""市場アンカー型条件付きロジット(2026-08-22新設、Step1)。

u_i = beta0・log(q_i) + beta1・z_近走_i + beta2・z_適性_i
  q_i    = 単勝オッズ由来のレース内正規化インプライド確率(1/oddsを合計1に正規化)
  z_近走  = timediff/form/margin/agari(既存jra_signals.compute_signals()の0-1値)の平均を
           レース内zscore標準化した合成特徴量
  z_適性  = interval/apt/concerned/bms/course/sire の平均を同様にzscore標準化した合成特徴量
勝率(モデル) p_i = softmax(u)_i (レース内)

新規シグナルの追加は行わない(既存jra_signals.compute_signals()の生値を平均・標準化するだけ)。
推定はscipy.optimize.minimizeでNLL(負の対数尤度)を最小化する(自由パラメータ3個のみ、
211レース規模なら数秒以内に収束する)。

2026-08-22のOpus 5サブエージェント調査で判明した設計上の要点:
  * 既存box5/4/3探索は非負制約のDirichlet単体上を探索していたため「市場が過大評価している
    シグナルを負の重みでフェードする」戦略が探索空間の外にあった。本モデルはbeta1/beta2に
    符号制約を課さない。
  * 市場オッズは条件付きロジットでほぼ完全較正(beta0単独フィットでbeta0≈0.998)されており、
    「レース内で順位付けして上位N頭を買う」設計では市場に対する優位性を出しにくい。本モデルは
    「上位N頭を選ぶ」のではなく「市場に対して割安(EV>=閾値)な馬だけ賭ける」設計にする。
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize

import jra_signals as JS

RECENT_FORM_SIGNALS = ["timediff", "form", "margin", "agari"]
APTITUDE_SIGNALS = ["interval", "apt", "concerned", "bms", "course", "sire"]
N_PARAMS = 3  # beta0(市場アンカー), beta1(近走合成), beta2(適性合成)
DEFAULT_EV_THRESHOLD = 1.2
DEFAULT_ODDS_CAP = 20.0


def normalized_implied_prob(odds: np.ndarray) -> np.ndarray:
    """単勝オッズ配列からレース内正規化インプライド確率を返す(1/oddsを合計1に正規化)。
    オッズが欠損・0以下の馬はNaN(呼び出し側で欠損として扱う)。"""
    odds = np.asarray(odds, dtype=float)
    inv = np.where(odds > 0, 1.0 / odds, np.nan)
    s = np.nansum(inv)
    if not np.any(~np.isnan(inv)) or s <= 0:
        return np.full_like(inv, np.nan)
    return inv / s


def _zscore_in_race(x: np.ndarray) -> np.ndarray:
    """レース内zscore標準化。欠損値は中立(0)扱い(combine_signalsの重み再配分と同じ思想、
    欠損馬をモデルから除外せず「市場情報+ゼロ寄与の合成特徴量」として扱う)。"""
    x = np.asarray(x, dtype=float)
    valid = ~np.isnan(x)
    if valid.sum() < 2:
        return np.zeros_like(x)
    mu, sd = np.nanmean(x), np.nanstd(x)
    if sd <= 1e-12:
        return np.zeros_like(x)
    out = (x - mu) / sd
    out[~valid] = 0.0
    return out


def extract_winner_idx(races: list, actual: dict) -> list:
    """各レースの実際の勝ち馬(単勝の払戻>0になる馬)の行indexを抽出する。複数該当(同着)は
    最初の1頭、該当なし(結果データ欠損)はNone。"""
    out = []
    for r in races:
        win_map = actual.get(r["race_id"], {}).get("単勝", {})
        umaban = r["df"]["umaban"].astype(int).to_numpy()
        winners = [i for i, u in enumerate(umaban) if win_map.get(int(u), 0) > 0]
        out.append(winners[0] if winners else None)
    return out


def build_composite_features(races: list, actual: dict, priors: dict, class_ordinal_map=None) -> list:
    """レースごとに odds/q/log_q/z_recent/z_apt/winner_idx を辞書で返す。"""
    class_map = class_ordinal_map if class_ordinal_map is not None else JS.CLASS_ORDINAL
    winner_idx_list = extract_winner_idx(races, actual)
    out = []
    for r, winner_idx in zip(races, winner_idx_list):
        df = r["df"]
        current_class = JS._class_ordinal(r["race_name"], class_map)
        sig = JS.compute_signals(df, current_class, priors, class_map)
        recent = np.nanmean(
            np.column_stack([sig[n].to_numpy(dtype=float) for n in RECENT_FORM_SIGNALS]), axis=1)
        apt = np.nanmean(
            np.column_stack([sig[n].to_numpy(dtype=float) for n in APTITUDE_SIGNALS]), axis=1)
        odds = pd.to_numeric(df["bias_win_odds"], errors="coerce").to_numpy(dtype=float)
        q = normalized_implied_prob(odds)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_q = np.log(np.where(q > 0, q, np.nan))
        out.append({
            "odds": odds, "q": q, "log_q": log_q,
            "z_recent": _zscore_in_race(recent), "z_apt": _zscore_in_race(apt),
            "winner_idx": winner_idx,
        })
    return out


def _utility(params: np.ndarray, f: dict) -> np.ndarray:
    beta0, beta1, beta2 = params
    return beta0 * f["log_q"] + beta1 * f["z_recent"] + beta2 * f["z_apt"]


def race_nll(params: np.ndarray, feats: list, idx=None) -> float:
    """条件付きロジットの負の対数尤度(レース平均)。idxで対象レースを絞れる
    (fit_fn(train_idx)からそのまま使う)。"""
    total, n_eval = 0.0, 0
    race_range = range(len(feats)) if idx is None else idx
    for i in race_range:
        f = feats[i]
        winner = f["winner_idx"]
        valid = ~np.isnan(f["log_q"])
        if winner is None or not valid[winner] or valid.sum() < 2:
            continue
        u = _utility(params, f)
        u_valid = u[valid]
        m = u_valid.max()
        logsumexp = m + np.log(np.exp(u_valid - m).sum())
        total += -(u[winner] - logsumexp)
        n_eval += 1
    return total / n_eval if n_eval else float("inf")


def fit_conditional_logit(feats: list, idx=None, x0=None) -> np.ndarray:
    """scipy.optimize.minimize(Nelder-Mead)でNLLを最小化し、beta=[beta0,beta1,beta2]を返す。"""
    x0 = np.array([1.0, 0.0, 0.0]) if x0 is None else np.asarray(x0, dtype=float)
    res = minimize(race_nll, x0, args=(feats, idx), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 3000, "maxfev": 3000})
    return res.x


def fit_beta0_only(feats: list, idx=None, x0: float = 1.0) -> np.ndarray:
    """市場のみモデル(beta1=beta2=0固定)。beta0だけを1次元最適化し[beta0,0,0]を返す
    (ゲート1の比較対象=「市場だけ知っている場合のNLL」を作るための縮小モデル)。"""
    def nll1(b0):
        return race_nll(np.array([b0[0], 0.0, 0.0]), feats, idx)
    res = minimize(nll1, np.array([x0]), method="Nelder-Mead",
                   options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 500})
    return np.array([res.x[0], 0.0, 0.0])


def predict_p(params: np.ndarray, feats: list) -> list:
    """各レースの馬ごとのモデル勝率配列(softmax、レース内合計1。オッズ欠損馬はNaN)を返す。"""
    out = []
    for f in feats:
        valid = ~np.isnan(f["log_q"])
        if not valid.any():
            out.append(np.full_like(f["log_q"], np.nan))
            continue
        u = _utility(params, f)
        u_valid = u[valid]
        m = u_valid.max()
        exp_u = np.exp(u_valid - m)
        p_valid = exp_u / exp_u.sum()
        p = np.full_like(u, np.nan)
        p[valid] = p_valid
        out.append(p)
    return out


def ev_picks(params: np.ndarray, feats: list, ev_threshold: float = DEFAULT_EV_THRESHOLD,
            odds_cap: float = DEFAULT_ODDS_CAP) -> list:
    """EV(=p×odds)がev_threshold以上かつodds<=odds_capの馬の行indexを返す
    (レースごと、空配列=そのレースは見送り)。"""
    p_list = predict_p(params, feats)
    picks = []
    for f, p in zip(feats, p_list):
        with np.errstate(invalid="ignore"):
            ev = p * f["odds"]
            sel_mask = (ev >= ev_threshold) & (f["odds"] <= odds_cap) & ~np.isnan(ev)
        picks.append(np.where(sel_mask)[0])
    return picks
