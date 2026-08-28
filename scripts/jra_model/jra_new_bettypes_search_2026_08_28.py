# -*- coding: utf-8 -*-
"""JRA Stage2 Phase J3(2026-08-28): 2頭軸流し・フォーメーション買いの事前登録済み11構成を
評価する。scripts/jra_model/jra_multi_axis_backtest.py/jra_formation_backtest.py(新設)を使用。

候補順位付けは現行本番(winner_v3.json、box5用に選定された10シグナル等重みではない実際の
重み)のスコアをそのまま使う(過去のNAR/JRA軸流し検証の前例「現行box重みをそのまま買い方
だけ変えて転用したらどうなるか」を踏襲)。Phase J2の正則化モデルが確立した知見(box3
Dirichlet等)が出た場合は改めて別スコアで追試できるよう、スコア計算を関数化しておく。

事前登録グリッド(計画書 valiant-cuddling-aho.md Phase J3、過去10ラウンドの教訓により
「数百パターンから選ぶ」型の探索はしない):
  2頭軸流し・3連複: K(相手候補数)∈{3,5,7}
  2頭軸流し・3連単(2軸box): K∈{3,5,7}
  フォーメーション・馬単: (A,B)∈{(1,3),(2,4),(1,5)}
  フォーメーション・3連単: (A,B,C)∈{(1,2,4),(2,3,5)}

各構成を全246レース(参考)・fit母集団(105レース)除外held-out141レースの両方で評価し、
held-out集合上でmodel vs marketのペア差分95%CI(block_bootstrap_diff)を主判定に使う。
"""
import json
import sys
from pathlib import Path

import numpy as np

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_backtest as JB  # noqa: E402
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_formation_backtest as FM  # noqa: E402
import jra_multi_axis_backtest as MA  # noqa: E402
import jra_signals as JS  # noqa: E402

OUT_JSON = DATA_DIR / "jra_new_bettypes_search_2026_08_28_result.json"
OUT_TXT = DATA_DIR / "jra_new_bettypes_search_2026_08_28_report.txt"
UNIT = JB.UNIT

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)")

winner_v3 = json.loads((DATA_DIR / "winner_v3.json").read_text(encoding="utf-8"))
fitted_on = winner_v3["fitted_on"]
NAMES = JS.LEGACY_SIGNALS
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
W_CURRENT = np.array([float(winner_v3["weights"].get(n, 0.0)) for n in NAMES])
ranges = JB.race_row_ranges(races) if hasattr(JB, "race_row_ranges") else None
if ranges is None:
    # jra_backtest.pyに無ければローカルで計算(nar_logistic.race_row_rangesと同じロジック)
    ranges = []
    pos = 0
    for m in mats_all:
        n = m["S"].shape[0]
        ranges.append((pos, pos + n))
        pos += n

scores_per_race = []
for m in mats_all:
    num, den = m["S"] @ W_CURRENT, m["A"] @ W_CURRENT
    scores_per_race.append(np.where(den > 0, num / den, -1e18))

blocks = JE.blocks_of(races)
held_out_blocks = JE.held_out_block_subset(fitted_on, races)
held_out_idx = np.where(np.isin(blocks, held_out_blocks))[0]
log(f"held-outブロック数={len(held_out_blocks)}  held-outレース数={len(held_out_idx)}")

ma_settler = MA.MultiAxisSettler(races, actual)
fm_settler = FM.FormationSettler(races, actual)


def market_axis2_picks(k_total: int) -> list:
    """市場ベンチマーク: 上位k_total人気を軸1・軸2・相手の順で並べたpicks(1〜2番人気を軸、
    3番人気以降を相手とする、市場版2頭軸)。"""
    picks = []
    for r in races:
        import pandas as pd
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        key = np.where(np.isnan(ninki), 1e18, ninki)
        picks.append(np.argsort(key, kind="stable")[:k_total])
    return picks


def market_formation_picks(a_size: int, b_size: int, c_size: int = 0) -> list:
    picks = []
    for r in races:
        import pandas as pd
        ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
        key = np.where(np.isnan(ninki), 1e18, ninki)
        order = np.argsort(key, kind="stable")
        n = len(order)
        a, b = order[:min(a_size, n)], order[:min(b_size, n)]
        c = order[:min(c_size, n)] if c_size else np.array([], dtype=order.dtype)
        picks.append((a, b, c))
    return picks


def eval_config(settler, picks_model: list, picks_market: list, bet_type: str,
                bet_types_all: list) -> dict:
    """settler.returns_for()で決済し、held-out集合上のmodel vs marketペア差分95%CIを含めた
    結果を返す(jra_eval.Evaluator相当の比推定量ブロックブートストラップを直接実装、
    2頭軸・フォーメーションSettlerはjra_eval.Evaluatorの外側にあるため専用に組む)。"""
    col = bet_types_all.index(bet_type)
    st_m, rt_m = settler.returns_for(picks_model)
    st_k, rt_k = settler.returns_for(picks_market)

    def rate(st, rt, idx=None):
        s = st[:, col] if idx is None else st[idx, col]
        r = rt[:, col] if idx is None else rt[idx, col]
        tot = s.sum()
        return float(r.sum() / tot * 100) if tot else 0.0

    model_all = rate(st_m, rt_m)
    market_all = rate(st_k, rt_k)
    model_ho = rate(st_m, rt_m, held_out_idx)
    market_ho = rate(st_k, rt_k, held_out_idx)

    by_block = {b: np.where(np.isin(blocks, [b]))[0] for b in held_out_blocks}
    rng = np.random.default_rng(hash(bet_type) % (2**31))
    ids = list(held_out_blocks)
    diffs = np.empty(2000)
    for k in range(2000):
        chosen = rng.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        sm, rm = st_m[idx, col].sum(), rt_m[idx, col].sum()
        sk, rk = st_k[idx, col].sum(), rt_k[idx, col].sum()
        vm = rm / sm * 100 if sm else 0.0
        vk = rk / sk * 100 if sk else 0.0
        diffs[k] = vm - vk
    return {
        "bet_type": bet_type,
        "model_all": model_all, "market_all": market_all, "excess_all": model_all - market_all,
        "model_held_out": model_ho, "market_held_out": market_ho,
        "excess_held_out": model_ho - market_ho,
        "held_out_ci": {"mean": float(diffs.mean()), "lo": float(np.percentile(diffs, 2.5)),
                        "hi": float(np.percentile(diffs, 97.5)), "n_blocks": len(ids)},
        "n_held_out_races": int(len(held_out_idx)),
    }


results = []

# --- 2頭軸流し(3連複・3連単) K in {3,5,7}
for K in (3, 5, 7):
    picks_model = MA.picks_2axis_from_scores(np.concatenate(scores_per_race), ranges,
                                             range(len(races)), K)
    picks_market = market_axis2_picks(2 + K)
    for bt in ["3連複", "3連単_2軸box", "3連単_2軸固定"]:
        r = eval_config(ma_settler, picks_model, picks_market, bt, MA.BET_TYPES_2AXIS)
        r["config"] = f"2軸流し K={K}"
        results.append(r)
        log(f"[2軸流し K={K}] {bt}: 全246={r['excess_all']:+.2f}pt  "
            f"held-out={r['excess_held_out']:+.2f}pt  "
            f"CI=[{r['held_out_ci']['lo']:+.2f},{r['held_out_ci']['hi']:+.2f}]")

# --- フォーメーション・馬単 (A,B) in {(1,3),(2,4),(1,5)}
for A, B in [(1, 3), (2, 4), (1, 5)]:
    picks_model = FM.picks_formation_from_scores(np.concatenate(scores_per_race), ranges,
                                                 range(len(races)), A, B, 0)
    picks_market = market_formation_picks(A, B, 0)
    r = eval_config(fm_settler, picks_model, picks_market, "馬単_フォーメーション",
                    FM.BET_TYPES_FORMATION)
    r["config"] = f"馬単フォーメーション A={A} B={B}"
    results.append(r)
    log(f"[馬単フォーメーション A={A} B={B}]: 全246={r['excess_all']:+.2f}pt  "
        f"held-out={r['excess_held_out']:+.2f}pt  "
        f"CI=[{r['held_out_ci']['lo']:+.2f},{r['held_out_ci']['hi']:+.2f}]")

# --- フォーメーション・3連単 (A,B,C) in {(1,2,4),(2,3,5)}
for A, B, C in [(1, 2, 4), (2, 3, 5)]:
    picks_model = FM.picks_formation_from_scores(np.concatenate(scores_per_race), ranges,
                                                 range(len(races)), A, B, C)
    picks_market = market_formation_picks(A, B, C)
    r = eval_config(fm_settler, picks_model, picks_market, "3連単_フォーメーション",
                    FM.BET_TYPES_FORMATION)
    r["config"] = f"3連単フォーメーション A={A} B={B} C={C}"
    results.append(r)
    log(f"[3連単フォーメーション A={A} B={B} C={C}]: 全246={r['excess_all']:+.2f}pt  "
        f"held-out={r['excess_held_out']:+.2f}pt  "
        f"CI=[{r['held_out_ci']['lo']:+.2f},{r['held_out_ci']['hi']:+.2f}]")

n_sig = sum(1 for r in results if r["held_out_ci"]["lo"] > 0)
log(f"\nheld-out CI下限>0(市場超過が有意)の構成数: {n_sig}/{len(results)}")

OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "held_out_blocks": held_out_blocks,
    "n_held_out_races": int(len(held_out_idx)), "results": results,
}, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
log(f"\n保存: {OUT_JSON}")
log(f"保存: {OUT_TXT}")
