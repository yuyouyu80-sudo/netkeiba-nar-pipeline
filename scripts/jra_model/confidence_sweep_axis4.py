# -*- coding: utf-8 -*-
"""予想4頭軸流し(軸+相手3頭)モデルの確信度フィルタリング検証(5〜10レース/日)。

scripts/jra_model/confidence_sweep_box4.py(BOX買い版、scratchpad常駐)と同じ設計
(確信度指標gap_pctはCONF_N=BOX_Nに固定、N_RANGEは同一ランキング上のカットオフ件数のみに
使う、Nを増やすほど採用レースが単調増加する入れ子構造)を踏襲する。決済は
scripts/jra_model/jra_axis_backtest.py(馬連・ワイド・3連複・馬単(軸流し/マルチ)・
3連単(軸流し/マルチ)の7区分、軸=スコア1位、相手=スコア2〜4位)を使う。

2026-08-21の重み探索(jra_axis_search_2026_08_21.py)の結論: 軸流し専用の新規500パターン
探索はREJECTED(選択バイアス診断のtrue_edge/sd比が採否ゲート2.0を大幅未達)。現行box4重み
(winner_box4.json)をそのまま軸流しに転用した場合も、全211レースでは市場を+23.35pt上回るが、
重み自体のfit母集団(105レース)を除外した公正な評価(106レース)では市場超過が消え、
点推定はむしろマイナス(-19.50pt、95%CI=[-51.19,+18.52]ptで0をまたぐ)に転じる。
つまり「市場に対する優位性が統計的に実証されたモデル」は今回の検証では見つからなかった。
本スクリプトはその前提のもと、現行box重み転用モデル(実用上の最有力候補)・軸流し専用
探索候補(不採用/参考)・市場ベンチマーク(軸=1番人気)の3系統を並記する透明な検証記録
として構築する。詳細はdata/jra_pipeline/jra_axis_search_2026_08_21_result.json参照。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_axis_backtest as AB  # noqa: E402
import jra_axis_eval as AE  # noqa: E402
import jra_dataset  # noqa: E402
import jra_signals as JS  # noqa: E402

BOX_N = 4  # 軸+相手3頭=合計4頭
CONF_N = BOX_N
N_RANGE = [5, 6, 7, 8, 9, 10]
CURRENT_WEIGHT_FILE = "winner_box4.json"
SEARCH_RESULT_FILE = "jra_axis_search_2026_08_21_result.json"
OUT_CSV = DATA_DIR / "confidence_sweep_axis4.csv"

data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
print(f"races={len(races)}  BOX_N={BOX_N}")

priors_all = JS.make_priors([r["df"] for r in races])
NAMES = JS.ALL_SIGNALS
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
settler = AB.AxisSettler(races, actual, box_n=BOX_N)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def summarize_subset(picks_by_race_idx: list, idx_list: list, model_name: str, label: str) -> pd.DataFrame:
    # 2026-08-21発見: AxisSettler.returns_for()はpicksが全レース分・元の並び順である
    # ことを前提にposition=race_indexとして扱う(BoxSettlerと同じ契約)。高確信度Nレース/日
    # のような部分集合を渡すとレースを取り違えて的中判定がずれる(IndexErrorで顕在化した)。
    # 部分集合は必ずreturns_for_at()で元のレースindexを明示して問い合わせる。
    pairs = [(i, picks_by_race_idx[i]) for i in idx_list]
    st, rt = settler.returns_for_at(pairs)
    rows = []
    n_races = len(idx_list)
    for j, bt in enumerate(AB.BET_TYPES_AXIS):
        s, r_ = int(st[:, j].sum()), int(rt[:, j].sum())
        hits = int((rt[:, j] > 0).sum())
        rows.append({
            "model": model_name, "scope": label, "bet_type": bt,
            "races": n_races, "hit_races": hits,
            "hit_rate_pct": round(hits / n_races * 100, 1) if n_races else 0.0,
            "total_stake": s, "total_return": r_,
            "return_rate_pct": round(r_ / s * 100, 1) if s else 0.0,
        })
    return pd.DataFrame(rows)


def analyze_model(model_name: str, w: np.ndarray) -> pd.DataFrame:
    conf_rows = []
    picks_by_race_idx = []
    for i, (r, m) in enumerate(zip(races, mats_all)):
        num, den = m["S"] @ w, m["A"] @ w
        score = np.where(den > 0, num / den, -1e18)
        n = len(r["df"])
        order = np.argsort(-score, kind="stable")
        sorted_scores = score[order]
        top_score, bottom_score = sorted_scores[0], sorted_scores[-1]
        spread = top_score - bottom_score
        if n > CONF_N:
            gap = sorted_scores[CONF_N - 1] - sorted_scores[CONF_N]
            gap_pct = gap / spread if spread > 0 else 0.0
        else:
            gap_pct = np.inf  # 軸+相手の合計がフィールド全体を覆う
        conf_rows.append({"race_idx": i, "kaisai_date": r["kaisai_date"], "gap_pct": gap_pct})
        picks_by_race_idx.append(order[:min(BOX_N, n)])

    conf_df = pd.DataFrame(conf_rows)
    all_idx = list(range(len(races)))
    summaries = [summarize_subset(picks_by_race_idx, all_idx, model_name, f"全{len(races)}レース")]
    conf_sorted = conf_df.sort_values("gap_pct", ascending=False, kind="stable")
    for n_cut in N_RANGE:
        selected = conf_sorted.groupby("kaisai_date", group_keys=False).head(n_cut)
        idx_list = selected["race_idx"].tolist()
        summaries.append(summarize_subset(
            picks_by_race_idx, idx_list, model_name,
            f"高確信度{n_cut}レース/日(計{len(idx_list)}レース)"))
    return pd.concat(summaries, ignore_index=True)


if __name__ == "__main__":
    current_w = json.loads((DATA_DIR / CURRENT_WEIGHT_FILE).read_text(encoding="utf-8"))
    W_CURRENT = wvec(current_w["weights"])
    current_result = analyze_model(
        f"軸流しモデル(現行box{BOX_N}重み転用、{CURRENT_WEIGHT_FILE})", W_CURRENT)

    search_result = json.loads((DATA_DIR / SEARCH_RESULT_FILE).read_text(encoding="utf-8"))
    cand = search_result["results_by_box"][str(BOX_N)]["best_full_population"]
    W_CANDIDATE = wvec(cand["weights"])
    candidate_result = analyze_model(
        f"軸流し専用探索候補(pattern#{cand['pattern_index']}・不採用/参考)", W_CANDIDATE)

    mkt_picks = AE.market_picks(races, BOX_N)
    mkt_st, mkt_rt = settler.returns_for(mkt_picks)
    mkt_rows = []
    for j, bt in enumerate(AB.BET_TYPES_AXIS):
        s, r_ = int(mkt_st[:, j].sum()), int(mkt_rt[:, j].sum())
        hits = int((mkt_rt[:, j] > 0).sum())
        n_races = len(races)
        mkt_rows.append({
            "model": "市場ベンチマーク(軸=1番人気・相手=2〜{}番人気)".format(BOX_N),
            "scope": f"全{n_races}レース", "bet_type": bt,
            "races": n_races, "hit_races": hits,
            "hit_rate_pct": round(hits / n_races * 100, 1) if n_races else 0.0,
            "total_stake": s, "total_return": r_,
            "return_rate_pct": round(r_ / s * 100, 1) if s else 0.0,
        })
    market_result = pd.DataFrame(mkt_rows)

    result = pd.concat([current_result, candidate_result, market_result], ignore_index=True)
    result.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print(f"\nwrote {OUT_CSV}")
