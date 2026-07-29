# -*- coding: utf-8 -*-
"""NAR「確信度」指標を過去の実際の払戻結果で較正するスクリプト。

現状の確信度(gap_pct: N位-N+1位のスコア差を(1位-最下位)幅で正規化した値)は、
モデルのスコア分布の形だけから決まる値で、実際の的中率と検証されたことが無かった。
ユーザーから「過去のレース結果から精度を上げられないか」との依頼を受け、以下を行う。

  1. 候補特徴量(gap_pct・score_spread・gap_top2、いずれも事前宣言・後出しなし)について、
     「その特徴量で分位バケットに分け、バケットごとの実測利益率(複勝+ワイドの当該レース
     単体での黒字/赤字)」というLOBO較正(1ブロックを除いた残りで分位点とバケット別実測率を
     計算し、除いたブロックに適用)のOOF Brier scoreを比較する。
  2. 「常に全体平均を予測する」自明な基準・現行の生gap_pctをそのまま確率として使う場合、
     とも比較し、較正が本当に改善しているかを確認する。
  3. box_n=3/4/5(5は4頭BOXと同じ重みを流用、既存設計と同じ)それぞれで独立に判定する。

【日々データが増える運用を想定した設計】
  - nar_dataset.load(rebuild=True) で常に最新の検証済みレース(race_results×payoutsの
    交差、日付ハードコードなし)から再構築する。
  - 分位点・バケット別実測率は「今ある全検証済みレース」から再計算するだけなので、
    このスクリプトを再実行するだけで較正表が最新化される(過去のJSONは上書き)。
  - 出力 confidence_calibration_nar.json 1本に box_n=3/4/5 の較正表・検証指標・
    fitted_on(日付一覧・レース数)・作成日時・コードハッシュをまとめる。

出力: confidence_calibration_nar.json / confidence_calibration_report.txt
"""
import json
import sys
from datetime import datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
DATA_DIR = LIB_DIR.parent.parent / "data" / "nar_pipeline"
sys.path.insert(0, str(LIB_DIR))
import nar_backtest as NB  # noqa: E402
import nar_dataset  # noqa: E402
import nar_eval as NE  # noqa: E402
import nar_signals as NS  # noqa: E402

K_BUCKETS = 4
CANDIDATES = ["gap_pct", "spread", "gap_top2"]
BOX_CONFIGS = {5: "winner_box4_nar.json", 4: "winner_box4_nar.json", 3: "winner_box3_nar.json"}

OUT_JSON = DATA_DIR / "confidence_calibration_nar.json"
OUT_TXT = DATA_DIR / "confidence_calibration_report.txt"
lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# --------------------------------------------------------------------- データ(常に最新化)
data = nar_dataset.load(rebuild=True)
races, actual = data["races"], data["actual"]
log(f"レース数: {len(races)}  日付: {data['dates']}")

priors = NS.make_priors(races)
NAMES = NS.ALL_SIGNALS
mats = NS.signal_matrices(races, priors, NAMES)
blocks = NE.blocks_of(races)


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def compute_features(box_n: int, W: np.ndarray, hit_mode: str = "profit") -> pd.DataFrame:
    ev = NE.Evaluator(races, actual, box_n=box_n)
    picks = NE.score_picks(mats, W, box_n)
    st, rt = ev.settler.returns_for(picks)
    if hit_mode == "profit":
        cols = [NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]
        profit = rt[:, cols].sum(axis=1) - st[:, cols].sum(axis=1)
        hit = (profit > 0).astype(float)
    elif hit_mode == "place":
        # 複勝(箱の中の誰かが3着以内)が的中したか。オッズの分散を除いた「順位付けの正しさ」。
        col = NB.BET_TYPES.index("複勝")
        hit = (rt[:, col] > 0).astype(float)
    else:
        raise ValueError(hit_mode)

    rows = []
    for i, m in enumerate(mats):
        num, den = m["S"] @ W, m["A"] @ W
        score = np.where(den > 0, num / den, -1e18)
        order = np.argsort(-score, kind="stable")
        s_sorted = score[order]
        n = len(s_sorted)
        spread = float(s_sorted[0] - s_sorted[-1])
        if n > box_n and spread > 0:
            gap_pct = float((s_sorted[box_n - 1] - s_sorted[box_n]) / spread)
        else:
            gap_pct = 1.0  # 全頭カバー、またはスコア差が全く無い場合は最大確信として扱う
        gap_top2 = float((s_sorted[0] - s_sorted[1]) / spread) if (n > 1 and spread > 0) else 0.0
        rows.append({
            "race_id": races[i]["race_id"], "kaisai_date": races[i]["kaisai_date"],
            "racecourse": races[i]["racecourse"], "field_size": n,
            "gap_pct": gap_pct, "spread": spread, "gap_top2": gap_top2, "hit": float(hit[i]),
        })
    return pd.DataFrame(rows)


def lobo_bucket_oof(df: pd.DataFrame, feature: str, k: int, blocks_arr: np.ndarray) -> np.ndarray:
    """LOBOで分位バケット較正のOOF予測確率を返す(train側だけで分位点・バケット実測率を計算)。"""
    oof = np.empty(len(df))
    vals = df[feature].to_numpy(dtype=float)
    hits = df["hit"].to_numpy(dtype=float)
    for b in sorted(set(blocks_arr)):
        train_idx = np.where(blocks_arr != b)[0]
        test_idx = np.where(blocks_arr == b)[0]
        train_vals, train_hits = vals[train_idx], hits[train_idx]
        qs = np.quantile(train_vals, np.linspace(0, 1, k + 1))
        qs = np.unique(qs)
        if len(qs) < 3:
            oof[test_idx] = train_hits.mean()
            continue
        edges = qs[1:-1]
        train_bucket = np.digitize(train_vals, edges)
        overall_rate = train_hits.mean()
        bucket_rate = {}
        for bk in np.unique(train_bucket):
            bucket_rate[bk] = train_hits[train_bucket == bk].mean()
        test_bucket = np.digitize(vals[test_idx], edges)
        oof[test_idx] = [bucket_rate.get(bk, overall_rate) for bk in test_bucket]
    return oof


def brier(pred: np.ndarray, actual_hit: np.ndarray) -> float:
    return float(np.mean((pred - actual_hit) ** 2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


results = {}
for box_n, json_name in BOX_CONFIGS.items():
  for hit_mode, hit_label in [("place", "複勝的中(3着以内)"), ("profit", "複勝+ワイド単体黒字")]:
    log("\n" + "=" * 72)
    log(f"box_n={box_n}(較正の重み元: {json_name})  hit定義: {hit_label}")
    log("=" * 72)
    w = json.loads((DATA_DIR / json_name).read_text(encoding="utf-8"))["weights"]
    W = wvec(w)
    df = compute_features(box_n, W, hit_mode=hit_mode)
    log(f"  対象レース数: {len(df)}  実測hit率({hit_label}): {df['hit'].mean() * 100:.1f}%")

    # --- 自明な基準: LOBOで「学習側の平均hit率」を常に予測する ---
    trivial_oof = np.empty(len(df))
    for b in sorted(set(blocks)):
        train_idx = np.where(blocks != b)[0]
        test_idx = np.where(blocks == b)[0]
        trivial_oof[test_idx] = df["hit"].to_numpy()[train_idx].mean()
    trivial_brier = brier(trivial_oof, df["hit"].to_numpy())
    log(f"  自明な基準(常に学習側平均を予測)のOOF Brier score: {trivial_brier:.4f}")

    # --- 現行: 生のgap_pctをそのまま確率として使う(較正なし) ---
    raw_brier = brier(df["gap_pct"].to_numpy(), df["hit"].to_numpy())
    log(f"  現行(生gap_pctをそのまま確率として使用)のBrier score: {raw_brier:.4f}(参考、OOFではなくin-sample)")

    cand_results = {}
    for feat in CANDIDATES:
        oof_pred = lobo_bucket_oof(df, feat, K_BUCKETS, blocks)
        b = brier(oof_pred, df["hit"].to_numpy())
        sp = spearman(df[feat].to_numpy(), df["hit"].to_numpy())
        cand_results[feat] = {"oof_brier": b, "spearman_with_hit": sp}
        log(f"  候補[{feat:10s}] OOF Brier(k={K_BUCKETS}分位較正)={b:.4f}  "
            f"生の値とhitのSpearman相関={sp:+.3f}")

    best_feat = min(cand_results, key=lambda f: cand_results[f]["oof_brier"])
    best_brier = cand_results[best_feat]["oof_brier"]
    beats_trivial = best_brier < trivial_brier
    log(f"\n  最良候補: {best_feat}(OOF Brier={best_brier:.4f} vs 自明な基準{trivial_brier:.4f})")
    log(f"  自明な基準を上回るか: {'YES' if beats_trivial else 'NO'}")

    # --- 本番用較正表: 全データで分位点・バケット別実測率を計算(次回はこのスクリプトの再実行で更新) ---
    final_vals = df[best_feat].to_numpy(dtype=float)
    final_hits = df["hit"].to_numpy(dtype=float)
    qs = np.unique(np.quantile(final_vals, np.linspace(0, 1, K_BUCKETS + 1)))
    edges = qs[1:-1].tolist()
    bucket_idx = np.digitize(final_vals, edges)
    bucket_table = []
    for bk in sorted(set(bucket_idx)):
        mask = bucket_idx == bk
        bucket_table.append({
            "bucket": int(bk), "n_races": int(mask.sum()),
            "hit_rate_pct": round(float(final_hits[mask].mean()) * 100, 1),
            "value_range": [float(final_vals[mask].min()), float(final_vals[mask].max())],
        })

    results.setdefault(box_n, {})[hit_mode] = {
        "hit_label": hit_label,
        "n_races": len(df), "overall_hit_rate_pct": round(float(df["hit"].mean()) * 100, 1),
        "trivial_baseline_oof_brier": trivial_brier,
        "raw_gap_pct_insample_brier": raw_brier,
        "candidates": cand_results,
        "chosen_feature": best_feat, "beats_trivial_baseline": bool(beats_trivial),
        "calibration_table": {
            "feature": best_feat, "n_buckets": K_BUCKETS, "edges": edges, "buckets": bucket_table,
        },
    }

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "fitted_on": {"n_races": len(races), "dates": data["dates"]},
    "method": "lobo_quantile_bucket_calibration",
    "k_buckets": K_BUCKETS, "candidates_tested": CANDIDATES,
    "results_by_box_n": {str(k): v for k, v in results.items()},
    "code_sha256": {
        name: sha256((LIB_DIR / name).read_bytes()).hexdigest()[:16]
        for name in ["nar_signals.py", "nar_eval.py", "nar_backtest.py", "nar_dataset.py",
                     "nar_confidence_calibrate.py"]
    },
    "created_at": datetime.now().isoformat(timespec="seconds"),
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
