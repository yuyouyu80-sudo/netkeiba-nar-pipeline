# -*- coding: utf-8 -*-
"""NAR「確信度」指標を過去の実際の払戻結果で較正するスクリプト。

現状の確信度(gap_pct: N位-N+1位のスコア差を(1位-最下位)幅で正規化した値)は、
モデルのスコア分布の形だけから決まる値で、実際の的中率と検証されたことが無かった。
ユーザーから「過去のレース結果から精度を上げられないか」との依頼を受け、以下を行う。

  1. 候補特徴量(gap_pct・score_spread・gap_top2、いずれも事前宣言・後出しなし)について、
     「その特徴量で分位バケットに分け、バケットごとの実測利益率」というLOBO較正
     (1ブロックを除いた残りで分位点とバケット別実測率を計算し、除いたブロックに適用)の
     OOF Brier scoreを比較する。
  2. 「常に全体平均を予測する」自明な基準とも比較し、較正が本当に改善しているかを確認する。
  3. box_n=3/4/5それぞれで独立に判定する(2026-08-01以降、box5もbox4の重み流用を
     やめ独立重みを持つため、3サイズとも完全に別のスコア分布・別の較正になる)。

【2026-07-30 追記: 符号チェックを追加】
  システムエンジニアレビューにより、box_n=5×place判定でgap_pctを採用していた従来ロジックは
  「OOF Brier scoreが最良」という理由だけで選ばれていたが、実際にはgap_pctと的中率の
  Spearman相関が負(-0.15、小頭数レースを除いても同様)で、「スコア差が大きいほど的中率が
  下がる」という直感に反した逆転が起きていることが判明した(較正自体はバケット別実測率を
  そのまま使うため、符号が逆でもBrier scoreは改善し得る = Brier score単体では
  「表示上の大小と的中率の大小が一致しているか」を保証しない)。
  このため、単純に「OOF Brierが最良の候補」を選ぶのではなく、Spearman相関の符号も
  必ず確認し、正の相関を持つ候補の中で最良のものを優先する(全候補が負相関の場合のみ
  警告付きでBrier最良を採用)方式に変更した。

【2026-07-30 追記: 段階的的中確率のはしご(5頭確信度〜1頭確信度)を追加】
  ユーザー要望により、「上位K頭picksの中に実際の1着馬が入っていたか」をK=5,4,3,2,1
  それぞれで的中ラベルとし、同じLOBO較正手法で確信度をはしご状に算出する
  (`topk_ladder`)。1着馬の的中判定は単勝の払戻(box_n=K頭を選んだときの単勝リターンが
  0より大きいか)をそのまま使う(nar_backtest.BoxSettlerが頭数<Kの場合を自動的に
  全頭カバーとして扱うため、小頭数レースの特別扱いは不要)。重みは予想5頭表示と同じ
  LADDER_WEIGHTS_JSON(2026-08-01以降はwinner_box5_nar.json)を一貫して使用し、
  Kが変わっても「同じ順位付けをどこまで信頼できるか」という一貫した意味になる
  ようにしている。

【日々データが増える運用を想定した設計】
  - nar_dataset.load(rebuild=True) で常に最新の検証済みレース(race_results×payoutsの
    交差、日付ハードコードなし)から再構築する。
  - 分位点・バケット別実測率は「今ある全検証済みレース」から再計算するだけなので、
    このスクリプトを再実行するだけで較正表が最新化される(過去のJSONは上書き)。
  - 出力 confidence_calibration_nar.json 1本に box_n=3/4/5 の較正表・topk_ladder・
    検証指標・fitted_on(日付一覧・レース数)・作成日時・コードハッシュをまとめる。

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
LADDER_CANDIDATES = ["gap_boundary_k", "gap_top2"]
# 2026-08-01: box5はbox4の重み流用をやめ、独立に300パターン探索した
# winner_box5_nar.jsonを持つ(nar_search300_2026_08_01.py)。
BOX_CONFIGS = {5: "winner_box5_nar.json", 4: "winner_box4_nar.json", 3: "winner_box3_nar.json"}
LADDER_KS = [5, 4, 3, 2, 1]
LADDER_WEIGHTS_JSON = "winner_box5_nar.json"  # 予想5頭表示(box5)と同じ重みを使う

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


def brier(pred: np.ndarray, actual_hit: np.ndarray) -> float:
    return float(np.mean((pred - actual_hit) ** 2))


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra = pd.Series(a).rank()
    rb = pd.Series(b).rank()
    return float(np.corrcoef(ra, rb)[0, 1])


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


def choose_feature_sign_safe(cand_results: dict) -> tuple:
    """OOF Brierが良い順に候補を並べ、Spearman相関が非負(>=0)の中から最良を選ぶ。
    非負の候補が1つも無い場合のみ、全候補中の最良(符号無視)を警告付きで返す。"""
    ranked = sorted(cand_results.items(), key=lambda kv: kv[1]["oof_brier"])
    for feat, r in ranked:
        if r["spearman_with_hit"] >= 0:
            return feat, False
    return ranked[0][0], True


def build_calibration_table(vals: np.ndarray, hits: np.ndarray, k_buckets: int) -> tuple:
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, k_buckets + 1)))
    edges = qs[1:-1].tolist()
    bucket_idx = np.digitize(vals, edges)
    table = []
    for bk in sorted(set(bucket_idx)):
        mask = bucket_idx == bk
        table.append({
            "bucket": int(bk), "n_races": int(mask.sum()),
            "hit_rate_pct": round(float(hits[mask].mean()) * 100, 1),
            "value_range": [float(vals[mask].min()), float(vals[mask].max())],
        })
    return edges, table


def max_hit_rate_bucket(table: list) -> dict:
    """実測的中率が最も高いバケットを返す(値の大小の並び順を仮定しない)。"""
    return max(table, key=lambda b: b["hit_rate_pct"])


# =======================================================================
# 1) 既存: box_n=3/4/5 × hit_mode=place/profit
# =======================================================================
def compute_features(box_n: int, W: np.ndarray, hit_mode: str = "profit") -> pd.DataFrame:
    ev = NE.Evaluator(races, actual, box_n=box_n)
    picks = NE.score_picks(mats, W, box_n)
    st, rt = ev.settler.returns_for(picks)
    if hit_mode == "profit":
        cols = [NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]
        profit = rt[:, cols].sum(axis=1) - st[:, cols].sum(axis=1)
        hit = (profit > 0).astype(float)
    elif hit_mode == "place":
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

    trivial_oof = np.empty(len(df))
    for b in sorted(set(blocks)):
        train_idx = np.where(blocks != b)[0]
        test_idx = np.where(blocks == b)[0]
        trivial_oof[test_idx] = df["hit"].to_numpy()[train_idx].mean()
    trivial_brier = brier(trivial_oof, df["hit"].to_numpy())
    log(f"  自明な基準(常に学習側平均を予測)のOOF Brier score: {trivial_brier:.4f}")

    raw_brier = brier(df["gap_pct"].to_numpy(), df["hit"].to_numpy())
    log(f"  現行(生gap_pctをそのまま確率として使用)のBrier score: {raw_brier:.4f}(参考、OOFではなくin-sample)")

    cand_results = {}
    for feat in CANDIDATES:
        oof_pred = lobo_bucket_oof(df, feat, K_BUCKETS, blocks)
        b = brier(oof_pred, df["hit"].to_numpy())
        sp = spearman(df[feat].to_numpy(), df["hit"].to_numpy())
        cand_results[feat] = {"oof_brier": b, "spearman_with_hit": sp}
        sign_note = "" if sp >= 0 else "  ※負の相関(値が大きいほど的中率が下がる)"
        log(f"  候補[{feat:10s}] OOF Brier(k={K_BUCKETS}分位較正)={b:.4f}  "
            f"生の値とhitのSpearman相関={sp:+.3f}{sign_note}")

    best_feat, sign_warning = choose_feature_sign_safe(cand_results)
    best_brier = cand_results[best_feat]["oof_brier"]
    beats_trivial = best_brier < trivial_brier
    log(f"\n  採用候補: {best_feat}(OOF Brier={best_brier:.4f} vs 自明な基準{trivial_brier:.4f})"
        f"  符号: {cand_results[best_feat]['spearman_with_hit']:+.3f}")
    if sign_warning:
        log("  ※※ 警告: 正の相関を持つ候補が無かったため、符号を無視してBrier最良を採用 ※※")
    log(f"  自明な基準を上回るか: {'YES' if beats_trivial else 'NO'}")

    final_vals = df[best_feat].to_numpy(dtype=float)
    final_hits = df["hit"].to_numpy(dtype=float)
    edges, bucket_table = build_calibration_table(final_vals, final_hits, K_BUCKETS)
    best_bucket = max_hit_rate_bucket(bucket_table)
    log(f"  実測的中率が最も高いバケット: {best_bucket['hit_rate_pct']}%"
        f"(値域 {best_bucket['value_range']}, n={best_bucket['n_races']})")

    results.setdefault(box_n, {})[hit_mode] = {
        "hit_label": hit_label,
        "n_races": len(df), "overall_hit_rate_pct": round(float(df["hit"].mean()) * 100, 1),
        "trivial_baseline_oof_brier": trivial_brier,
        "raw_gap_pct_insample_brier": raw_brier,
        "candidates": cand_results,
        "chosen_feature": best_feat, "chosen_feature_sign_warning": bool(sign_warning),
        "beats_trivial_baseline": bool(beats_trivial),
        "calibration_table": {
            "feature": best_feat, "n_buckets": K_BUCKETS, "edges": edges, "buckets": bucket_table,
            "best_hit_rate_bucket": best_bucket,
        },
    }

# =======================================================================
# 2) 新規: 段階的的中確率のはしご(5頭確信度〜1頭確信度)
# =======================================================================
log("\n" + "#" * 72)
log("段階的的中確率のはしご(topk_ladder): 上位K頭picksに実際の1着馬が入っていたか")
log("#" * 72)

ladder_w = json.loads((DATA_DIR / LADDER_WEIGHTS_JSON).read_text(encoding="utf-8"))["weights"]
LADDER_W = wvec(ladder_w)

ladder_results = {}
for K in LADDER_KS:
    log("\n" + "-" * 60)
    log(f"K={K}(上位{K}頭に実際の1着馬が入っている確率)")
    log("-" * 60)
    ev = NE.Evaluator(races, actual, box_n=K)
    picks = NE.score_picks(mats, LADDER_W, K)
    st, rt = ev.settler.returns_for(picks)
    win_col = NB.BET_TYPES.index("単勝")
    hit = (rt[:, win_col] > 0).astype(float)

    rows = []
    for i, m in enumerate(mats):
        num, den = m["S"] @ LADDER_W, m["A"] @ LADDER_W
        score = np.where(den > 0, num / den, -1e18)
        order = np.argsort(-score, kind="stable")
        s_sorted = score[order]
        n = len(s_sorted)
        spread = float(s_sorted[0] - s_sorted[-1])
        if n > K and spread > 0:
            gap_boundary_k = float((s_sorted[K - 1] - s_sorted[K]) / spread)
        else:
            gap_boundary_k = 1.0
        gap_top2 = float((s_sorted[0] - s_sorted[1]) / spread) if (n > 1 and spread > 0) else 0.0
        rows.append({
            "race_id": races[i]["race_id"], "field_size": n,
            "gap_boundary_k": gap_boundary_k, "gap_top2": gap_top2, "hit": float(hit[i]),
        })
    df = pd.DataFrame(rows)
    log(f"  対象レース数: {len(df)}  実測hit率(上位{K}頭に1着馬が入っている割合): {df['hit'].mean() * 100:.1f}%")

    trivial_oof = np.empty(len(df))
    for b in sorted(set(blocks)):
        train_idx = np.where(blocks != b)[0]
        test_idx = np.where(blocks == b)[0]
        trivial_oof[test_idx] = df["hit"].to_numpy()[train_idx].mean()
    trivial_brier = brier(trivial_oof, df["hit"].to_numpy())
    log(f"  自明な基準のOOF Brier score: {trivial_brier:.4f}")

    cand_results = {}
    for feat in LADDER_CANDIDATES:
        oof_pred = lobo_bucket_oof(df, feat, K_BUCKETS, blocks)
        b = brier(oof_pred, df["hit"].to_numpy())
        sp = spearman(df[feat].to_numpy(), df["hit"].to_numpy())
        cand_results[feat] = {"oof_brier": b, "spearman_with_hit": sp}
        sign_note = "" if sp >= 0 else "  ※負の相関"
        log(f"  候補[{feat:15s}] OOF Brier={b:.4f}  Spearman相関={sp:+.3f}{sign_note}")

    best_feat, sign_warning = choose_feature_sign_safe(cand_results)
    best_brier = cand_results[best_feat]["oof_brier"]
    beats_trivial = best_brier < trivial_brier
    log(f"  採用候補: {best_feat}(OOF Brier={best_brier:.4f} vs 自明な基準{trivial_brier:.4f})"
        f"  自明な基準を上回るか: {'YES' if beats_trivial else 'NO'}")
    if sign_warning:
        log("  ※※ 警告: 正の相関を持つ候補が無かったため、符号を無視してBrier最良を採用 ※※")

    final_vals = df[best_feat].to_numpy(dtype=float)
    final_hits = df["hit"].to_numpy(dtype=float)
    edges, bucket_table = build_calibration_table(final_vals, final_hits, K_BUCKETS)
    best_bucket = max_hit_rate_bucket(bucket_table)

    ladder_results[K] = {
        "n_races": len(df), "overall_hit_rate_pct": round(float(df["hit"].mean()) * 100, 1),
        "trivial_baseline_oof_brier": trivial_brier,
        "candidates": cand_results,
        "chosen_feature": best_feat, "chosen_feature_sign_warning": bool(sign_warning),
        "beats_trivial_baseline": bool(beats_trivial),
        "calibration_table": {
            "feature": best_feat, "n_buckets": K_BUCKETS, "edges": edges, "buckets": bucket_table,
            "best_hit_rate_bucket": best_bucket,
        },
    }

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "fitted_on": {"n_races": len(races), "dates": data["dates"]},
    "method": "lobo_quantile_bucket_calibration_sign_safe",
    "k_buckets": K_BUCKETS, "candidates_tested": CANDIDATES,
    "results_by_box_n": {str(k): v for k, v in results.items()},
    "topk_ladder": {
        "weights_source": LADDER_WEIGHTS_JSON,
        "hit_definition": "上位K頭picksの中に実際の1着馬が含まれているか(単勝リターン>0)",
        "ks": LADDER_KS,
        "candidates_tested": LADDER_CANDIDATES,
        "results_by_k": {str(k): v for k, v in ladder_results.items()},
    },
    "code_sha256": {
        name: sha256((LIB_DIR / name).read_bytes()).hexdigest()[:16]
        for name in ["nar_signals.py", "nar_eval.py", "nar_backtest.py", "nar_dataset.py",
                     "nar_confidence_calibrate.py"]
    },
    "created_at": datetime.now().isoformat(timespec="seconds"),
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
