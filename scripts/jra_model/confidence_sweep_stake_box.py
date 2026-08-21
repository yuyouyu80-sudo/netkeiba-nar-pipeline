# -*- coding: utf-8 -*-
"""Front3(3): box5/4/3(BOX買い)の確信度ベース・ステーク配分の効果測定(2026-08-21新設)。

既存のconfidence_sweep_axis{5,4,3}.py(軸流し用)と同型の設計を踏襲しつつ、box5/4/3を
1本のスクリプトにまとめる(コピペ3分割はしない、BOX_NSをループする)。3方式を並記する:
  (a) 均等ステーク: 全レース倍率1.0(現状のベースライン)。
  (b) 既存0/1フィルタ: 高確信度Nレース/日(N=5..10)のみ採用、非採用レースは賭けない
      (confidence_sweep_axis{5,4,3}.pyの流儀と同一)。
  (c) 連続乗数: jra_stake_weighting.multiplier_from_rank による日次パーセンタイル順位
      →[0.5,1.5]の連続ステーク乗数(0/1フィルタの一般化)。

決済エンジン(jra_backtest.BoxSettler)・UNIT=100固定は変更しない。乗数は
jra_eval.Evaluator.evaluate/full_table/block_bootstrapに2026-08-21追加した
multipliers/idx引数で事後適用する(数学的に等価)。

対象モデルは現行本番モデル(winner_v3.json/winner_box4.json/winner_box3.json)のみ
(2026-08-21時点でFront1/2の新候補重みはまだ採否未確定のため対象外。採用されたら再実行する)。

出力: data/jra_pipeline/confidence_sweep_stake_box.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_backtest as JB  # noqa: E402
import jra_dataset  # noqa: E402
import jra_eval as JE  # noqa: E402
import jra_signals as JS  # noqa: E402
import jra_stake_weighting as SW  # noqa: E402

BOX_NS = (5, 4, 3)
WINNER_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
N_RANGE = [5, 6, 7, 8, 9, 10]
LO, HI = SW.DEFAULT_LO, SW.DEFAULT_HI
OUT_CSV = DATA_DIR / "confidence_sweep_stake_box.csv"
OUT_JSON = DATA_DIR / "confidence_sweep_stake_box_summary.json"

data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
NAMES = JS.ALL_SIGNALS
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)
date_arr = np.array([r["kaisai_date"] for r in races])


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def block_bootstrap_diff_multipliers(ev: JE.Evaluator, picks: list, mult_a: np.ndarray,
                                     mult_b: np.ndarray, bets=JE.OBJ_BETS, n: int = 2000,
                                     seed: int = 11) -> dict:
    """同一picks・同一レース集合に対し、2種類のステーク乗数(mult_a - mult_b)の差を
    ブロック単位でペアブートストラップする。jra_eval.Evaluator.block_bootstrap_diffは
    「2つの異なるpicks」を同一multipliersで比較する設計のため、「同一picksの異なる
    multipliers」を比較する用途にはそのまま使えない(multipliersは1組しか渡せない)。
    本関数はconfidence_sweep_stake_box.py専用にその逆(picks固定・multipliers可変)を行う。"""
    st, rt = ev.settler.returns_for(picks)
    cols = [JB.BET_TYPES.index(b) for b in bets]
    ma = np.asarray(mult_a, dtype=float)[:, None]
    mb = np.asarray(mult_b, dtype=float)[:, None]
    st_a, rt_a = st * ma, rt * ma
    st_b, rt_b = st * mb, rt * mb
    by_block = {b: np.where(ev.blocks == b)[0] for b in ev.block_ids}
    rng = np.random.default_rng(seed)
    ids = list(ev.block_ids)
    out = np.empty(n)
    for k in range(n):
        chosen = rng.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        sa, ra = st_a[np.ix_(idx, cols)].sum(), rt_a[np.ix_(idx, cols)].sum()
        sb, rb = st_b[np.ix_(idx, cols)].sum(), rt_b[np.ix_(idx, cols)].sum()
        rate_a = ra / sa * 100 if sa else 0.0
        rate_b = rb / sb * 100 if sb else 0.0
        out[k] = rate_a - rate_b
    return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5)), "n_blocks": len(ids)}


def rows_from_table(tbl: pd.DataFrame, model_name: str, scope: str) -> list:
    out = []
    for _, row in tbl.iterrows():
        out.append({
            "model": model_name, "scope": scope, "bet_type": row["bet_type"],
            "races": int(row["races"]), "hit_races": int(row["hit_races"]),
            "hit_rate_pct": row["hit_rate_pct"],
            "total_stake": row["stake"], "total_return": row["return"],
            "return_rate_pct": row["return_rate_pct"],
        })
    return out


all_rows = []
summary_by_box = {}

for BOX_N in BOX_NS:
    print(f"\n{'=' * 60}\nbox_n={BOX_N}\n{'=' * 60}")
    winner = json.loads((DATA_DIR / WINNER_FILES[BOX_N]).read_text(encoding="utf-8"))
    W = wvec(winner["weights"])
    model_label = f"現行box{BOX_N}モデル(pattern{winner['pattern_id']}, {WINNER_FILES[BOX_N]})"

    ev = JE.Evaluator(races, actual, box_n=BOX_N)
    picks = JE.score_picks(mats_all, W, BOX_N)

    # 確信度指標(gap_pct、CONF_N=BOX_N)を日次パーセンタイルの元データとして算出
    # (confidence_sweep_axis{5,4,3}.pyのanalyze_model()と同一ロジック)。
    conf_rows = []
    for i, (r, m) in enumerate(zip(races, mats_all)):
        num, den = m["S"] @ W, m["A"] @ W
        score = np.where(den > 0, num / den, -1e18)
        n = len(r["df"])
        sorted_scores = np.sort(score)[::-1]
        top_score, bottom_score = sorted_scores[0], sorted_scores[-1]
        spread = top_score - bottom_score
        if n > BOX_N:
            gap = sorted_scores[BOX_N - 1] - sorted_scores[BOX_N]
            gap_pct = gap / spread if spread > 0 else 0.0
        else:
            gap_pct = np.inf
        conf_rows.append({"race_idx": i, "kaisai_date": r["kaisai_date"], "gap_pct": gap_pct})
    conf_df = pd.DataFrame(conf_rows)

    # (a) 均等ステーク(ベースライン、全レース)
    tbl_baseline = ev.full_table(picks)
    all_rows += rows_from_table(tbl_baseline, model_label, f"均等ステーク(全{len(races)}レース)")
    r_baseline = ev.evaluate(picks)
    print(f"  均等ステーク: 複勝+ワイド={r_baseline['model']:.2f}%  市場差={r_baseline['excess']:+.2f}pt")

    # (b) 既存0/1フィルタ(高確信度Nレース/日)
    for n_cut in N_RANGE:
        idx_sel = np.where(SW.binary_from_topn(conf_df, n_cut) > 0)[0]
        tbl = ev.full_table(picks, idx=idx_sel)
        all_rows += rows_from_table(
            tbl, model_label, f"0/1フィルタ:高確信度{n_cut}レース/日(計{len(idx_sel)}レース)")

    # (c) 連続乗数(multiplier_from_rank、[0.5,1.5])
    cont_mult = SW.multiplier_from_rank(conf_df, lo=LO, hi=HI)
    tbl_cont = ev.full_table(picks, multipliers=cont_mult)
    all_rows += rows_from_table(
        tbl_cont, model_label, f"連続乗数(スコア差順位→[{LO},{HI}]、全{len(races)}レース)")
    r_cont = ev.evaluate(picks, multipliers=cont_mult)
    print(f"  連続乗数: 複勝+ワイド={r_cont['model']:.2f}%  市場差={r_cont['excess']:+.2f}pt")

    # 均等ステーク vs 連続乗数の点推定95%CI(参考、対応のないCI同士の重複判定は保守的)。
    boot_baseline = ev.block_bootstrap(picks, n=2000, seed=31)
    boot_cont = ev.block_bootstrap(picks, n=2000, seed=31, multipliers=cont_mult)
    print(f"  均等ステーク 95%CI=[{boot_baseline['lo']:.2f}, {boot_baseline['hi']:.2f}]  "
          f"連続乗数 95%CI=[{boot_cont['lo']:.2f}, {boot_cont['hi']:.2f}]")

    # 本命指標: 同一ブロックリサンプルで対応させたペアブートストラップによる
    # (連続乗数 - 均等ステーク)差のCI(block_bootstrap_diff_multipliers、本ファイル冒頭で定義)。
    ones = np.ones(len(races))
    boot_diff_cont_vs_baseline = block_bootstrap_diff_multipliers(ev, picks, cont_mult, ones, seed=71)
    ci = boot_diff_cont_vs_baseline
    print(f"  → 連続乗数−均等ステーク 差の95%CI=[{ci['lo']:+.2f}, {ci['hi']:+.2f}]pt"
          f"(下限が0を超える場合のみ、連続乗数が統計的に均等ステークを上回ると言える)")

    summary_by_box[BOX_N] = {
        "model_file": WINNER_FILES[BOX_N],
        "baseline": {"model": r_baseline["model"], "excess": r_baseline["excess"],
                    "bootstrap_ci": boot_baseline},
        "continuous_multiplier": {"lo": LO, "hi": HI, "model": r_cont["model"],
                                  "excess": r_cont["excess"], "bootstrap_ci": boot_cont},
        "diff_continuous_vs_baseline": ci,
        "decision": "有意差なし(95%CIが0をまたぐ)" if ci["lo"] <= 0 <= ci["hi"] else
                    ("連続乗数が統計的に優位" if ci["lo"] > 0 else "均等ステークが統計的に優位"),
    }

pd.DataFrame(all_rows).to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "obj_bets": JE.OBJ_BETS, "n_range": N_RANGE, "lo": LO, "hi": HI,
    "note": "Front3(3): 均等ステーク/0-1フィルタ/連続乗数の3方式比較。対象は現行本番モデルのみ"
            "(Front1/2の新候補重みが採用された場合は再実行が必要)。",
    "summary_by_box": summary_by_box,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
print(f"\nwrote {OUT_CSV}")
print(f"wrote {OUT_JSON}")
