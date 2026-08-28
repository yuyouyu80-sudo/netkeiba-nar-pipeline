# -*- coding: utf-8 -*-
"""「33ラップ理論」シグナル(lap33_fit)の回収率(ROI)検証(Part2追補、ユーザー依頼)。

jra_lap33_signal_gate_2026_08_28.py のG1判定は Evaluator.evaluate() の
cost_weighted_rate(複勝+ワイドのΣ払戻/Σ賭金×100)を既に使っており、「市場超過pt」は
的中率の差ではなく最初から回収率(ROI)の差だった。本スクリプトはユーザー依頼
「勝率ではなく回収率として成り立つか」に応え、
  (1) 券種別(単勝〜3連単)の的中率・回収率をbaseline/candidate/市場で内訳表示、
  (2) 主要4券種(単勝/複勝/馬連/ワイド)についてbaseline/candidate単独の絶対回収率の
      95%CIと、控除率ベースの理論ブレークイーブン(JE.breakeven_pct)との比較、
を追加で確認する。新しい採否ゲートを事後的に立てるものではなく、既に確定した不採用判定
(jra_lap33_signal_gate_2026_08_28.py)を回収率の観点から補強する記述的な追補という位置づけ。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import jra_backtest as JB
import jra_dataset
import jra_eval as JE
import jra_lap33_signals as L33
import jra_signals as JS

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_TXT = OUT_DIR / "jra_lap33_signal_roi_2026_08_28_report.txt"

BOX_NS = [4, 5, 3]
POOL_TRUE_PROD = list(JS.LEGACY_SIGNALS)
NAMES = POOL_TRUE_PROD + ["lap33_fit"]
ROI_BETS = ["単勝", "複勝", "馬連", "ワイド"]  # ブートストラップで詳しく見る主要券種

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# ============================================================ データロード(gate scriptと同一)
log("データロード中...")
data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = JS.make_priors([r["df"] for r in races])
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)")

lap33_lookup = L33.load_lap33_lookup()
race_meta = L33.load_race_surface_distance()
history_index = L33.build_history_index()
fit = L33.lap33_fit_matrix(races, history_index, lap33_lookup, race_meta)

mats_all = []
for r in races:
    current_class = JS._class_ordinal(r["race_name"], JS.CLASS_ORDINAL)
    sig = JS.compute_signals(r["df"], current_class, priors_all)
    sig["lap33_fit"] = fit[r["race_id"]]["lap33_fit"]
    S = np.column_stack([sig[n].fillna(0.0).to_numpy(dtype=float) for n in NAMES])
    A = np.column_stack([sig[n].notna().to_numpy(dtype=float) for n in NAMES])
    mats_all.append({"S": S, "A": A})


def equal_w(names_subset) -> np.ndarray:
    d = {n: 1.0 / len(names_subset) for n in names_subset}
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


w_base = equal_w(POOL_TRUE_PROD)
w_cand = equal_w(NAMES)

summary = {}
for box_n in BOX_NS:
    log(f"\n{'=' * 72}\nbox_n={box_n}\n{'=' * 72}")
    ev = JE.Evaluator(races, actual, box_n=box_n)
    picks_base = JE.score_picks(mats_all, w_base, box_n)
    picks_cand = JE.score_picks(mats_all, w_cand, box_n)
    picks_mkt = JE.market_picks(races, box_n)

    full_base = ev.full_table(picks_base).set_index("bet_type")
    full_cand = ev.full_table(picks_cand).set_index("bet_type")
    full_mkt = ev.full_table(picks_mkt).set_index("bet_type")

    log(f"\n-- 券種別 的中率/回収率(N={len(races)}レース) --")
    log(f"{'券種':6s} {'的中率(base/cand/市場)':28s} {'回収率(base/cand/市場)':30s}")
    for bt in JB.BET_TYPES:
        hb, hc, hm = full_base.loc[bt, "hit_rate_pct"], full_cand.loc[bt, "hit_rate_pct"], full_mkt.loc[bt, "hit_rate_pct"]
        rb, rc, rm = full_base.loc[bt, "return_rate_pct"], full_cand.loc[bt, "return_rate_pct"], full_mkt.loc[bt, "return_rate_pct"]
        log(f"{bt:6s} {hb:5.1f}% / {hc:5.1f}% / {hm:5.1f}%      {rb:6.1f}% / {rc:6.1f}% / {rm:6.1f}%")

    log(f"\n-- 主要4券種: 絶対回収率の95%CIと理論ブレークイーブン --")
    roi_rows = {}
    for bt in ROI_BETS:
        be = JE.breakeven_pct(box_n, bets=[bt])
        boot_base = ev.block_bootstrap(picks_base, bets=[bt])
        boot_cand = ev.block_bootstrap(picks_cand, bets=[bt])
        boot_mkt = ev.block_bootstrap(picks_mkt, bets=[bt])
        diff = ev.block_bootstrap_diff(picks_cand, picks_base, bets=[bt])
        base_above_be = boot_base["lo"] > be
        cand_above_be = boot_cand["lo"] > be
        roi_rows[bt] = {
            "breakeven_pct": be,
            "baseline": boot_base, "candidate": boot_cand, "market": boot_mkt,
            "diff_cand_minus_base": diff,
            "baseline_sig_above_breakeven": base_above_be,
            "candidate_sig_above_breakeven": cand_above_be,
        }
        log(f"  [{bt}] 理論BE={be:.1f}%")
        log(f"    baseline : {boot_base['mean']:6.1f}%  95%CI=[{boot_base['lo']:6.1f},{boot_base['hi']:6.1f}]"
            f"  {'BE超え(有意)' if base_above_be else 'BE超え示せず'}")
        log(f"    candidate: {boot_cand['mean']:6.1f}%  95%CI=[{boot_cand['lo']:6.1f},{boot_cand['hi']:6.1f}]"
            f"  {'BE超え(有意)' if cand_above_be else 'BE超え示せず'}")
        log(f"    市場     : {boot_mkt['mean']:6.1f}%  95%CI=[{boot_mkt['lo']:6.1f},{boot_mkt['hi']:6.1f}]")
        log(f"    候補-基準差分95%CI=[{diff['lo']:+.1f},{diff['hi']:+.1f}]")

    summary[f"box{box_n}"] = {
        "full_table_base": full_base.reset_index().to_dict("records"),
        "full_table_cand": full_cand.reset_index().to_dict("records"),
        "full_table_market": full_mkt.reset_index().to_dict("records"),
        "roi_bootstrap": roi_rows,
    }

log(f"\nwrote {OUT_TXT.name}")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")

import json  # noqa: E402
OUT_JSON = OUT_DIR / "jra_lap33_signal_roi_2026_08_28_result.json"
OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
print(f"wrote {OUT_JSON.name}")
