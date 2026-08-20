# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-20): 「競馬場・クラスによって検証・モデリングを分けると
勝率・回収率は上がらないか」という提案の**事後診断**(独立の重み探索ではない)。

[[project_jra_box4box3_degradation_root_cause_2026_08_12]]で「全クラス区分で例外なく
圧勝している」状態そのものが過学習の兆候と判定された前例があり、NARは競馬場別レース数の
偏りが大きい(盛岡213 〜 船橋60、実測)。競馬場×クラスを同時に(2軸で)割ると1区分の標本が
数レースまで薄くなり、300パターン探索が失敗した「標本が薄いところで最良値を選ぶ」罠に
陥る。そのため本スクリプトは重みを独立探索せず、**現行(等重み)モデルをそのまま**
競馬場軸・クラス軸それぞれ単独で内訳分解し、系統的な差(市場比excessのブロック
ブートストラップCI)があるかどうかだけを検証する。box_n=5(本番の「予想5頭」相当)を対象。

出力: nar_diagnostic_venue_class_2026_08_20_result.json / _report.txt (scratchpad)。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(LIB_DIR))
import nar_backtest as NB  # noqa: E402
import nar_dataset  # noqa: E402
import nar_eval as NE  # noqa: E402
import nar_signals as NS  # noqa: E402

BOX_N = 5
MIN_RACES_FOR_SIGNAL = 100  # プランで定めた「標本が十分」の目安

OUT_DIR = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
OUT_JSON = OUT_DIR / "nar_diagnostic_venue_class_2026_08_20_result.json"
OUT_TXT = OUT_DIR / "nar_diagnostic_venue_class_2026_08_20_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


# クラス序列(nar_signals.CLASS_ORDINAL_NAR)を4段階に粗く束ねる。区間は半開区間
# [lo, hi) なので、ラベルは各区分に実際に含まれる最上位クラスで表記する(2026-08-20
# ゲート1レビューで、当初ラベルが実装と1クラスずれていた指摘を反映して修正済み。
# 例: 旧ラベル「下級(新馬〜C3)」はC3(ord=1.0)が実際には次の区分に入ってしまっていた)。
CLASS_TIERS = [
    ("下級(新馬・未勝利・C4)", 0.0, 1.0),
    ("中級(C3〜B4)", 1.0, 3.5),
    ("中上級(B3〜A3)", 3.5, 6.0),
    ("上級(A2〜重賞)", 6.0, 8.01),
]


def class_tier_of(race_name: str) -> str | None:
    c = NS.class_ordinal(race_name)
    if pd.isna(c):
        return None
    for label, lo, hi in CLASS_TIERS:
        if lo <= c < hi:
            return label
    return None


data = nar_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
priors_all = NS.make_priors(races)
dead = NS.detect_dead(races, priors_all)
alive_base = [n for n in NS.LEGACY_SIGNALS + NS.NEW_SIGNALS if n not in dead]
v2_alive = [n for n in NS.CANDIDATE_SIGNALS_V2 if n not in dead]
v4_alive = [n for n in NS.CANDIDATE_SIGNALS_V4 if n not in dead]
POOL = alive_base + v2_alive + v4_alive
NAMES = NS.ALL_SIGNALS


def equal_w(subset) -> np.ndarray:
    d = {n: 1.0 / len(subset) for n in subset}
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


W_EQUAL = equal_w(POOL)

# 2026-08-20ゲート1(統計学者レビュー)指摘: 競馬場12区分+クラス4区分=16回の個別検定を
# 補正なしに行うと、真の効果がゼロでも約0.8回(=16×0.05)は偶然「有意」と出る。
# Bonferroni補正(有意水準を検定数で割る)した99.7%CIも併記し、「actionable」判定は
# 補正後CIの下限が0を超える場合のみとする(素の95%CIは参考として残す)。
N_TESTS = 12 + 4  # 競馬場12 + クラス階層4


def _block_bootstrap_excess(ev: NE.Evaluator, picks: list, mkt_picks: list, alpha: float,
                            n: int = 2000, seed: int = 17) -> dict:
    """モデルpicksと市場picksの複勝+ワイド回収率の**差**(excess)をブロック単位で
    ブートストラップする。2026-08-21、実行結果を精査して発見・修正したバグへの対応:
    当初はev.block_bootstrap(picks)を直接使っており、これは「回収率そのもの」
    (常に正の値、50〜100%程度)のCIを返すだけで、市場との比較にはなっていなかった。
    そのため回収率が実際には市場を下回る区分まで「lo>0だから有意」と誤判定していた。"""
    st, rt = ev.settler.returns_for(picks)
    mkt_st, mkt_rt = ev.settler.returns_for(mkt_picks)
    cols = [NB.BET_TYPES.index(b) for b in NE.OBJ_BETS]
    by_block = {b: np.where(ev.blocks == b)[0] for b in ev.block_ids}
    rng = np.random.default_rng(seed)
    ids = list(ev.block_ids)
    out = np.empty(n)
    for k in range(n):
        chosen = rng.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        s, r = st[np.ix_(idx, cols)].sum(), rt[np.ix_(idx, cols)].sum()
        ms, mr = mkt_st[np.ix_(idx, cols)].sum(), mkt_rt[np.ix_(idx, cols)].sum()
        v = r / s * 100 if s else 0.0
        mv = mr / ms * 100 if ms else 0.0
        out[k] = v - mv
    return {"mean": float(out.mean()), "lo": float(np.percentile(out, alpha / 2 * 100)),
            "hi": float(np.percentile(out, (1 - alpha / 2) * 100))}


log(f"レース数: {len(races)}  現行(等重み)プール({len(POOL)}シグナル): {POOL}")
log(f"box_n={BOX_N}  標本十分の目安: {MIN_RACES_FOR_SIGNAL}レース以上  "
    f"検定数={N_TESTS}(Bonferroni補正有意水準={0.05 / N_TESTS:.4f})")


def diagnose(label: str, subset_races: list) -> dict:
    n = len(subset_races)
    if n < 5:
        return {"label": label, "n_races": n, "skipped": "標本が5レース未満のため計算省略"}
    mats = NS.signal_matrices(subset_races, priors_all, NAMES)
    ev = NE.Evaluator(subset_races, actual, box_n=BOX_N)
    picks = NE.score_picks(mats, W_EQUAL, BOX_N)
    mkt_picks = NE.market_picks(subset_races, BOX_N)
    r = ev.evaluate(picks)
    n_blocks = len(ev.block_ids)
    if n_blocks < 3:
        boot = {"mean": None, "lo": None, "hi": None, "note": f"ブロック数{n_blocks}<3のためCI省略"}
        boot_bonf = boot
    else:
        boot = _block_bootstrap_excess(ev, picks, mkt_picks, alpha=0.05, n=2000, seed=17)
        boot_bonf = _block_bootstrap_excess(ev, picks, mkt_picks, alpha=0.05 / N_TESTS, n=2000, seed=17)
    sig_beats_market_uncorrected = boot.get("lo") is not None and boot["lo"] > 0
    sig_beats_market_bonferroni = boot_bonf.get("lo") is not None and boot_bonf["lo"] > 0
    enough_sample = n >= MIN_RACES_FOR_SIGNAL
    return {
        "label": label, "n_races": n, "n_blocks": n_blocks,
        "model_pct": r["model"], "market_pct": r["market"], "excess_pt": r["excess"],
        "bootstrap_ci_95": boot, "bootstrap_ci_bonferroni": boot_bonf,
        "significant_vs_market_uncorrected_95pct": bool(sig_beats_market_uncorrected),
        "significant_vs_market_bonferroni": bool(sig_beats_market_bonferroni),
        "enough_sample": bool(enough_sample),
        "actionable": bool(sig_beats_market_bonferroni and enough_sample),
    }


log("\n" + "=" * 72)
log("競馬場軸(単軸)")
log("=" * 72)
by_venue = {}
for venue in sorted({r["racecourse"] for r in races}):
    subset = [r for r in races if r["racecourse"] == venue]
    res = diagnose(venue, subset)
    by_venue[venue] = res
    if "skipped" in res:
        log(f"  {venue}: {res['skipped']}")
    else:
        ci = res["bootstrap_ci_95"]
        ci_bonf = res["bootstrap_ci_bonferroni"]
        ci_txt = f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}]" if ci.get("lo") is not None else ci.get("note", "N/A")
        ci_bonf_txt = (f"[{ci_bonf['lo']:+.2f}, {ci_bonf['hi']:+.2f}]"
                       if ci_bonf.get("lo") is not None else ci_bonf.get("note", "N/A"))
        log(f"  {venue:6s} n={res['n_races']:4d}({res['n_blocks']:3d}ブロック)  "
            f"市場差={res['excess_pt']:+.2f}pt  95%CI={ci_txt}  Bonferroni補正CI={ci_bonf_txt}  "
            f"actionable={'YES' if res['actionable'] else 'no'}")

log("\n" + "=" * 72)
log("クラス軸(単軸)")
log("=" * 72)
by_class = {}
for label, _lo, _hi in CLASS_TIERS:
    subset = [r for r in races if class_tier_of(r["race_name"]) == label]
    res = diagnose(label, subset)
    by_class[label] = res
    if "skipped" in res:
        log(f"  {label}: {res['skipped']}")
    else:
        ci = res["bootstrap_ci_95"]
        ci_bonf = res["bootstrap_ci_bonferroni"]
        ci_txt = f"[{ci['lo']:+.2f}, {ci['hi']:+.2f}]" if ci.get("lo") is not None else ci.get("note", "N/A")
        ci_bonf_txt = (f"[{ci_bonf['lo']:+.2f}, {ci_bonf['hi']:+.2f}]"
                       if ci_bonf.get("lo") is not None else ci_bonf.get("note", "N/A"))
        log(f"  {label:16s} n={res['n_races']:4d}({res['n_blocks']:3d}ブロック)  "
            f"市場差={res['excess_pt']:+.2f}pt  95%CI={ci_txt}  Bonferroni補正CI={ci_bonf_txt}  "
            f"actionable={'YES' if res['actionable'] else 'no'}")

n_actionable_venue = sum(1 for v in by_venue.values() if v.get("actionable"))
n_actionable_class = sum(1 for v in by_class.values() if v.get("actionable"))
log("\n" + "=" * 72)
log("まとめ")
log("=" * 72)
log(f"競馬場軸: {n_actionable_venue}/{len(by_venue)}区分が「Bonferroni補正後も有意差あり・標本十分」")
log(f"クラス軸: {n_actionable_class}/{len(by_class)}区分が「Bonferroni補正後も有意差あり・標本十分」")
log(f"(補正なし95%CIだけで判定すると、{N_TESTS}回の検定のうち真の効果がゼロでも偶然"
    f"約{0.05 * N_TESTS:.1f}回は「有意」と出る計算になるため、actionable判定は必ず"
    "Bonferroni補正後CIを基準にしている)")
log("どちらの軸も0区分なら、現行の全体プールモデルを維持するのが妥当という結論を支持する"
    "(専門家レビュー(フェーズ3)で最終判断)。")

OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races_total": len(races), "box_n": BOX_N, "pool": POOL,
    "min_races_for_signal": MIN_RACES_FOR_SIGNAL,
    "by_venue": by_venue, "by_class": by_class,
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
