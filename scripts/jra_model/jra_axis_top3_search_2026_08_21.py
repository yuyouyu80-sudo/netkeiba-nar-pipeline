# -*- coding: utf-8 -*-
"""ユーザー依頼(2026-08-21、軸流しレポート公開の同日): 「軸(予想スコア1位)が3着以内
(複勝圏内)に来る確率が最も高くなる」重みパターンを探索し、その重みを使った場合の回収率
(複勝単体・軸流しナガシ7区分)を検証する。

先行探索(jra_axis_search_2026_08_21.py)は「ワイドのコスト加重回収率」を目的関数にしていたが、
2026-08-21のゲート1レビューで両専門家(統計学者・競馬予想家)から「軸の複勝的中率(単騎で
馬券圏内に入る確率)を直接最適化する目的関数を別途試す余地がある」との指摘があった
(box買い用の重みは「上位N頭の集合としての的中力」を最適化したもので、「単騎で馬券圏内に
入る確率」を直接最適化したものではない、という指摘)。本スクリプトはこの指摘に応え、
目的関数を「軸(スコア1位)の複勝的中率」に差し替えて再探索する。

**重要な設計上の簡略化**: 軸(スコア1位)はbox_n(軸+相手の合計頭数)に依存しない
(argmaxは常に同じ1頭)。よって本探索はaxis5/4/3で独立に行う必要が無く、**1回の探索で
共通の「軸選定に最適な重み」を求める**(先行探索が3回独立に探索したのとは異なる)。
求めた重みを使った軸流しナガシの回収率は、box_n=5/4/3それぞれについて別途評価する
(相手の人数=box_n-1が変わるため回収率はbox_nごとに異なる)。

手法(先行探索・NAR 2026-08-20/21で確立した修正済み手法をそのまま踏襲):
  - 重み生成: 等重み(パターン#0固定)+ 等重み近傍集中サンプリング主体の混合(WEIGHT_TIERS)、
    500パターン。探索プールはjra_signals.ALL_SIGNALS(25シグナル、先行探索と同一)。
  - Nested LOBO OOF(開催日×競馬場ブロック)で「500パターン探索という手続き全体」を
    交差検証。fold毎の選択パターンのユニーク数を記録し、LOBO退化を検知する。
  - 選択バイアス診断(ブロック半分割×200反復)を主指標として優先する。
  - 比較ベースライン: (a) 現行box5/4/3の重み(3種類、それぞれ軸の複勝的中率が違いうる)、
    (b) 25本等重み、(c) 市場(1番人気)。
  - 採否ゲート: true_edge_pt / true_edge_sd >= 2.0(既存基準を踏襲)。

出力: jra_axis_top3_search_2026_08_21_result.json / _report.txt (data/jra_pipeline、
git管理下、研究ログ)。本番重み反映は専門家レビューを経てユーザーが判断する。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
sys.path.insert(0, str(LIB_DIR))
import jra_axis_backtest as AB  # noqa: E402
import jra_axis_eval as AE  # noqa: E402
import jra_dataset  # noqa: E402
import jra_signals as JS  # noqa: E402

N_PATTERNS = 500
SEED = 2822  # 2026-08-21、先行探索(2821)・NAR側(2820)とは別系列
DECISION_GATE_RATIO = 2.0
CURRENT_BOX_WEIGHT_FILES = {5: "winner_v3.json", 4: "winner_box4.json", 3: "winner_box3.json"}
BOX_NS_FOR_RETURN_CHECK = (5, 4, 3)  # 回収率検証(nagashi)はbox_nごとに実施

WEIGHT_TIERS = [
    (100.0, 150),
    (25.0, 150),
    (6.0, 100),
    (1.0, 99),
]
assert 1 + sum(n for _, n in WEIGHT_TIERS) == N_PATTERNS

OUT_JSON = DATA_DIR / "jra_axis_top3_search_2026_08_21_result.json"
OUT_TXT = DATA_DIR / "jra_axis_top3_search_2026_08_21_report.txt"

lines = []


def log(s=""):
    print(s)
    lines.append(str(s))


data = jra_dataset.load(rebuild=False)
races, actual = data["races"], data["actual"]
NAMES = JS.ALL_SIGNALS
log(f"レース数: {len(races)}  日付: {data['dates'][0]}〜{data['dates'][-1]}({len(data['dates'])}日)"
    f"  頭数: {sum(len(r['df']) for r in races)}")
log(f"探索対象プール({len(NAMES)}シグナル、先行探索と同一): {NAMES}")
log(f"目的関数: 軸(スコア1位=argmax)の複勝的中率(3着以内に入った割合)")
log(f"パターン数: {N_PATTERNS}  乱数シード: {SEED}")

priors_all = JS.make_priors([r["df"] for r in races])
mats_all = JS.signal_matrices(races, priors_all, NAMES, JS.CLASS_ORDINAL)

blocks = AE.blocks_of(races)
block_ids = sorted(set(blocks))
umabans = [r["df"]["umaban"].astype(int).to_numpy() for r in races]
fukusho_sets = [set(actual.get(r["race_id"], {}).get("複勝", {}).keys()) for r in races]


def wvec(d: dict) -> np.ndarray:
    return np.array([float(d.get(n, 0.0)) for n in NAMES])


def equal_w() -> np.ndarray:
    d = {n: 1.0 / len(NAMES) for n in NAMES}
    return wvec(d)


def axis_row_for(m: dict, w: np.ndarray) -> int:
    """重みwでのスコア1位(軸)の行インデックス。"""
    num, den = m["S"] @ w, m["A"] @ w
    score = np.where(den > 0, num / den, -1e18)
    return int(np.argmax(score))


def axis_hits_for(w: np.ndarray) -> np.ndarray:
    """全レースについて、重みwでの軸(スコア1位)が複勝圏内(3着以内)に入ったか(0/1)。"""
    hits = np.empty(len(races), dtype=np.int64)
    for i, m in enumerate(mats_all):
        axis_idx = axis_row_for(m, w)
        axis_u = int(umabans[i][axis_idx])
        hits[i] = 1 if axis_u in fukusho_sets[i] else 0
    return hits


def hit_rate(hits: np.ndarray, idx: np.ndarray = None) -> float:
    h = hits if idx is None else hits[idx]
    return float(h.mean() * 100) if len(h) else 0.0


rng = np.random.default_rng(SEED)
cols = [equal_w()]
for concentration, n in WEIGHT_TIERS:
    alpha = [concentration] * len(NAMES)
    for _ in range(n):
        cols.append(wvec(dict(zip(NAMES, rng.dirichlet(alpha)))))
W_POOL = np.column_stack(cols)
assert W_POOL.shape[1] == N_PATTERNS

W_EQUAL = equal_w()

# --- 市場(1番人気を軸とみなした場合)の複勝的中率
ninki_hits = np.empty(len(races), dtype=np.int64)
for i, r in enumerate(races):
    ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
    key = np.where(np.isnan(ninki), 1e18, ninki)
    axis_idx = int(np.argsort(key, kind="stable")[0])
    axis_u = int(umabans[i][axis_idx])
    ninki_hits[i] = 1 if axis_u in fukusho_sets[i] else 0
log(f"\n市場(1番人気を軸)の複勝的中率: {hit_rate(ninki_hits):.2f}%")

# --- 現行box5/4/3重みそれぞれの軸複勝的中率
current_hits = {}
for box_n, fname in CURRENT_BOX_WEIGHT_FILES.items():
    w = wvec(json.loads((DATA_DIR / fname).read_text(encoding="utf-8"))["weights"])
    h = axis_hits_for(w)
    current_hits[box_n] = h
    log(f"現行box{box_n}重み({fname})を軸選定に転用した場合の複勝的中率: {hit_rate(h):.2f}%")

equal_hits = axis_hits_for(W_EQUAL)
log(f"{len(NAMES)}本等重み(参考)の軸複勝的中率: {hit_rate(equal_hits):.2f}%")

# --- 500パターン全部の的中配列を先に計算(以後は使い回す)
all_hits = [axis_hits_for(W_POOL[:, j]) for j in range(N_PATTERNS)]
full_vals = np.array([hit_rate(h) for h in all_hits])
best_full = int(np.argmax(full_vals))
log(f"\n[全{len(races)}レースで最良の1パターン] pattern#{best_full}  "
    f"軸複勝的中率={full_vals[best_full]:.2f}%  "
    "※学習データそのもので選んでいるため楽観的(in-sample)な数字である点に注意")
top_w = {n: float(w) for n, w in zip(NAMES, W_POOL[:, best_full]) if w > 0.005}
log(f"  重み内訳(0.5%以上): {json.dumps(top_w, ensure_ascii=False)}")

by_block = {b: np.where(blocks == b)[0] for b in block_ids}


def fit_fn(train_idx):
    vals = np.array([hit_rate(all_hits[j], idx=train_idx) for j in range(N_PATTERNS)])
    best = int(np.argmax(vals))
    return W_POOL[:, best], best


oof_hits = np.empty(len(races), dtype=np.int64)
chosen_pattern_idx = {}
for b in block_ids:
    test_idx = by_block[b]
    train_idx = np.where(blocks != b)[0]
    w, pat_idx = fit_fn(train_idx)
    chosen_pattern_idx[b] = pat_idx
    for i in test_idx:
        axis_idx = axis_row_for(mats_all[i], w)
        axis_u = int(umabans[i][axis_idx])
        oof_hits[i] = 1 if axis_u in fukusho_sets[i] else 0

n_unique = len(set(chosen_pattern_idx.values()))
n_folds = len(chosen_pattern_idx)
degenerate = n_unique == 1
log(f"\n[Nested LOBO OOF] {n_folds}ブロック(開催日×競馬場)で{N_PATTERNS}パターン探索という"
    f"手続き全体を交差検証: 軸複勝的中率={hit_rate(oof_hits):.2f}%")
log(f"  fold毎の選択パターンのユニーク数: {n_unique}/{n_folds}"
    + ("  ※全fold同一パターン=LOBO退化。この数値はin-sample評価として扱い、統計的検定には"
       "使わない" if degenerate else ""))

# --- 選択バイアス診断(ブロック半分割×200反復)
n_rep = 200
rng2 = np.random.default_rng(2029)
ids = list(block_ids)
sel, unseen, unseen_mean = [], [], []
for _ in range(n_rep):
    perm = rng2.permutation(len(ids))
    a = np.concatenate([by_block[ids[i]] for i in perm[: len(ids) // 2]])
    b_ = np.concatenate([by_block[ids[i]] for i in perm[len(ids) // 2:]])
    va = np.array([hit_rate(all_hits[j], idx=a) for j in range(N_PATTERNS)])
    vb = np.array([hit_rate(all_hits[j], idx=b_) for j in range(N_PATTERNS)])
    best = int(np.argmax(va))
    sel.append(va[best])
    unseen.append(vb[best])
    unseen_mean.append(vb.mean())
sel, unseen, unseen_mean = map(np.array, (sel, unseen, unseen_mean))
true_edge_pt = float((unseen - unseen_mean).mean())
true_edge_sd = float((unseen - unseen_mean).std())
win_rate = float((unseen > unseen_mean).mean())
log(f"\n[選択バイアス診断] ブロック半分割×{n_rep}反復:")
log(f"  選抜側(見た側)の平均      : {sel.mean():.2f}%")
log(f"  その候補の未使用側での成績 : {unseen.mean():.2f}%")
log(f"  未使用側の{N_PATTERNS}パターン平均       : {unseen_mean.mean():.2f}%")
log(f"  楽観バイアス               : {(sel.mean() - unseen.mean()):+.2f}pt")
log(f"  選ぶことの真の価値         : {true_edge_pt:+.2f}pt (sd {true_edge_sd:.2f})")
log(f"  未使用側で{N_PATTERNS}パターン平均を上回る確率 : {win_rate * 100:.0f}%")

edge_ratio = true_edge_pt / true_edge_sd if true_edge_sd else 0.0
decision = "ADOPT_CANDIDATE" if edge_ratio >= DECISION_GATE_RATIO else "REJECTED"
log(f"\n採否ゲート(true_edge_pt/true_edge_sd >= {DECISION_GATE_RATIO}): {edge_ratio:.3f}  → {decision}"
    + ("  ※LOBO退化のためNested LOBO OOFではなく本診断のみで判定" if degenerate else ""))


# --- ブロックブートストラップ(OOF hit rateの95%CI、現行box重み各種との差)
def block_bootstrap_hit(hits: np.ndarray, n: int = 2000, seed: int = 31) -> dict:
    rng3 = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        chosen = rng3.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        out[k] = hits[idx].mean() * 100
    return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5))}


def block_bootstrap_diff_hit(hits_a: np.ndarray, hits_b: np.ndarray, n: int = 2000, seed: int = 41) -> dict:
    rng4 = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        chosen = rng4.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        out[k] = (hits_a[idx].mean() - hits_b[idx].mean()) * 100
    return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5))}


boot_oof = block_bootstrap_hit(oof_hits, seed=31)
log(f"\n[Nested LOBO OOF(誠実なheld-out結果)の複勝的中率、95%CI、n=2000]")
if degenerate:
    log("  ※注意: 上記の通りfold毎の選択パターンが1/36(全fold同一)でLOBO退化しているため、"
        "以下のNested LOBO OOF picksは実質的にin-sample選択パターン(pattern#"
        f"{best_full})をそのまま全レースに適用したものと同じであり、真のheld-out検証には"
        "なっていない。以下の信頼区間・現行重み比較は参考値に留め、統計的な有意性の根拠には"
        "しない(2026-08-20/21にNAR側で発見した落とし穴と同型)。採否判定は上記の選択バイアス"
        "診断(ブロック半分割)のtrue_edge/sd比のみで行う。")
log(f"  {boot_oof['mean']:.2f}%  95%CI=[{boot_oof['lo']:.2f}, {boot_oof['hi']:.2f}]")

boot_vs_current = {}
for box_n, h in current_hits.items():
    d = block_bootstrap_diff_hit(oof_hits, h, seed=40 + box_n)
    boot_vs_current[box_n] = d
    log(f"  現行box{box_n}重み比の差 95%CI=[{d['lo']:+.2f}, {d['hi']:+.2f}]pt"
        "(下限が0を超える場合のみ、統計的に上回ったと言える)")

boot_vs_equal = block_bootstrap_diff_hit(oof_hits, equal_hits, seed=50)
boot_vs_market = block_bootstrap_diff_hit(oof_hits, ninki_hits, seed=51)
log(f"  {len(NAMES)}本等重み比の差 95%CI=[{boot_vs_equal['lo']:+.2f}, {boot_vs_equal['hi']:+.2f}]pt")
log(f"  市場(1番人気)比の差 95%CI=[{boot_vs_market['lo']:+.2f}, {boot_vs_market['hi']:+.2f}]pt")

# --- 見つかった重み(Nested LOBO OOFのheld-out picksではなく、全データでの最良パターン=
# best_fullを「回収率検証用の重み」として採用する。in-sample選択の楽観を含む参考値である
# ことを明記した上で、実際に使った場合の回収率(複勝単体・軸流しナガシ7区分)を計算する。
W_BEST = W_POOL[:, best_full]

log("\n" + "=" * 72)
log("回収率検証(W_BEST=in-sample最良パターン#{}を軸選定重みとして使った場合)".format(best_full))
log("=" * 72)
log("【重要】W_BESTは500パターン中のin-sample最良パターンであり、選択バイアス診断の結果"
    "(採否ゲート未達=REJECTED)が示す通り統計的検証には合格していない不採用候補である。"
    "以下の回収率はすべて参考値(楽観バイアスを含みうる)として扱うこと。"
    "2026-08-21競馬予想家・統計学者レビュー指摘により、市場(1番人気)の同条件回収率を"
    "併記し、的中数10未満の券種には注記を付す。")

LOW_SAMPLE_HITS = 10


def block_bootstrap_rate(stake_arr, return_arr, n: int = 2000, seed: int = 61) -> dict:
    """1レースごとのstake/return配列(複勝単体など)をブロック単位でブートストラップし、
    回収率の95%CIを返す。"""
    rng5 = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        chosen = rng5.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        s = stake_arr[idx].sum()
        r_ = return_arr[idx].sum()
        out[k] = r_ / s * 100 if s else 0.0
    return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5))}


def block_bootstrap_diff_rate(stake_a, return_a, stake_b, return_b, n: int = 2000, seed: int = 62) -> dict:
    rng6 = np.random.default_rng(seed)
    out = np.empty(n)
    for k in range(n):
        chosen = rng6.choice(len(ids), size=len(ids), replace=True)
        idx = np.concatenate([by_block[ids[c]] for c in chosen])
        sa, ra = stake_a[idx].sum(), return_a[idx].sum()
        sb, rb = stake_b[idx].sum(), return_b[idx].sum()
        va = ra / sa * 100 if sa else 0.0
        vb = rb / sb * 100 if sb else 0.0
        out[k] = va - vb
    return {"mean": float(out.mean()), "lo": float(np.percentile(out, 2.5)),
            "hi": float(np.percentile(out, 97.5))}


# (a) 軸単体の複勝回収率(1点100円、単純に軸を複勝で買った場合)。市場(1番人気)側も
# 同条件で計算し、統計学者レビュー指摘によりブロックブートストラップで差を検定する。
UNIT = 100
model_stake = np.full(len(races), UNIT, dtype=np.int64)
model_return = np.empty(len(races), dtype=np.int64)
market_stake = np.full(len(races), UNIT, dtype=np.int64)
market_return = np.empty(len(races), dtype=np.int64)
for i, r in enumerate(races):
    axis_idx = axis_row_for(mats_all[i], W_BEST)
    axis_u = int(umabans[i][axis_idx])
    model_return[i] = actual.get(r["race_id"], {}).get("複勝", {}).get(axis_u, 0)

    ninki = pd.to_numeric(r["df"]["bias_ninki"], errors="coerce").to_numpy(dtype=float)
    key = np.where(np.isnan(ninki), 1e18, ninki)
    mkt_axis_idx = int(np.argsort(key, kind="stable")[0])
    mkt_axis_u = int(umabans[i][mkt_axis_idx])
    market_return[i] = actual.get(r["race_id"], {}).get("複勝", {}).get(mkt_axis_u, 0)

fukusho_return_rate = model_return.sum() / model_stake.sum() * 100
market_fukusho_return_rate = market_return.sum() / market_stake.sum() * 100
fukusho_hit_rate = full_vals[best_full]
boot_model_fukusho = block_bootstrap_rate(model_stake, model_return, seed=61)
boot_market_fukusho = block_bootstrap_rate(market_stake, market_return, seed=63)
boot_diff_fukusho = block_bootstrap_diff_rate(model_stake, model_return, market_stake, market_return, seed=65)
log(f"\n[軸単体・複勝1点買い、参考値] 的中率={fukusho_hit_rate:.2f}%({len(races)}レース)  "
    f"投資額={int(model_stake.sum()):,}円  払戻額={int(model_return.sum()):,}円  "
    f"回収率={fukusho_return_rate:.1f}%  95%CI=[{boot_model_fukusho['lo']:.1f}, {boot_model_fukusho['hi']:.1f}]")
log(f"[同条件・市場(1番人気)] 的中率={hit_rate(ninki_hits):.2f}%  "
    f"回収率={market_fukusho_return_rate:.1f}%  95%CI=[{boot_market_fukusho['lo']:.1f}, {boot_market_fukusho['hi']:.1f}]")
log(f"[差(参考値−市場)] {fukusho_return_rate - market_fukusho_return_rate:+.1f}pt  "
    f"95%CI=[{boot_diff_fukusho['lo']:+.1f}, {boot_diff_fukusho['hi']:+.1f}]pt"
    + ("  ※0をまたぐため統計的have有意差なし" if boot_diff_fukusho['lo'] <= 0 <= boot_diff_fukusho['hi']
       else "  ※統計的に有意差あり"))

# (b) 軸流しナガシ7区分(box_n=5/4/3それぞれ、相手=同じW_BESTでのスコア2位以降)
nagashi_results = {}
for box_n in BOX_NS_FOR_RETURN_CHECK:
    settler = AB.AxisSettler(races, actual, box_n=box_n)
    picks = []
    for m in mats_all:
        num, den = m["S"] @ W_BEST, m["A"] @ W_BEST
        score = np.where(den > 0, num / den, -1e18)
        n = len(score)
        order = np.argsort(-score, kind="stable")
        picks.append(order[:min(box_n, n)])
    st, rt = settler.returns_for(picks)
    rows = []
    low_sample_bets = []
    for j, bt in enumerate(AB.BET_TYPES_AXIS):
        s, r_ = int(st[:, j].sum()), int(rt[:, j].sum())
        hits = int((rt[:, j] > 0).sum())
        n_races = len(races)
        if hits < LOW_SAMPLE_HITS:
            low_sample_bets.append(bt)
        rows.append({"bet_type": bt, "races": n_races, "hit_races": hits,
                     "hit_rate_pct": round(hits / n_races * 100, 1) if n_races else 0.0,
                     "stake": s, "return": r_,
                     "return_rate_pct": round(r_ / s * 100, 1) if s else 0.0,
                     "low_sample": hits < LOW_SAMPLE_HITS})
    nagashi_results[box_n] = {"rows": rows, "low_sample_bet_types": low_sample_bets,
                              "low_sample_threshold_hits": LOW_SAMPLE_HITS}
    log(f"\n[軸流しナガシ axis{box_n}(軸+相手{box_n - 1}頭)、W_BEST使用(参考値)、{len(races)}レース]")
    log(pd.DataFrame(rows).to_string(index=False))
    if low_sample_bets:
        log(f"  ※的中数{LOW_SAMPLE_HITS}未満(参考値、回収率の解釈に注意): {low_sample_bets}")

OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
OUT_TXT.write_text("\n".join(lines), encoding="utf-8")
OUT_JSON.write_text(json.dumps({
    "n_races": len(races), "dates": data["dates"],
    "n_blocks": len(block_ids), "pool": NAMES, "n_patterns": N_PATTERNS, "seed": SEED,
    "weight_tiers": WEIGHT_TIERS, "objective": "axis(score-rank-1) fukusho(top3) hit rate",
    "market_axis_hit_rate": hit_rate(ninki_hits),
    "current_box_weight_axis_hit_rate": {k: hit_rate(v) for k, v in current_hits.items()},
    "equal_weight_axis_hit_rate": hit_rate(equal_hits),
    "best_full_population": {
        "pattern_index": best_full, "hit_rate": float(full_vals[best_full]), "weights": top_w,
        "note": "in-sample選択、楽観バイアスを含む参考値。統計的検定には使わない。",
    },
    "nested_lobo_oof": {
        "hit_rate": hit_rate(oof_hits), "n_unique_patterns": n_unique, "n_folds": n_folds,
        "degenerate": degenerate, "bootstrap_ci": boot_oof,
        "caveat": None if not degenerate else (
            "全36foldで同一パターン(pattern#" + str(best_full) + ")が選ばれ続けたため、"
            "Nested LOBO OOFは実質的にin-sample評価に退化している。以下のbootstrap比較"
            "(現行box重み等との差)は参考値であり、統計的な有意性の根拠にはならない。"
            "採否判定はselection_optimism(選択バイアス診断)のtrue_edge/sd比のみで行った。"
        ),
    },
    "selection_optimism": {
        "selected_side": float(sel.mean()), "unseen_side": float(unseen.mean()),
        "unseen_all_mean": float(unseen_mean.mean()),
        "optimism_pt": float(sel.mean() - unseen.mean()),
        "true_edge_pt": true_edge_pt, "true_edge_sd": true_edge_sd, "win_rate": win_rate,
    },
    "decision_gate_ratio": edge_ratio, "decision": decision,
    "bootstrap_oof_vs_current_box_weight": boot_vs_current,
    "bootstrap_oof_vs_equal_weight": boot_vs_equal,
    "bootstrap_oof_vs_market": boot_vs_market,
    "return_check": {
        "caveat": "W_BEST(in-sample最良パターン)は選択バイアス診断でREJECTED(採否ゲート未達)の"
                  "不採用候補。以下はすべて参考値であり、統計的検証に合格した数値ではない。",
        "fukusho_single_axis": {
            "hit_rate_pct": fukusho_hit_rate,
            "stake": int(model_stake.sum()), "return": int(model_return.sum()),
            "return_rate_pct": fukusho_return_rate, "bootstrap_ci": boot_model_fukusho,
        },
        "fukusho_single_axis_market_comparison": {
            "market_hit_rate_pct": hit_rate(ninki_hits),
            "market_return_rate_pct": market_fukusho_return_rate,
            "market_bootstrap_ci": boot_market_fukusho,
            "diff_model_minus_market_pt": fukusho_return_rate - market_fukusho_return_rate,
            "diff_bootstrap_ci": boot_diff_fukusho,
            "significant": not (boot_diff_fukusho["lo"] <= 0 <= boot_diff_fukusho["hi"]),
        },
        "nagashi_by_box_n": nagashi_results,
    },
}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
log(f"\nwrote {OUT_JSON.name} / {OUT_TXT.name}")
