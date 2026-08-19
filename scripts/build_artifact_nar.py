# -*- coding: utf-8 -*-
"""地方競馬(NAR)版の着順予想レポート(build_artifact.py のJRA版を移植)。

JRA版との違い:
  - モデルは2種(通常戦=pattern29のNAR専用重み、新馬戦=JRAのwinner_shinba.json暫定流用)。
    未勝利戦モデルは対象外(地方競馬に構造的に存在しないため、ユーザー決定によりスコープ外)。
  - 対象日付はdata/nar_pipeline/race_names_nar_*.csvから自動検出する(JRAのように日付を
    ハードコードしない - NARはまだデータ収集中で日付が増え続けるため)。
  - confidence_sweep(7段階スコープタブ)・オッズスパークライン(watch_odds.py未対応)は
    対象外。BOX回収率は単一テーブル表示のみ。
  - waku(枠番)列は地方競馬の馬柱データに存在しない(厩舎コメント欄由来)ため、pick-table上
    は常に「-」表示になる。枠連のBOX回収率も検証対象外(常に0)。lede/methodに明記する。
  - 金沢の一部レースで「枠単」が「枠連」として誤ラベル付けされたpayout行が存在するため、
    combinationに「→」を含む行は_parse_comboで除外する(search_patterns_nar.py等と同じ対処)。
"""
import html
import itertools
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "common"))
import confidence_calibrate as CC  # noqa: E402
DATA_DIR = PROJECT_ROOT / "data" / "nar_pipeline"
NAR_DATA = PROJECT_ROOT / "data"

# --- 日付マニフェスト(ハードコードせず自動検出) ---
NAR_DATES = sorted(p.stem.replace("race_names_nar_", "") for p in DATA_DIR.glob("race_names_nar_*.csv"))
DATE_MANIFEST = []
for _d in NAR_DATES:
    DATE_MANIFEST.append({
        "date": _d,
        "newspaper": True,  # race_names_nar_{d}.csvが存在する時点でnewspaper取得済み
        "results": (NAR_DATA / "race_results" / "nar" / "2026" / f"{_d}.csv").exists(),
        "payouts": (NAR_DATA / "payouts" / "nar" / "2026" / f"{_d}.csv").exists(),
    })
for _entry in DATE_MANIFEST:
    _entry["status"] = "verified" if _entry["results"] and _entry["payouts"] else "pending"
    _entry["predicted"] = (DATA_DIR / f"predictions_nar_{_entry['date']}.csv").exists()

WEEKDAY_JP = ["月", "火", "水", "木", "金", "土", "日"]


def format_date_label(date_key: str) -> tuple[str, str]:
    d = datetime.strptime(date_key, "%Y%m%d")
    short = f"{d.month}/{d.day}"
    full = f"{d.month}月{d.day}日({WEEKDAY_JP[d.weekday()]})"
    return short, full


def format_lede_dates(dates: list[str]) -> str:
    if not dates:
        return ""
    parsed = [datetime.strptime(d, "%Y%m%d") for d in dates]
    first = parsed[0]
    if all(p.year == first.year and p.month == first.month for p in parsed):
        days = "・".join(f"{p.day}日" for p in parsed)
        return f"{first.year}年{first.month}月{days}"
    return "・".join(f"{p.year}年{p.month}月{p.day}日" for p in parsed)


SURFACE_MAP = {"芝": "芝", "ダ": "ダート"}


def esc(s) -> str:
    return html.escape(str(s)) if pd.notna(s) else ""


def _load_meta_verified(date_key: str) -> pd.DataFrame:
    df = pd.read_csv(NAR_DATA / "race_results" / "nar" / "2026" / f"{date_key}.csv", dtype=str)
    return df[["race_id", "race_number", "surface", "distance_m", "going", "weather", "start_time"]].drop_duplicates()


def _load_meta_pending(date_key: str) -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / f"race_names_nar_{date_key}.csv", dtype=str)
    m = df[["race_id", "race_number", "surface", "distance_m", "start_time"]].drop_duplicates().copy()
    m["going"] = pd.NA
    m["weather"] = pd.NA
    return m


verified_dates = [e["date"] for e in DATE_MANIFEST if e["status"] == "verified"]
pending_predictable_dates = [
    e["date"] for e in DATE_MANIFEST
    if e["status"] == "pending" and e["newspaper"] and (DATA_DIR / f"predictions_nar_{e['date']}.csv").exists()
]


def _load_pred(path: Path, model: str) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str)
    df["model"] = model
    return df


# --- 予想の読み込み(通常戦=日付ごとのpredictions_nar_{date}.csv、新馬戦=単一ファイル) ---
pred_frames = []
for _e in DATE_MANIFEST:
    _p = DATA_DIR / f"predictions_nar_{_e['date']}.csv"
    if _p.exists():
        pred_frames.append(_load_pred(_p, "pattern29"))
_shinba_path = DATA_DIR / "predictions_shinba_nar.csv"
if _shinba_path.exists():
    pred_frames.append(_load_pred(_shinba_path, "shinba"))
pred = pd.concat(pred_frames, ignore_index=True)
pred["_score"] = pd.to_numeric(pred["_score"], errors="coerce")
pred["bias_ninki"] = pd.to_numeric(pred["bias_ninki"], errors="coerce")
pred["bias_win_odds"] = pd.to_numeric(pred["bias_win_odds"], errors="coerce")

_dupe_check = pred.groupby("race_id")["model"].nunique()
_dupes = _dupe_check[_dupe_check > 1]
assert _dupes.empty, f"race_id が複数モデルにまたがっています: {_dupes.index.tolist()}"
_intra_model_dupes = pred[pred.duplicated(subset=["race_id", "pred_rank"], keep=False)]
assert _intra_model_dupes.empty, (
    f"同一race_id×pred_rankの重複行があります: {sorted(_intra_model_dupes['race_id'].unique().tolist())}"
)

meta_frames = [_load_meta_verified(d) for d in verified_dates]
meta_frames += [_load_meta_pending(d) for d in pending_predictable_dates]
meta = pd.concat(meta_frames, ignore_index=True).set_index("race_id")

total_races_scored = pred["race_id"].nunique()
total_races_verified = pred[pred["kaisai_date"].isin(verified_dates)]["race_id"].nunique()

# --- 実績データ(検証済み日付のみ): 着順・複勝払戻・BOX組合せ払戻 ---
COMBO_BET_TYPES = ["馬連", "ワイド", "馬単", "3連複", "3連単"]
ALL_BET_TYPES = ["単勝", "複勝"] + COMBO_BET_TYPES

_result_frames = [
    pd.read_csv(NAR_DATA / "race_results" / "nar" / "2026" / f"{d}.csv", dtype=str)[
        ["race_id", "umaban", "finish_pos"]
    ]
    for d in verified_dates
]
finish_df = pd.concat(_result_frames, ignore_index=True) if _result_frames else pd.DataFrame(
    columns=["race_id", "umaban", "finish_pos"]
)
FINISH_MAP = {(rid, uma): fp for rid, uma, fp in zip(finish_df["race_id"], finish_df["umaban"], finish_df["finish_pos"])}
FINISH_DNF_LABEL = {"取": "取消", "中": "中止", "除": "除外"}


def finish_label(race_id: str, umaban: str) -> str | None:
    fp = FINISH_MAP.get((race_id, str(umaban)))
    if fp is None:
        return None
    return FINISH_DNF_LABEL.get(fp, f"{fp}着")


def _parse_combo(bet_type: str, combo_text: str):
    if bet_type in ("単勝", "複勝"):
        return int(combo_text)
    if bet_type in ("馬単", "3連単"):
        return tuple(int(x) for x in combo_text.split("→"))
    # 金沢の一部レース(202646072603-607)で「枠単」が「枠連」に誤ラベル付けされたまま
    # combinationだけ「→」区切りで保存されている実データ不整合への対処(NAR共通スクリプトと同じ)。
    if "→" in combo_text:
        return None
    return frozenset(int(x) for x in combo_text.split("-"))


_payout_frames = [
    pd.read_csv(NAR_DATA / "payouts" / "nar" / "2026" / f"{d}.csv", dtype=str) for d in verified_dates
]
payouts = pd.concat(_payout_frames, ignore_index=True) if _payout_frames else pd.DataFrame(
    columns=["race_id", "bet_type", "combination", "payout"]
)
if not payouts.empty:
    payouts["payout"] = payouts["payout"].astype(int)

# --- 簡易結果(2026-07-30、nar.netkeiba.com/race/result.htmlから発走当日中に取得できるもの)。
#     正式な確定データ(db.netkeiba.com、ACTUAL_MAPS)は翌日反映のため、それより前に
#     発走済みレースの1〜3着・単勝払戻だけを速報として出す(検証・回収率計算には使わない)。
QUICK_RESULT_MAP: dict[str, dict] = {}
for _qr_path in sorted(DATA_DIR.glob("quick_result_nar_*.csv")):
    _qr_df = pd.read_csv(_qr_path, dtype=str)
    for _row in _qr_df.to_dict("records"):
        QUICK_RESULT_MAP[_row["race_id"]] = _row


def quick_result_html(race_id: str) -> str:
    r = QUICK_RESULT_MAP.get(race_id)
    if r is None or not r.get("finish1_horse"):
        return ""
    parts = [
        f'1着 {esc(r["finish1_umaban"])} {esc(r["finish1_horse"])}',
        f'2着 {esc(r["finish2_umaban"])} {esc(r["finish2_horse"])}',
        f'3着 {esc(r["finish3_umaban"])} {esc(r["finish3_horse"])}',
    ]
    payout = f'単勝{esc(r["tansho_payout"])}({esc(r["tansho_ninki"])})' if r.get("tansho_payout") else ""
    return (
        '<div class="quick-result"><span class="quick-result-label">結果速報</span>'
        f'<span class="quick-result-body">{" / ".join(parts)}{" ・ " + payout if payout else ""}</span></div>'
    )


ACTUAL_MAPS: dict[str, dict[str, dict]] = {}
for _race_id, _g in payouts.groupby("race_id"):
    _per_bt = {}
    for _bt in ALL_BET_TYPES:
        _rows = _g[_g["bet_type"] == _bt]
        _m: dict = {}
        for _c, _p in zip(_rows["combination"], _rows["payout"]):
            _key = _parse_combo(_bt, _c)
            if _key is None:
                continue
            _m[_key] = _m.get(_key, 0) + _p
        _per_bt[_bt] = _m
    ACTUAL_MAPS[_race_id] = _per_bt


def place_payout(race_id: str, umaban: str) -> int:
    if race_id not in ACTUAL_MAPS:
        return 0
    try:
        key = int(umaban)
    except (ValueError, TypeError):
        return 0
    return ACTUAL_MAPS[race_id]["複勝"].get(key, 0)


_COMBO_BUILDERS = {
    "馬連": lambda u: [frozenset(c) for c in itertools.combinations(u, 2)],
    "ワイド": lambda u: [frozenset(c) for c in itertools.combinations(u, 2)],
    "馬単": lambda u: list(itertools.permutations(u, 2)),
    "3連複": lambda u: [frozenset(c) for c in itertools.combinations(u, 3)],
    "3連単": lambda u: list(itertools.permutations(u, 3)),
}


def _fmt_combo(bet_type: str, key) -> str:
    if bet_type in ("馬単", "3連単"):
        return "→".join(str(x) for x in key)
    return "-".join(str(x) for x in sorted(key))


def box_actual_results(race_id: str, umabans: list[int]) -> list[dict]:
    if race_id not in ACTUAL_MAPS:
        return []
    actual = ACTUAL_MAPS[race_id]
    out = []
    for bt in COMBO_BET_TYPES:
        hits = [(c, actual[bt].get(c, 0)) for c in _COMBO_BUILDERS[bt](umabans)]
        hits = [(c, amt) for c, amt in hits if amt > 0]
        if hits:
            for combo, amt in hits:
                out.append({"bet_type": bt, "hit": True, "label": _fmt_combo(bt, combo), "amount": amt})
        else:
            out.append({"bet_type": bt, "hit": False, "label": "", "amount": 0})
    return out


# --- 確信度較正(2026-07-30、nar_confidence_calibrate.py。CONF_MAPより先に読み込む必要あり) ---
_confidence_calib_path = DATA_DIR / "confidence_calibration_nar.json"
CONFIDENCE_CALIB = (
    json.loads(_confidence_calib_path.read_text(encoding="utf-8"))
    if _confidence_calib_path.exists() else {}
)

# --- レース単位の確信度(2026-08-12、JRA/NAR確信度統一。単勝ベースtopk_ladderのみに
#     統一し、box_n×place/profit較正(confidence_calibrated_pct)は廃止した。バッジは
#     ladder_conf_5_pctが統計的に有意(show_pct=True)なら較正済み%、そうでなければ
#     JRAと同じ3段階(高/中/低確信度)フォールバックにする) ---
conf_race = pd.read_csv(DATA_DIR / "confidence_per_race_nar.csv", dtype=str)
conf_race["gap_pct_5"] = pd.to_numeric(conf_race["gap_pct_5"])
conf_race["gap_boundary_5"] = pd.to_numeric(conf_race.get("gap_boundary_5"))
conf_race["ladder_conf_5_pct"] = pd.to_numeric(conf_race.get("ladder_conf_5_pct"))
conf_race["ladder_conf_5_show_pct"] = conf_race.get("ladder_conf_5_show_pct") == "True"
conf_race["selected_5"] = conf_race["selected_5"] == "True"
LADDER_KS = [5, 4, 3, 2, 1]
for _k in LADDER_KS:
    conf_race[f"ladder_conf_{_k}_pct"] = pd.to_numeric(conf_race.get(f"ladder_conf_{_k}_pct"))

# 較正済み%を出さないレース向けの3段階しきい値(JRA側と同じ発想: 較正を主張せず、
# 検証済みレース群の中でのgap_boundary_5の相対位置=三分位だけを見る)。
_gap5_pool = conf_race["gap_boundary_5"].dropna()
_gap5_pool = _gap5_pool[~_gap5_pool.isin([float("inf")])]
_GAP5_TERCILES = (
    tuple(_gap5_pool.quantile([1 / 3, 2 / 3])) if len(_gap5_pool) >= 3 else (0.0, 1.0)
)

CONF_MAP = {
    rid: (row["gap_pct_5"], row["gap_boundary_5"], row["ladder_conf_5_pct"],
         row["ladder_conf_5_show_pct"], row["selected_5"])
    for rid, row in zip(conf_race["race_id"], conf_race.to_dict("records"))
}
LADDER_MAP = {
    rid: tuple(row[f"ladder_conf_{k}_pct"] for k in LADDER_KS)
    for rid, row in zip(conf_race["race_id"], conf_race.to_dict("records"))
}

# 同じK(頭数)同士で比較した順位分け(2026-07-30追加)。上位20%=赤、20〜50%=黄、
# それ以外は無着色。K列ごとに全レースを通した相対順位で決めるため、日付や場は区別しない。
for _k in LADDER_KS:
    conf_race[f"ladder_tier_{_k}"] = conf_race[f"ladder_conf_{_k}_pct"].rank(pct=True, method="average")
LADDER_TIER_MAP = {
    rid: tuple(row[f"ladder_tier_{k}"] for k in LADDER_KS)
    for rid, row in zip(conf_race["race_id"], conf_race.to_dict("records"))
}


def _ladder_tier_class(pct_rank: float) -> str:
    if pd.isna(pct_rank):
        return ""
    if pct_rank >= 0.8:
        return " ladder-top20"
    if pct_rank >= 0.5:
        return " ladder-top50"
    return ""


CONF_TITLE = (
    "上位K頭picksの中に実際の1着馬が入っていたかをLOBO較正した確率(単勝ベース、K=5)。"
    "統計的に安定して自明な基準を上回れないK(NARは1025レース・93ブロックあってもK=5は"
    "現時点で該当)は較正済み%の代わりに高/中/低の3段階で表示する。「高確信度」は同日開催"
    "レース中でgap_pct_5(5位-6位差の正規化値)が上位5件であることを意味する(較正とは独立、"
    "JRA側と同一ロジック)。"
)


def confidence_badge(race_id: str) -> str:
    entry = CONF_MAP.get(race_id)
    if entry is None:
        return ""
    gap_pct, gap5, pct, show_pct, is_high = entry
    if pd.notna(gap_pct) and math.isinf(gap_pct):
        text = "確信度 全頭差"
    elif show_pct and pd.notna(pct):
        text = f"{'高確信度' if is_high else '確信度'} {pct:.0f}%"
    elif pd.notna(gap5):
        text = CC.tier_label(gap5, _GAP5_TERCILES)
    else:
        text = "確信度不明"
    cls = "conf-badge is-high" if is_high else "conf-badge"
    return f'<span class="{cls}" title="{esc(CONF_TITLE)}">{esc(text)}</span>'


_ODDS_REFRESH_TITLE = (
    "このレースだけ人気・オッズ・馬体重(bias.html 1ページ、数秒)を再取得して"
    "予想表示へ即時反映します。1日分まとめての再取得(fetch-board上部)より高速です。"
)


def odds_refresh_button(race_id: str) -> str:
    """発走前(検証待ち)のレースカードにのみ表示する、レース単位の軽量オッズ再取得ボタン。
    全レース分をまとめて取得するfetch-boardの「最新オッズを再取得」ボタンと違い、
    1レースだけbias.html(数秒)を取得してpredictions CSVへ直接反映するため高速。"""
    cmd = f"python scripts/refresh_race_display.py --race-id {race_id}"
    return (
        f'<button type="button" class="copy-btn" data-copy="{esc(cmd)}" '
        f'title="{esc(_ODDS_REFRESH_TITLE)}">📋 このレースだけ最新オッズ</button>'
    )


_LADDER_TITLE = (
    "段階的な的中確率のはしご: 「上位K頭picksの中に実際の1着馬が入っていたか」を的中と定義し、"
    "K=5,4,3,2,1それぞれで独立にLOBO較正した値(2026-07-30新設)。Kを絞るほど的中は難しくなるため、"
    "一般に5頭側が高く1頭側が低くなる。検証済みレース数がまだ少なく、K=3,2,1は現時点では"
    "自明な基準(常に平均を予測)を安定して上回れていない(下表の淡色表示)。日々検証レースが"
    "増えるたびにnar_confidence_calibrate.pyを再実行すれば自動的に更新される。"
    "色は同じK(頭数)の中での相対順位: 赤=上位20%、黄=上位20〜50%、無色=それ以外。"
)


def confidence_ladder_html(race_id: str) -> str:
    vals = LADDER_MAP.get(race_id)
    if vals is None or all(pd.isna(v) for v in vals):
        return ""
    tiers = LADDER_TIER_MAP.get(race_id, (float("nan"),) * len(LADDER_KS))
    beats = CONFIDENCE_CALIB.get("topk_ladder", {}).get("results_by_k", {})
    cells = []
    for k, v, tier in zip(LADDER_KS, vals, tiers):
        show_pct = bool(beats.get(str(k), {}).get("show_pct"))
        dim = "" if show_pct else ' style="opacity:.55"'
        txt = f"{v:.0f}%" if pd.notna(v) else "-"
        tier_cls = _ladder_tier_class(tier)
        cells.append(f'<div class="ladder-cell{tier_cls}"{dim}><b>{k}頭</b> {esc(txt)}</div>')
    return (
        f'<div class="conf-ladder" title="{esc(_LADDER_TITLE)}">'
        + "".join(cells) + "</div>"
    )


MODEL_BADGE_META = {
    "shinba": ("新馬戦モデル", "#3D5A80", "#FFFFFF"),
}


def model_badge(model: str) -> str:
    meta_ = MODEL_BADGE_META.get(model)
    if meta_ is None:
        return ""
    label, bg, fg = meta_
    return f'<span class="model-badge" style="--model-bg:{bg};--model-fg:{fg}">{esc(label)}</span>'


WAKU_COLORS = {
    "1": ("#FFFFFF", "#1A1C1F", "#B7BAB6"),
    "2": ("#2B2B2B", "#F2F2F0", "#2B2B2B"),
    "3": ("#C33B2E", "#FFFFFF", "#C33B2E"),
    "4": ("#2C5FA8", "#FFFFFF", "#2C5FA8"),
    "5": ("#E6C21A", "#1A1C1F", "#E6C21A"),
    "6": ("#2E8B4E", "#FFFFFF", "#2E8B4E"),
    "7": ("#E07B24", "#1A1C1F", "#E07B24"),
    "8": ("#E091B0", "#1A1C1F", "#E091B0"),
}
WAKU_FALLBACK = ("#9A9E9F", "#1A1C1F", "#9A9E9F")


def waku_css_rules() -> str:
    rules = [".waku { background: %s; color: %s; border: 1px solid %s; }" % WAKU_FALLBACK]
    for key, (bg, fg, border) in WAKU_COLORS.items():
        rules.append(f'.waku[data-waku="{key}"] {{ background: {bg}; color: {fg}; border-color: {border}; }}')
    return "\n".join(rules)


_WEIGHT_RE = re.compile(r"^\s*(\d+)\(([+-]?\d*)\)\s*$")


def format_horse_weight(raw) -> str:
    """bias_horse_weight の生値("462(+2)"/"430(0)"/"445()"/"0()"/空欄)を表示用に整形する。
    "0(...)" はnetkeiba側の「馬体重測定不能」を表す規約値。発走約1時間前まではそもそも
    未発表のため空欄になる。"""
    if pd.isna(raw) or str(raw).strip() == "":
        return "-"
    m = _WEIGHT_RE.match(str(raw))
    if not m:
        return "-"
    weight, diff = m.group(1), m.group(2)
    if weight == "0":
        return "計不"
    if diff in ("", "0", "+0", "-0"):
        return f"{weight}kg (±0)" if diff else f"{weight}kg"
    return f"{weight}kg ({diff})"


def waku_chip(waku) -> str:
    # NARの馬柱データには枠番(waku)列が存在しない(厩舎コメント欄由来のため常にNaN)。
    # 「-」を表示することで「データなし」を明示する(空欄より誤解が少ない)。
    try:
        w = str(int(float(waku)))
    except (ValueError, TypeError):
        w = None
    label = w if w else "-"
    data_attr = f' data-waku="{esc(w)}"' if w else ""
    return f'<span class="waku"{data_attr}>{esc(label)}</span>'


def race_card(race_id: str, race_name: str, rows: pd.DataFrame, racecourse: str | None = None) -> str:
    m = meta.loc[race_id] if race_id in meta.index else None
    if m is not None:
        parts = []
        if racecourse:
            parts.append(esc(racecourse))
        surface_val = m["surface"] if pd.notna(m["surface"]) else None
        distance_val = m["distance_m"] if pd.notna(m["distance_m"]) else None
        if surface_val or distance_val:
            surface_label = esc(SURFACE_MAP.get(surface_val, surface_val or ""))
            parts.append(f"{surface_label}{esc(distance_val)}m" if distance_val else surface_label)
        if pd.notna(m["going"]):
            parts.append(esc(m["going"]))
        if pd.notna(m["weather"]):
            parts.append(esc(m["weather"]))
        parts.append(f"発走{esc(m['start_time'])}")
        sub = "・".join(parts)
        rnum = m["race_number"]
    else:
        sub = ""
        rnum = "?"

    has_actual = race_id in ACTUAL_MAPS
    model = rows["model"].iloc[0] if "model" in rows.columns and not rows.empty else None

    body_rows = []
    for _, r in rows.iterrows():
        rank = int(r["pred_rank"])
        ninki = r["bias_ninki"]
        ninki_i = int(ninki) if pd.notna(ninki) and ninki == ninki else None
        longshot = ninki_i is not None and ninki_i >= 6 and rank <= 3
        _raw_score = r["_score"] if pd.notna(r["_score"]) else 0.0
        score_pct = max(0.0, min(1.0, _raw_score)) * 100
        odds = r["bias_win_odds"]
        odds_txt = f"{float(odds):.1f}" if pd.notna(odds) and odds != "" else "-"
        ninki_txt = f"{ninki_i}人気" if ninki_i is not None else "-"
        mark_cls = " is-mark" if longshot else ""

        result_cells = ""
        place_cls = ""
        if has_actual:
            fl = finish_label(race_id, r["umaban"])
            fl_cls = " is-won" if fl == "1着" else ""
            if fl in ("1着", "2着", "3着"):
                place_cls = f" place-{fl[0]}"
            place_amt = place_payout(race_id, r["umaban"])
            place_txt = f"¥{place_amt:,}" if place_amt > 0 else "-"
            result_cells = (
                f'<td class="c-finish{fl_cls}">{esc(fl) if fl else "-"}</td>'
                f'<td class="c-place">{place_txt}</td>'
            )

        weight_txt = format_horse_weight(r["bias_horse_weight"]) if "bias_horse_weight" in r.index else "-"

        body_rows.append(
            f"""<tr class="rank-{rank}{place_cls}">
              <td class="c-rank"><span class="rank-badge">{rank}</span></td>
              <td class="c-waku">{waku_chip(r['waku'])}</td>
              <td class="c-uma">{esc(r['umaban'])}</td>
              <td class="c-name">{esc(r['horse_name'])}<span class="mark{mark_cls}">{'穴' if longshot else ''}</span></td>
              <td class="c-ninki">{ninki_txt}</td>
              <td class="c-odds">{odds_txt}</td>
              <td class="c-weight">{esc(weight_txt)}</td>
              <td class="c-score">
                <div class="score-bar"><div class="score-bar-fill" style="width:{score_pct:.0f}%"></div></div>
                <span class="score-num">{score_pct:.0f}</span>
              </td>{result_cells}
            </tr>"""
        )

    extra_th = '<th>着順</th><th>複勝払戻</th>' if has_actual else ""

    result_strip = ""
    if has_actual:
        umabans = [int(u) for u in rows["umaban"]]
        chips = []
        for item in box_actual_results(race_id, umabans):
            if item["hit"]:
                chips.append(
                    f'<span class="payout-chip is-hit">{esc(item["bet_type"])} '
                    f'{esc(item["label"])} ¥{item["amount"]:,}</span>'
                )
            else:
                chips.append(f'<span class="payout-chip is-miss">{esc(item["bet_type"])} 不的中</span>')
        result_strip = (
            '<div class="result-strip"><span class="result-strip-label">予想5頭BOX 払戻実績</span>'
            f'{"".join(chips)}</div>'
        )

    quick_result = "" if has_actual else quick_result_html(race_id)

    return f"""
    <article class="race-card" id="race-{esc(race_id)}">
      <header class="race-head">
        <span class="race-num">{esc(rnum)}R</span>
        <div class="race-head-text">
          <h3><a class="race-name-link" href="https://nar.netkeiba.com/race/shutuba.html?race_id={esc(race_id)}" target="_blank" rel="noopener">{esc(race_name)}</a>{model_badge(model)}{confidence_badge(race_id)}{'' if has_actual else odds_refresh_button(race_id)}</h3>
          <p class="race-sub">{sub}</p>
        </div>
      </header>
      <div class="pick-table-wrap">
        <table class="pick-table">
          <thead>
            <tr><th></th><th></th><th>馬番</th><th>馬名</th><th>人気</th><th>単勝</th><th>馬体重</th><th>score</th>{extra_th}</tr>
          </thead>
          <tbody>
            {''.join(body_rows)}
          </tbody>
        </table>
      </div>
      {confidence_ladder_html(race_id)}
      {quick_result}
      {result_strip}
    </article>"""


def build_course_blocks(date_key: str, date_df: pd.DataFrame) -> tuple[str, list[tuple[str, str, int]]]:
    course_blocks = []
    nav_entries = []
    for course in date_df["racecourse"].drop_duplicates():
        course_df = date_df[date_df["racecourse"] == course]
        race_ids_ordered = course_df[["race_id", "race_name"]].drop_duplicates()
        race_ids_ordered["rnum"] = race_ids_ordered["race_id"].map(
            lambda rid: int(meta.loc[rid, "race_number"]) if rid in meta.index else 99
        )
        race_ids_ordered = race_ids_ordered.sort_values("rnum")

        cards = []
        for _, rr in race_ids_ordered.iterrows():
            rows = course_df[course_df["race_id"] == rr["race_id"]].sort_values("pred_rank")
            cards.append(race_card(rr["race_id"], rr["race_name"], rows))

        anchor = esc(f"{date_key}-{course}")
        nav_entries.append((anchor, course, len(race_ids_ordered)))
        course_blocks.append(
            f"""
        <section class="course-block" id="sec-{anchor}">
          <h3 class="course-head">{esc(course)}<span class="course-count">{len(race_ids_ordered)}R</span></h3>
          <div class="race-grid">{''.join(cards)}</div>
        </section>"""
        )
    return "".join(course_blocks), nav_entries


def build_time_ordered_blocks(date_key: str, date_df: pd.DataFrame) -> tuple[str, list[tuple[str, str, int]]]:
    """当日(pending)分専用: 競馬場でグルーピングせず、発走時刻順に全競馬場を横断して並べる。"""
    race_ids_ordered = date_df[["race_id", "race_name", "racecourse"]].drop_duplicates()
    race_ids_ordered["start_time"] = race_ids_ordered["race_id"].map(
        lambda rid: meta.loc[rid, "start_time"] if rid in meta.index else "99:99"
    )
    race_ids_ordered["start_time"] = race_ids_ordered["start_time"].fillna("99:99")
    race_ids_ordered = race_ids_ordered.sort_values(["start_time", "racecourse"], kind="stable")

    cards = []
    for _, rr in race_ids_ordered.iterrows():
        rows = date_df[date_df["race_id"] == rr["race_id"]].sort_values("pred_rank")
        cards.append(race_card(rr["race_id"], rr["race_name"], rows, racecourse=rr["racecourse"]))

    anchor = esc(f"{date_key}-timeline")
    nav_entries = [(anchor, "発走時刻順(全競馬場)", len(race_ids_ordered))]
    html_block = f"""
        <section class="course-block" id="sec-{anchor}">
          <h3 class="course-head">発走時刻順・全競馬場<span class="course-count">{len(race_ids_ordered)}R</span></h3>
          <div class="race-grid">{''.join(cards)}</div>
        </section>"""
    return html_block, nav_entries


sections = []
nav_items = []
for date_key in reversed(verified_dates):
    date_df = pred[pred["kaisai_date"] == date_key]
    if date_df.empty:
        continue
    course_html, nav_entries = build_course_blocks(date_key, date_df)
    _short, full_label = format_date_label(date_key)
    for anchor, course, _cnt in nav_entries:
        nav_items.append(f'<a href="#sec-{anchor}">{esc(full_label)} {esc(course)}</a>')
    sections.append(f"""
    <section class="date-block">
      <h2 class="date-head">{esc(full_label)}</h2>
      {course_html}
    </section>""")

pending_sections = []
pending_nav_items = []
for date_key in reversed(pending_predictable_dates):
    date_df = pred[pred["kaisai_date"] == date_key]
    if date_df.empty:
        continue
    course_html, nav_entries = build_time_ordered_blocks(date_key, date_df)
    _short, full_label = format_date_label(date_key)
    for anchor, course, _cnt in nav_entries:
        pending_nav_items.append(f'<a class="is-pending" href="#sec-{anchor}">{esc(full_label)} {esc(course)}</a>')
    pending_sections.append(f"""
    <section class="pending-section" id="sec-pending-{date_key}">
      <h2 class="date-head">{esc(full_label)}<span class="status-badge is-pending">検証待ち</span></h2>
      <p class="pending-note">
        この日の結果・払戻データはまだ存在しないため、的中率・回収率の検証はできません。
        人気・オッズは前日〜当日朝時点の値です。結果反映後に更新予定です。
        当日分は競馬場ごとではなく、発走時刻順に全競馬場を横断して並べています。
      </p>
      {course_html}
    </section>""")

verified_short = [format_date_label(d)[0] for d in verified_dates]
pending_short = [format_date_label(d)[0] for d in pending_predictable_dates]
title_date_part = "・".join(verified_short) if verified_short else "(検証済み日付なし)"
if pending_short:
    title_date_part += f"(+{'・'.join(pending_short)}検証待ち)"
page_title = f"地方競馬 馬柱データ予想 - {title_date_part}"

_courses_verified = sorted(pred[pred["kaisai_date"].isin(verified_dates)]["racecourse"].unique()) if verified_dates else []
lede_html = (
    f"{esc(format_lede_dates(verified_dates))}開催の地方競馬(NAR)"
    f"{esc('・'.join(_courses_verified))}を対象に、レースの性質(通常戦・新馬戦)に応じて"
    "専用にチューニングした2モデルで1着〜5着を予想。"
    f"全{total_races_verified}レースが対象です。単勝人気・オッズは分析軸に混ぜると循環参照に"
    "なるため、いずれのモデルも予想スコアには一切使用していません(表には参考情報として併記のみ)。"
    "地方競馬の馬柱データには厩舎コメント欄・調教評価欄が存在しないため、枠番(waku)は表内で"
    "常に「-」表示、枠連のBOX回収率も検証対象外です。各モデルの詳細は下の「予想ロジックの内訳」で"
    "確認できます。"
)
if pending_predictable_dates:
    pending_races_n = pred[pred["kaisai_date"].isin(pending_predictable_dates)]["race_id"].nunique()
    lede_html += (
        f" {esc(format_lede_dates(pending_predictable_dates))}開催分({pending_races_n}レース)も"
        "同じモデル体制で予想済みですが、結果・払戻データが未取得のため回収率検証はまだ"
        "できていません(該当セクションに表示)。"
    )

# --- methodセクション: winner_shinba.json を正本にする ---
# 通常戦モデル(予想5頭)は2026-08-01よりwinner_box5_nar.jsonを正本にする。
# 2026-07-29〜07-31まではwinner_box4_nar.jsonを流用していたが、ユーザー依頼により
# box5にも独立の300パターン探索を実施(結果は不採用、等重み17シグナル)。レースカードの
# 予想5頭(predictions_nar_{date}.csv、predict_top5_nar.py生成)・予想5頭BOX回収率検証
# (confidence_sweep_baseline_nar.csv)もすべてwinner_box5_nar.jsonの重みを使う。
winner = json.loads((DATA_DIR / "winner_box5_nar.json").read_text(encoding="utf-8"))
MODEL_WEIGHTS = winner["weights"]
PATTERN_ID = winner["pattern_id"]
MODEL_LABEL = winner.get("model_label", f"pattern{PATTERN_ID}")
MODEL_DEAD = winner.get("dead_signals", [])
MODEL_ALIVE = winner.get("alive_signals", [])

# CONFIDENCE_CALIBはCONF_MAP定義より前(ファイル前方)で読み込み済み。


def _confidence_calibration_note(box_n: int) -> str:
    """2026-08-12、JRA/NAR確信度統一に伴い書き換え。旧来のbox_n×place/profit較正
    (results_by_box_n)は廃止し、JRA共通の単勝ベースtopk_ladder(K=box_n)の較正結果を
    説明する。box5→K=5、box4→K=4、box3→K=3を1対1で対応させている。"""
    r = CONFIDENCE_CALIB.get("topk_ladder", {}).get("results_by_k", {}).get(str(box_n), {})
    if not r:
        return ""
    show_pct = r.get("show_pct", False)
    ci_lo, ci_hi = r.get("brier_gain_ci95", [0, 0])
    sign_warning = r.get("chosen_feature_sign_warning", False)
    warn_html = (
        "<br><br><b style=\"color:#c0392b\">注意:</b> 正の相関を持つ候補が無かったため、"
        "符号を無視してBrier score最良の候補を採用しています(表示上の大小と実測的中率の"
        "大小が一致しない可能性があります)。"
    ) if sign_warning else ""
    verdict = (
        f"自明な基準を統計的に上回った(ブロック単位ブートストラップ95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]、"
        "下限がプラス)ため、較正済み%をそのまま表示している。"
    ) if show_pct else (
        f"自明な基準を安定して上回れていない(95%CI=[{ci_lo:+.4f}, {ci_hi:+.4f}]、または"
        f"ブロック数{r.get('n_blocks_total', 0)}が目安の{r.get('min_blocks_for_pct', 60)}に"
        "未達)ため、較正済み%の代わりに高/中/低の3段階(相対的な目安)で表示している。"
    )
    return (
        "<br><br>＜確信度指標の較正(2026-08-12、JRA/NAR共通ロジックに統一)＞ "
        f"「上位{box_n}頭picksの中に実際の1着馬が入っていたか」(単勝ベース)を的中と定義し、"
        f"検証済み{r.get('n_races', 0)}レース・{r.get('n_blocks_total', 0)}ブロック"
        f"(実測hit率{r.get('overall_hit_rate_pct', 0):.1f}%)に対するLOBO較正(1ブロックを除いた"
        "残りだけで較正パラメータを推定し、除いたブロックで評価)を行った。主手法は1変数"
        "ロジスティック回帰(Platt scaling)。"
        f"OOF Brier score {r.get('chosen_oof_brier', 0):.4f}と自明な基準(常に平均を予測、"
        f"{r.get('trivial_baseline_oof_brier', 0):.4f})との差を、ブロック単位ブートストラップ"
        f"(n=2000)で95%信頼区間評価した。{verdict}"
        f"{warn_html}"
        "検証済みレースが増えるたびにnar_confidence_calibrate.pyを再実行するだけで自動的に"
        "最新化される。"
    )

WEIGHT_META = {
    "form": ("近走成績(クラス補正)",
             "直近3走の着順を新しいレースほど重く加重平均。当時のレースクラスと今回のクラスとの差に応じて着順を補正"),
    "style": ("脚質×展開",
              "コース別脚質勝率・複勝率(縮約後)のブレンドに加え、レース全体の逃げ・先行馬の比率からペースの流れを推定し個別補正"),
    "apt": ("コース適性", "スピード指数カテゴリのコース別勝率(縮約後)。NARの馬柱データには該当列が存在しないため常にNaN、他シグナルへ自動再配分される"),
    "bms": ("母父適性", "母父(ブルードメアサイア)の勝率・複勝率・単勝回収率のブレンド(縮約後)"),
    "speed": ("スピード指数", "最高値・直近5走平均・前走値の平均"),
    "waku": ("枠順バイアス", "今回の枠順における当該コースの勝率(縮約後、集団統計ベース。現在の枠番そのものとは別物)"),
    "sire": ("種牡馬適性", "父の勝率・複勝率・単勝回収率のブレンド(縮約後)"),
    "jt": ("騎手・厩舎", "当該コース条件での騎手・調教師勝率平均(縮約後)"),
    "distance": ("距離適性", "今回と同距離帯における勝率・複勝率・単勝回収率のブレンド(縮約後)"),
    "train": ("調教評価", "追い切りランク(A〜E)。NARの馬柱データには厩舎コメント欄が存在しないため常にNaN、他シグナルへ自動再配分される"),
}

WEIGHT_META_EXTRA = {
    "comment": ("厩舎コメント評価",
                "netkeiba馬柱の厩舎コメント欄に付く評価マーク(3段階)を数値化。JRA向けに探索された"
                "新馬戦専用シグナルだが、NARの馬柱データには厩舎コメント欄が存在しないため常にNaN、"
                "実質使用されていない"),
    # --- 2026-07-29 の再モデリングで4頭BOXモデルに追加した候補シグナル ---
    "jockey": ("騎手", "当該コース条件での騎手勝率(縮約後)。従来は調教師と平均して1本にまとめていたものを分離"),
    "trainer": ("調教師", "当該コース条件での調教師勝率(縮約後)。従来は騎手と平均して1本にまとめていたものを分離"),
    "jt_return": ("騎手・厩舎の回収率", "当該コース条件での騎手・調教師の単勝回収率のブレンド(縮約後)。勝率ではなく回収率ベース"),
    "nige": ("先行力", "過去5走の1コーナー通過順を各レースの出走頭数で割った相対位置を、新しい走ほど重く加重平均。地方競馬で最も構造的に効く逃げ・先行有利を捉える"),
    "concerned": ("当該コース+距離の自己成績", "この馬自身の「今回とまったく同じ競馬場・同じ距離」での勝率・複勝率・単勝回収率のブレンド(縮約後)"),
    "course": ("競馬場適性", "この馬自身の「今回と同じ競馬場・同じ馬場種別」(例: ダート高知)での勝率・複勝率・単勝回収率のブレンド(縮約後)"),
    "margin": ("着差", "直近3走の着差(秒)を新しい走ほど重く加重平均。着順より情報量が多い"),
    "agari": ("上がり3F", "直近3走の上がり3ハロンタイムの平均。末脚の速さ"),
    # --- 2026-07-29 の3頭BOX再モデリングで検証した候補シグナル(いずれも不採用、重み0.0)。
    "interval": ("休養日数適性", "「中○週」「1年以上」等の休養日数帯における、この馬自身の勝率・複勝率・単勝回収率のブレンド(縮約後)。LOBO OOFで基準を悪化させたため不採用"),
    "kinryo": ("斤量帯適性", "「57kg」等の斤量帯における、この馬自身の勝率・複勝率・単勝回収率のブレンド(縮約後)。LOBO OOFで基準をわずかに下回ったため不採用"),
    # --- 2026-07-29 ユーザー依頼で追加、300パターン自由探索(3回目の更新)で採用。
    "timediff": ("直近5走の1着とのタイム差", "直近5走の1着とのタイム差(秒)を新しい走ほど重く加重平均した独立シグナル(marginの3走版を5走に拡張)。当初のLOBO OOF二値採否では統計的検出力不足で不採用と判定されたが、ユーザー指示によりサンプル増加を見込んで採用"),
    "class_ninki": ("当時のクラス×人気順補正近走成績(v2)", "直近5走の「当時のクラス序列×1/当時の人気順」を市場価値の代理指標とし、人気順を上回る着順となった場合のみ加点する片側ボーナスをブレンド。専門家・エンジニアレビューで非線形性による歪みの懸念が指摘されている"),
    "weight": ("直近馬体重の増減幅(v2)", "当日再取得したbias_horse_weightの増減幅。プラス(増加)ほど高評価、かつ絶対値が小さいほど高評価の2成分をブレンド。専門家レビューでは「馬体重増加が必ずしも好材料とは限らない」との実務上の懸念があり、2026-07-29の300パターン探索でも重みは0.6%とごく小さかった(現在は等重み)"),
}
ALL_WEIGHT_META = {**WEIGHT_META, **WEIGHT_META_EXTRA}
assert set(MODEL_WEIGHTS.keys()) <= set(ALL_WEIGHT_META.keys()), (
    "ALL_WEIGHT_META に winner_box4_nar.json の重みキーが足りません: "
    f"{set(MODEL_WEIGHTS.keys()) - set(ALL_WEIGHT_META.keys())}"
)


def build_weight_li(weights: dict, meta_: dict, dead: list = None) -> list[str]:
    """重み内訳のリスト項目を作る。dead に挙がったシグナル(NARで値が存在せず、
    実際には他シグナルへ再配分されているもの)は表示しない。表示すると
    『スピード指数21.3%』のように、実態と正反対の印象を与えるため。"""
    dead = set(dead or [])
    li = []
    for key, w in sorted(weights.items(), key=lambda kv: -kv[1]):
        if key in dead or w <= 0:
            continue
        label, desc = meta_[key]
        li.append(f"<li><b>{esc(label)}</b> {w * 100:.1f}% ― {esc(desc)}</li>")
    return li


def _factor_v2_note(factor_v2: dict, base_label: str = "基準") -> str:
    """2026-07-29のユーザー依頼(直近5走の1着とのタイム差・当時のクラス×人気順補正・
    直近馬体重の増減幅)のLOBO OOF検証結果を、method_note文末に追記する共通ヘルパー。"""
    if not factor_v2:
        return ""
    oof = factor_v2.get("factor_oof_box4", {})
    opt = factor_v2.get("selection_optimism_box4", {})
    b3check = factor_v2.get("box3_consistency_check", {})
    base_excess = oof.get(base_label, {}).get("excess", 0.0)
    best_name = max(oof, key=lambda k: oof[k].get("excess", -999)) if oof else base_label
    best_excess = oof.get(best_name, {}).get("excess", 0.0)
    edge = opt.get("true_edge_pt", 0.0)
    edge_sd = opt.get("true_edge_sd", 0.0)
    b3_extra = (
        f"3頭BOX側にも同じ重みを適用して{b3check.get('final', {}).get('excess', 0.0):+.2f}pt"
        f"(基準{b3check.get('base', {}).get('excess', 0.0):+.2f}pt)と、box_n=4での結論との"
        "一貫性を確認済みです。" if b3check else ""
    )
    return (
        "<br><br>＜追加検証(2026-07-29、ユーザー依頼): 直近5走の1着とのタイム差・"
        "当時のクラス×人気順補正・直近馬体重の増減幅＞ この3候補の全8通りの組み合わせ"
        f"(すべて等重み)をbox_n=4でLOBO OOF判定しました。最良候補「{esc(best_name)}」は"
        f"市場差{best_excess:+.2f}pt(基準は{base_excess:+.2f}pt)と単独では有望でしたが、"
        f"8候補から選ぶこと自体の正味の価値は{edge:+.2f}pt(標準偏差{edge_sd:.2f})でsdの2倍"
        f"({2 * edge_sd:.2f}pt)を超えず、統計的に「たまたま良く見えただけ」と区別できないため"
        "不採用としました(interval/kinryoと同じ採否ゲート)。基準モデルを維持しています。"
        + b3_extra
    )


weight_li = build_weight_li(MODEL_WEIGHTS, ALL_WEIGHT_META, dead=MODEL_DEAD)

winner_shinba = json.loads((DATA_DIR / "winner_shinba.json").read_text(encoding="utf-8"))
SHINBA_WEIGHTS = winner_shinba["weights"]
SHINBA_PATTERN_ID = winner_shinba["pattern_id"]
assert set(SHINBA_WEIGHTS.keys()) <= set(WEIGHT_META) | set(WEIGHT_META_EXTRA), (
    "WEIGHT_META/WEIGHT_META_EXTRA が winner_shinba.json の重みキーと一致していません。"
)
shinba_weight_li = build_weight_li(SHINBA_WEIGHTS, {**WEIGHT_META, **WEIGHT_META_EXTRA})

has_shinba_predictions = _shinba_path.exists()


def drop_unverifiable_bets(df: pd.DataFrame) -> pd.DataFrame:
    """賭け金が一度も発生していない券種の行を落とす。

    NARの馬柱データには枠番の列が無いため、枠連は賭け目そのものを構成できず
    total_stake が常に0になる。これを「回収率0.0%」として表示すると
    『このモデルは枠連が全く当たらない』と読めてしまうが、実際は『一度も買っていない』。
    誤解を生む行なので描画前に除外する。"""
    if "total_stake" not in df.columns:
        return df
    return df[df["total_stake"] > 0].reset_index(drop=True)


def build_simple_box_section(csv_name: str, anchor: str, heading: str, extra_note: str = "") -> str | None:
    """NAR版のBOX回収率検証(単一スコープのみ、JRAのconfidence_sweepタブや人気順比較は
    対象外)。box_return_nar.py / predict_shinba_nar.py が出力するbox_return_summary形式
    (bet_type,races,hit_races,hit_rate_pct,total_stake,total_return,return_rate_pct)を
    そのまま1テーブルとして描画する。"""
    path = DATA_DIR / csv_name
    if not path.exists():
        return None
    df = pd.read_csv(path, dtype=str)
    for _c in ["races", "hit_races", "total_stake", "total_return", "hit_rate_pct", "return_rate_pct"]:
        df[_c] = pd.to_numeric(df[_c])
    df = drop_unverifiable_bets(df)
    n_races = int(df["races"].iloc[0]) if not df.empty else 0

    def _rate_cls(rate: float) -> str:
        return "is-plus" if rate >= 100 else ("is-mid" if rate >= 80 else "is-minus")

    rows_html = []
    for _, r in df.iterrows():
        rows_html.append(f"""<tr>
          <td class="bt-name">{esc(r['bet_type'])}</td>
          <td>{int(r['hit_races'])}/{int(r['races'])}<span class="box-sub">({r['hit_rate_pct']:.1f}%)</span></td>
          <td class="num">¥{int(r['total_stake']):,}</td>
          <td class="num">¥{int(r['total_return']):,}</td>
          <td class="num rate {_rate_cls(r['return_rate_pct'])}">{r['return_rate_pct']:.1f}%</td>
        </tr>""")

    return f"""
    <section class="box-section" id="{anchor}">
      <details class="box-details">
        <summary class="box-head">{esc(heading)} 予想5頭BOX 回収率検証</summary>
        <p class="box-lede">
          対象{n_races}レースで、予想上位5頭をBOX(全通り)購入した場合の実際の払い戻しとの答え合わせです。
        </p>
        <div class="box-table-wrap">
          <table class="box-table">
            <thead><tr><th>券種</th><th>的中レース</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr></thead>
            <tbody>{''.join(rows_html)}</tbody>
          </table>
        </div>
        <details class="method box-method">
          <summary>計算方法・前提条件</summary>
          <p class="method-note">
            1点100円換算、単勝・複勝は5点、馬連・ワイド・3連複は10点、馬単は20点、3連単は60点。
            枠連は枠番データが無いため対象外(常に0)。{esc(extra_note)}
          </p>
        </details>
      </details>
    </section>"""


BET_ORDER = ["単勝", "複勝", "枠連", "馬連", "ワイド", "馬単", "3連複", "3連単"]
BET_ORDER_MAP = {b: i for i, b in enumerate(BET_ORDER)}


def short_scope_label(scope: str) -> str:
    m = re.match(r"^全(\d+)レース$", scope)
    if m:
        return f"全{m.group(1)}R"
    m = re.match(r"^高確信度(\d+)レース/日", scope)
    if m:
        return f"{m.group(1)}R/日"
    return scope


def build_tabbed_box_section(csv_name: str, anchor: str, id_prefix: str, heading: str,
                              lede: str, method_note: str) -> tuple[str, str]:
    """BOX4/BOX3用の確信度タブ付きセクション(JRA版build_artifact.pyのbuild_tabbed_box_section
    を移植)。confidence_sweep_box4_nar.py/confidence_sweep_box3_nar.pyが出力する
    scope/bet_type/races/hit_races/total_stake/total_return/hit_rate_pct/return_rate_pct
    形式のCSVをタブUIとして描画する。"""
    conf = pd.read_csv(DATA_DIR / csv_name, dtype=str)
    for _c in ["races", "hit_races", "total_stake", "total_return", "hit_rate_pct", "return_rate_pct"]:
        conf[_c] = pd.to_numeric(conf[_c])
    conf = drop_unverifiable_bets(conf)
    scopes = list(dict.fromkeys(conf["scope"]))  # ファイル出現順を保ったまま重複除去

    tab_elems = []
    tab_panels = []
    tab_css_rules = []
    for i, scope in enumerate(scopes):
        tab_id = f"conf-tab-{id_prefix}-{i}"
        panel_id = f"conf-panel-{id_prefix}-{i}"
        checked = " checked" if i == 0 else ""
        tab_elems.append(
            f'<input type="radio" name="conf-tab-{id_prefix}" id="{tab_id}"{checked} class="conf-tab-input">'
            f'<label for="{tab_id}" class="conf-tab-label">{esc(short_scope_label(scope))}</label>'
        )

        sub = conf[conf["scope"] == scope].copy()
        sub["_order"] = sub["bet_type"].map(BET_ORDER_MAP)
        sub = sub.sort_values("_order")
        rows_html = []
        for _, r in sub.iterrows():
            rate = r["return_rate_pct"]
            hit_rate = r["hit_rate_pct"]
            cls = "is-plus" if rate >= 100 else ("is-mid" if rate >= 80 else "is-minus")
            rows_html.append(
                f"""<tr>
          <td class="bt-name">{esc(r['bet_type'])}</td>
          <td>{int(r['hit_races'])}/{int(r['races'])}<span class="box-sub">({hit_rate:.1f}%)</span></td>
          <td class="num">¥{int(r['total_stake']):,}</td>
          <td class="num">¥{int(r['total_return']):,}</td>
          <td class="num rate {cls}">{rate:.1f}%</td>
        </tr>"""
            )

        tab_panels.append(
            f"""<div class="conf-tab-panel" id="{panel_id}">
          <p class="conf-scope-label">{esc(scope)}</p>
          <div class="box-table-wrap">
            <table class="box-table">
              <thead><tr><th>券種</th><th>的中レース</th><th>投資額</th><th>払戻額</th><th>回収率</th></tr></thead>
              <tbody>{''.join(rows_html)}</tbody>
            </table>
          </div>
        </div>"""
        )
        tab_css_rules.append(f"#{tab_id}:checked ~ .conf-tabs-panels #{panel_id} {{ display: block; }}")

    section_html = f"""
    <section class="box-section" id="{anchor}">
      <details class="box-details">
        <summary class="box-head">{esc(heading)}</summary>
        <p class="box-lede">
          {lede}
        </p>
        <div class="conf-tabs">
          {''.join(tab_elems)}
          <div class="conf-tabs-panels">{''.join(tab_panels)}</div>
        </div>
        <details class="method box-method">
          <summary>計算方法・前提条件</summary>
          <p class="method-note">
            {method_note}
          </p>
        </details>
      </details>
    </section>"""
    return "\n".join(tab_css_rules), section_html


conf_tabs_css_box5, box_section = build_tabbed_box_section(
    "confidence_sweep_baseline_nar.csv", "sec-box", "b5", "通常戦モデル 予想5頭BOX 回収率検証",
    lede=(
        "各レースの予想1〜5位をBOX(全通り)購入した場合の実際の払い戻しとの答え合わせ。"
        f"<b>{esc(MODEL_LABEL)}</b>を使用しています。"
        "2026-08-01、取得データ増加(253レース)を受けbox5にも独立の300パターン自由探索を"
        "実施しましたが、ユーザー依頼によるプロのシステムエンジニアレビューの結果"
        "「探索は構造的に成立しない」と判定され不採用、等重みを維持しています"
        "(box4・box3も同時に2026-07-29採用のpattern#157を撤回し等重みへ差し戻し済み)。"
        "詳細と検証データの信頼性に関する留意点(検証レースの77%が発走後取得である点)は"
        "「予想4頭BOXモデルの内訳」を参照してください。タブで対象レースの絞り込み条件を"
        "切り替えられます。"
    ),
    method_note=(
        "1点100円換算、単勝・複勝は5点、馬連・ワイド・3連複は10点、馬単は20点、3連単は60点。"
        "枠連は枠番データが無いため対象外(常に0)。JRAの控除率(競馬全体の理論回収率は約70〜80%)を"
        "上回れば、予想モデルが市場平均を上回っていることを意味します。「高確信度Nレース/日」は"
        "オッズ・人気を一切使わず、レース内で1位と2位のスコア差を(1位−最下位)の幅で正規化した"
        "比率(gap_top2)が大きい順にレースを選んだものです。NARは1場あたり最大12レースの開催が"
        "あるため、JRAのN=5〜10ではなくN=5〜12まで対象にしています。単一の最良Nを示すものではなく、"
        "回収率の振れ幅は大きめです。300パターン探索の経緯・専門家レビューの詳細は"
        "「予想4頭BOXモデルの内訳」を参照してください。"
        + _confidence_calibration_note(5)
        + "<br><br>＜段階的的中確率のはしご(2026-08-12、JRAと共通ロジックに統一)＞ 各レースカードに、"
        "「上位K頭picksの中に実際の1着馬が入っていたか」をK=5,4,3,2,1それぞれで"
        "独立にLOBO較正した確信度を表示しています(予想5頭表示と同じ重みを使用)。"
        + "、".join(
            f"{k}頭側は実測hit率{v.get('overall_hit_rate_pct', 0):.0f}%・"
            f"{'較正済み%を表示' if v.get('show_pct') else '3段階(高/中/低)表示にフォールバック'}"
            for k, v in sorted(
                CONFIDENCE_CALIB.get("topk_ladder", {}).get("results_by_k", {}).items(),
                key=lambda kv: -int(kv[0]),
            )
        )
        + "。ブロック単位ブートストラップ95%CIで自明な基準を安定して上回れないK、または"
        "ブロック数が目安(60)未満のKは表では淡色表示にしています。今後検証済みレースが増える"
        "たびにnar_confidence_calibrate.pyを再実行すれば自動的に再較正されます。"
    ),
)

winner_box4 = json.loads((DATA_DIR / "winner_box4_nar.json").read_text(encoding="utf-8"))
BOX4_WEIGHTS = winner_box4["weights"]
BOX4_PATTERN_ID = winner_box4["pattern_id"]
BOX4_LABEL = winner_box4.get("model_label", f"pattern{BOX4_PATTERN_ID}")
BOX4_DEAD = winner_box4.get("dead_signals", [])
BOX4_OBJ = winner_box4.get("objective", {})
BOX4_BIAS = winner_box4.get("selection_bias_measured", {})
BOX4_FACTOR_V2 = winner_box4.get("factor_test_2026_07_29_v2", {})
assert set(BOX4_WEIGHTS.keys()) <= set(ALL_WEIGHT_META.keys()), (
    "ALL_WEIGHT_META に winner_box4_nar.json の重みキーが足りません: "
    f"{set(BOX4_WEIGHTS.keys()) - set(ALL_WEIGHT_META.keys())}"
)
box4_weight_li = build_weight_li(BOX4_WEIGHTS, ALL_WEIGHT_META, dead=BOX4_DEAD)

winner_box3 = json.loads((DATA_DIR / "winner_box3_nar.json").read_text(encoding="utf-8"))
BOX3_WEIGHTS = winner_box3["weights"]
BOX3_PATTERN_ID = winner_box3["pattern_id"]
BOX3_LABEL = winner_box3.get("model_label", f"pattern{BOX3_PATTERN_ID}")
BOX3_DEAD = winner_box3.get("dead_signals", [])
BOX3_REJECTED = winner_box3.get("candidate_signals_rejected", {})
BOX3_OBJ = winner_box3.get("objective", {})
BOX3_FACTOR_BIAS = winner_box3.get("selection_bias_measured_for_factor_choice", {})
BOX3_FREE_BIAS = winner_box3.get("free_search_harm_reconfirmation", {})
BOX3_LEGACY_REF = winner_box3.get("rejected_alternatives", {}).get("現行pattern6", {})
BOX3_PAIRED = winner_box3.get("paired_bootstrap", {})
BOX3_FACTOR_V2 = winner_box3.get("factor_test_2026_07_29_v2", {})
assert set(BOX3_WEIGHTS.keys()) <= set(ALL_WEIGHT_META.keys()), (
    "ALL_WEIGHT_META に winner_box3_nar.json の重みキーが足りません: "
    f"{set(BOX3_WEIGHTS.keys()) - set(ALL_WEIGHT_META.keys())}"
)
box3_weight_li = build_weight_li(BOX3_WEIGHTS, ALL_WEIGHT_META, dead=BOX3_DEAD)

_b4_model = BOX4_OBJ.get("model_pct", 0.0)
_b4_market = BOX4_OBJ.get("market_pct", 0.0)
_b4_excess = BOX4_OBJ.get("excess_pt", 0.0)
_b4_search8 = winner_box4.get("search300_2026_08_01", {})
_b4_r8 = _b4_search8.get("results_by_box", {}).get("4", {})
_b4_insample_excess = _b4_r8.get("in_sample_best_excess_pt", 0.0)
_b4_nested_excess = _b4_r8.get("nested_lobo_oof_excess_pt", 0.0)
_b4_edge8 = _b4_r8.get("selection_true_edge_pt", 0.0)
_b4_edge_sd8 = _b4_r8.get("selection_true_edge_sd", 0.0)
_b4_win8 = _b4_r8.get("win_rate_vs_pool_mean", 0.0) * 100
_b4_review8 = _b4_search8.get("professional_review", {})
conf_tabs_css_box4, box_section_box4 = build_tabbed_box_section(
    "confidence_sweep_box4_nar.csv", "sec-box4", "b4", "予想4頭BOX 回収率検証",
    lede=(
        "<b>2026-08-01、データ増加(253レース)を受けた再検証の結果、等重み17シグナルに"
        "差し戻したモデル</b>による検証です。2026-07-29には300パターン自由探索で選んだ"
        "パターン#157(差のある重み)をユーザー判断で採用していましたが、データが253"
        "レースに増えたため同じ手法で再探索したところ、ユーザー依頼によるプロのシステム"
        "エンジニアレビューで「探索は構造的に成立しない」と判定され、2026-08-01付けで"
        "等重みへ差し戻しました。"
        f"現行モデル(等重み17シグナル)の実測市場差は{_b4_excess:+.2f}pt。"
        "300パターン探索を再度試した際のin-sample最良は"
        f"{_b4_insample_excess:+.2f}ptと良く見えましたが、これは探索の楽観バイアスであり、"
        f"12ブロックの交差検証(Nested LOBO OOF)による誠実な推定は市場差"
        f"{_b4_nested_excess:+.2f}ptと現行の等重みを大きく下回りました。"
        "詳細と経緯は下記「計算方法・前提条件」を参照してください。"
    ),
    method_note=(
        "1点100円換算、単勝・複勝は4点、馬連・ワイド・3連複は6点、馬単は12点、3連単は24点。"
        "枠連は馬柱データに枠番の列が無く賭け目自体を作れないため、検証対象から外しています"
        "(以前は「回収率0.0%」と表示していましたが、これは『当たらない』ではなく『買っていない』"
        "という意味だったため、行ごと削除しました)。"
        "「高確信度Nレース/日」は<b>1位と2位のスコア差</b>を(1位−最下位)の幅で正規化した"
        "比率が大きい順にレースを選んでいます(詳細は下記＜確信度指標の較正＞参照)。"
        "NARは1場あたり最大12レースの開催があるため、JRAのN=5〜10ではなくN=5〜12まで"
        "対象にしています。"
        "＜2026-08-01の300パターン再探索と専門家レビュー＞ 検証レース増加(126→253)を"
        "受け、2026-07-29と同じ手法(Dirichlet分布・seed=2029・全17シグナル)で300パターン"
        "を再探索しました。全253レースで最良のものを選んだ学習データそのものでの市場差は"
        f"{_b4_insample_excess:+.2f}pt(現行の等重みは{_b4_excess:+.2f}pt)という改善に"
        "見えましたが、これは「そのデータで選んだ」ことによる楽観を含みます。開催日×"
        "競馬場23ブロックの交差検証(Nested LOBO OOF)による誠実な推定は市場差"
        f"{_b4_nested_excess:+.2f}ptで、現行の等重みモデルはおろか市場そのものも大きく"
        "下回りました。選択バイアス診断(ブロック半分割×200反復)でも、選ぶことの正味の"
        f"価値は{_b4_edge8:+.2f}pt(標準偏差{_b4_edge_sd8:.2f})、未使用側で300パターン平均を"
        f"上回る確率は{_b4_win8:.0f}%でした。"
        "ユーザーから明示的に依頼されたプロのシステムエンジニア(サブエージェント)による"
        "検証手法レビューでは、"
        "300パターンの回収率の標準偏差(3.3〜4.0pt)から見てin-sampleの「勝ち幅」は純ノイズ"
        "で説明できる範囲であること、Dirichlet(1)サンプリングが等重み近傍を探索しない設計上"
        "の偏りにより<b>box4では現行の等重みが300パターン中97パーセンタイルに位置する"
        "(=探索は基準より悪いパターン群の最大値を基準と比べていたに等しい)</b>ことなどが"
        "指摘され、"
        f"「{esc(_b4_review8.get('conclusion', ''))}」"
        "という結論に至りました。この結果を提示した上でユーザーに確認したところ、"
        "レビュー推奨通り3BOXサイズすべて等重みを維持することが明示的に選択されたため、"
        "2026-07-29に採用したpattern#157は撤回しています。"
        "なお検証に使った253レースの77%が発走後に馬柱データを取得したものである点が"
        "レビュー過程で判明しており、検証数値全般の信頼性に関わる制約として別途記録して"
        "います(対応は保留、詳細はwinner_box4_nar.jsonのdata_provenance_caveat_2026_08_01)。"
        "300パターン探索の詳細な経緯・過去の専門家レビュー(2026-07-29時点)は下記に残して"
        "あります。"
        + _factor_v2_note(BOX4_FACTOR_V2, base_label="基準")
        + _confidence_calibration_note(4)
    ),
)
_b3_model = BOX3_OBJ.get("model_pct", 0.0)
_b3_market = BOX3_OBJ.get("market_pct", 0.0)
_b3_excess = BOX3_OBJ.get("excess_pt", 0.0)
_b3_search8 = winner_box3.get("search300_2026_08_01", {})
_b3_r8 = _b3_search8.get("results_by_box", {}).get("3", {})
_b3_insample_excess = _b3_r8.get("in_sample_best_excess_pt", 0.0)
_b3_nested_excess = _b3_r8.get("nested_lobo_oof_excess_pt", 0.0)
_b3_edge8 = _b3_r8.get("selection_true_edge_pt", 0.0)
_b3_edge_sd8 = _b3_r8.get("selection_true_edge_sd", 0.0)
_b3_win8 = _b3_r8.get("win_rate_vs_pool_mean", 0.0) * 100
_b3_review8 = _b3_search8.get("professional_review", {})
conf_tabs_css_box3, box_section_box3 = build_tabbed_box_section(
    "confidence_sweep_box3_nar.csv", "sec-box3", "b3", "予想3頭BOX 回収率検証",
    lede=(
        "<b>2026-08-01、データ増加(253レース)を受けた再検証の結果、等重み17シグナルに"
        "差し戻したモデル</b>による検証です。2026-07-29には300パターン自由探索で選んだ"
        "パターン#157(4頭BOXと共通)をユーザー判断で採用していましたが、データ増加後の"
        "再探索をプロのシステムエンジニアがレビューした結果、box3を含む全BOXサイズで"
        "「探索は構造的に成立しない」と判定され、2026-08-01付けで等重みへ差し戻しました"
        "(詳細は「予想4頭BOXモデルの内訳」参照)。"
        f"現行モデル(等重み17シグナル)の実測市場差は{_b3_excess:+.2f}pt。"
        f"300パターン再探索のin-sample最良は{_b3_insample_excess:+.2f}ptと良く見えましたが、"
        f"12ブロックの交差検証(Nested LOBO OOF)による誠実な推定は市場差"
        f"{_b3_nested_excess:+.2f}ptにとどまりました。"
        "3頭まで絞ると3連複は「BOXした3頭がそのまま実際の1〜3着」でないと的中しない"
        "1点賭けになるため、5頭・4頭BOXより的中率は大きく下がる一方、当たった場合の倍率は"
        "跳ね上がります。タブで対象レースの絞り込み条件を切り替えられます。"
    ),
    method_note=(
        "1点100円換算、単勝・複勝は3点、馬連・ワイド・3連複は3点、馬単は6点、3連単は6点"
        "(3連複だけは3頭の組み合わせが1通りしかないため実質1点賭け)。枠連は枠番データが"
        "無いため対象外(常に0)。「高確信度Nレース/日」は4頭BOX側と同じ<b>1位と2位のスコア差</b>"
        "を(1位−最下位)の幅で正規化した比率が大きい順にレースを選んでいます"
        "(詳細は下記＜確信度指標の較正＞参照)。"
        "NARは1場あたり最大12レースの開催があるため、JRAのN=5〜10ではなく"
        "N=5〜12まで対象にしています。単一の最良Nを示すものではなく、対象レースがそもそも"
        "少ない上に3連複・3連単は的中数も一桁になりやすく、回収率が数百%単位で跳ねても"
        "偶然の振れ幅である可能性が高い点にご注意ください。"
        "＜2026-08-01の300パターン再探索と専門家レビュー＞ 全253レースで最良の1パターンを"
        f"選んだ学習データそのものでの市場差は{_b3_insample_excess:+.2f}pt"
        f"(現行の等重みは{_b3_excess:+.2f}pt)でしたが、23ブロックのNested LOBO交差検証に"
        f"よる誠実な推定は市場差{_b3_nested_excess:+.2f}ptにとどまりました。"
        "選択バイアス診断(ブロック半分割×200反復)では、選ぶことの正味の価値は"
        f"{_b3_edge8:+.2f}pt(標準偏差{_b3_edge_sd8:.2f})、未使用側で300パターン平均を"
        f"上回る確率は{_b3_win8:.0f}%でした。"
        "プロのシステムエンジニア(サブエージェント)によるレビューでは、box4での診断"
        "(300パターンの回収率の標準偏差3.3〜4.0pt・等重みが300パターン中97パーセンタイル"
        "に位置する等)がbox3にも同様に当てはまるとして、"
        f"「{esc(_b3_review8.get('recommendation', ''))}」"
        "と結論づけました。ユーザーはレビュー推奨通り、box3も含め全BOXサイズで等重みを"
        "維持することを明示的に選択したため、2026-07-29に採用したpattern#157は撤回して"
        "います。旧pattern6・旧いinterval/kinryo検証の詳細な経緯は「予想3頭BOXモデルの"
        "内訳」を参照してください。"
        + _factor_v2_note(BOX3_FACTOR_V2, base_label="基準")
        + _confidence_calibration_note(3)
    ),
)
conf_tabs_css = conf_tabs_css_box5 + "\n" + conf_tabs_css_box4 + "\n" + conf_tabs_css_box3
box_section_shinba = build_simple_box_section(
    "box_return_summary_shinba_nar.csv", "sec-box-shinba", "新馬戦モデル",
    extra_note=f"対象は{winner_shinba.get('fitted_on', {}).get('n_races', '?')}レースではなくNARの新馬戦{SHINBA_PATTERN_ID and ''}のみで、現状2レースしかなく統計的な信頼性は極めて低い参考値です。",
)

# --- NAR開催日 データ取得状況ボード ---
fetch_board_rows = []
for entry in reversed(DATE_MANIFEST):
    d = entry["date"]
    _short, full_label = format_date_label(d)
    newspaper_cmd = f"python scripts/fetch_newspaper.py --date {d} --circuit nar"
    pilot_cmd = f"python scripts/run_pilot.py --date {d} --circuit nar"
    predict_cmd = f"python scripts/predict_top5_nar.py --date {d}"
    refresh_cmd = (
        f"python scripts/refresh_bias.py --date {d} --circuit nar\n"
        f"python scripts/predict_top5_nar.py --date {d}\n"
        f"python scripts/fetch_quick_result_nar.py --date {d}"
    )

    def _fb_cell(done: bool, cmd: str) -> str:
        if done:
            return '<td class="fb-cell is-done"><span class="status-badge is-done">完了</span></td>'
        return (
            '<td class="fb-cell is-todo">'
            '<span class="status-badge is-todo">未取得</span>'
            f'<button type="button" class="copy-btn" data-copy="{esc(cmd)}">📋 コマンドをコピー</button>'
            "</td>"
        )

    if entry["status"] == "verified":
        refresh_cell = '<td class="fb-cell fb-cell-muted">確定済み(対象外)</td>'
    elif not entry["newspaper"]:
        refresh_cell = '<td class="fb-cell fb-cell-muted">先にnewspaper取得が必要</td>'
    else:
        refresh_cell = (
            '<td class="fb-cell">'
            f'<button type="button" class="copy-btn" data-copy="{esc(refresh_cmd)}">📋 最新オッズを再取得</button>'
            "</td>"
        )

    fetch_board_rows.append(f"""<tr>
      <td class="fb-date">{esc(full_label)}</td>
      {_fb_cell(entry["newspaper"], newspaper_cmd)}
      {_fb_cell(entry["results"] and entry["payouts"], pilot_cmd)}
      {_fb_cell(entry["predicted"], predict_cmd)}
      {refresh_cell}
    </tr>""")

TODAY_CMD_TEMPLATE = (
    "python scripts/fetch_newspaper.py --date {DATE} --circuit nar\n"
    "python scripts/run_pilot.py --date {DATE} --circuit nar"
)

fetch_board_section = f"""
    <section class="fetch-board-section" id="sec-fetch-board">
      <h2 class="box-head">地方競馬(NAR) データ取得状況</h2>
      <p class="box-lede">
        newspaper(馬柱)・レース結果と払戻データの取得状況です。対象14場(高知・水沢・金沢・盛岡ほか)分をまとめて扱います。
        「最新オッズ・馬体重」は発走前(検証待ち)の日付でのみ、単勝オッズ・人気・馬体重の3列だけを
        bias.html 1ページから再取得する軽量版です(1レース約1分かかる馬柱の全項目取得とは別物)。
        馬体重は発走約1時間前に発表されるため、それより前は空欄のままです。取消・除外馬が出た
        場合もこのボタンで検出されます。あわせて、nar.netkeiba.comの簡易結果ページ(発走直後に
        確定、db.netkeiba.comの正式データより早い)から1〜3着・単勝払戻を取得し、発走済みレースの
        カードに速報として表示します(正式な回収率検証には使わず、翌日の確定データで別途検証します)。
      </p>
      <div class="fetch-board-today">
        <div class="fetch-board-today-text">
          <strong>本日のデータ取得</strong>
          <p>コピーしたコマンドをClaude Codeに貼り付けると、本日分の馬柱・レース結果と払戻の取得から予想生成・回収率検証・レポート更新まで自動で行われます。</p>
        </div>
        <button type="button" class="copy-btn copy-btn-today" data-copy-template="{esc(TODAY_CMD_TEMPLATE)}">📋 本日分のコマンドをコピー</button>
      </div>
      <div class="box-table-wrap">
        <table class="box-table fetch-board-table">
          <thead><tr><th>日付</th><th>newspaper</th><th>レース結果と払戻</th><th>モデリング</th><th>最新オッズ・馬体重</th></tr></thead>
          <tbody>{''.join(fetch_board_rows)}</tbody>
        </table>
      </div>
      <p class="fetch-board-note">
        実行はご自身でClaude Codeにご依頼ください。ボタンはコマンドをコピーするだけで、
        データを取得しません。
      </p>
    </section>"""

nav_parts = [
    '<span class="jumpnav-label">JUMP</span>',
    '<a href="#sec-fetch-board">取得状況</a>',
    '<a href="#sec-box">通常戦回収率</a>',
    '<a href="#sec-box4">4頭BOX回収率</a>',
    '<a href="#sec-box3">3頭BOX回収率</a>',
]
if box_section_shinba:
    nav_parts.append('<a href="#sec-box-shinba">新馬戦回収率</a>')
if pending_nav_items:
    nav_parts.append('<span class="jumpnav-group is-pending">検証待ち</span>')
    nav_parts.extend(pending_nav_items)
if nav_items:
    nav_parts.append('<span class="jumpnav-group">検証済み</span>')
    nav_parts.extend(nav_items)

jumpnav_html = f"""
<nav class="jumpnav">
  <div class="jumpnav-inner">
    {''.join(nav_parts)}
  </div>
</nav>"""

CSS = r"""
:root {
  --bg: #ECEEEA;
  --bg-elev: #F7F8F5;
  --bg-card: #FFFFFF;
  --ink: #1A1C1B;
  --ink-muted: #5B5F5A;
  --ink-faint: #656960;
  --rule: #D2D5CD;
  --rule-strong: #B7BBB1;
  --accent: #2C6E49;
  --accent-ink: #FFFFFF;
  --accent-soft: #D9E9DE;
  --accent-soft-ink: #1E4A31;
  --pending: #6B5B95;
  --pending-ink: #FFFFFF;
  --pending-soft: #E7E1F0;
  --pending-soft-ink: #4A3D6B;
  --place2-soft: #D9E4F2;
  --place2-soft-ink: #1F4E79;
  --place3-soft: #F5EAB3;
  --place3-soft-ink: #6B5610;
  --tier-red-soft: #F3D9D3;
  --tier-red-soft-ink: #8A2E1E;
  --score-track: #E2E4DD;
  --shadow: 0 1px 2px rgba(20, 22, 18, 0.06), 0 6px 16px -10px rgba(20, 22, 18, 0.18);
  --serif: "Yu Mincho", "YuMincho", "Hiragino Mincho ProN", "Noto Serif JP", "Georgia", serif;
  --sans: "Yu Gothic", "YuGothic", "Hiragino Sans", "Noto Sans JP", "Segoe UI", sans-serif;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141815; --bg-elev: #1A1F1B; --bg-card: #1F2521; --ink: #E6E8E2; --ink-muted: #98A093;
    --ink-faint: #9AA294; --rule: #2C332C; --rule-strong: #3B443A; --accent: #63C08A; --accent-ink: #0E1F15;
    --accent-soft: #1E3A28; --accent-soft-ink: #A9E6BF; --pending: #A997D6; --pending-ink: #1F1830;
    --pending-soft: #332B4A; --pending-soft-ink: #D4C8EC; --score-track: #29302A;
    --place2-soft: #1F3A52; --place2-soft-ink: #A9CBEA; --place3-soft: #4A3D0F; --place3-soft-ink: #F0DA8C;
    --tier-red-soft: #4A241D; --tier-red-soft-ink: #F0B39F;
    --shadow: 0 1px 2px rgba(0, 0, 0, 0.3), 0 8px 20px -12px rgba(0, 0, 0, 0.5);
  }
}
:root[data-theme="dark"] {
  --bg: #141815; --bg-elev: #1A1F1B; --bg-card: #1F2521; --ink: #E6E8E2; --ink-muted: #98A093;
  --ink-faint: #9AA294; --rule: #2C332C; --rule-strong: #3B443A; --accent: #63C08A; --accent-ink: #0E1F15;
  --accent-soft: #1E3A28; --accent-soft-ink: #A9E6BF; --pending: #A997D6; --pending-ink: #1F1830;
  --pending-soft: #332B4A; --pending-soft-ink: #D4C8EC; --score-track: #29302A;
  --place2-soft: #1F3A52; --place2-soft-ink: #A9CBEA; --place3-soft: #4A3D0F; --place3-soft-ink: #F0DA8C;
  --tier-red-soft: #4A241D; --tier-red-soft-ink: #F0B39F;
  --shadow: 0 1px 2px rgba(0,0,0,0.3), 0 8px 20px -12px rgba(0,0,0,0.5);
}
:root[data-theme="light"] {
  --bg: #ECEEEA; --bg-elev: #F7F8F5; --bg-card: #FFFFFF; --ink: #1A1C1B; --ink-muted: #5B5F5A;
  --ink-faint: #656960; --rule: #D2D5CD; --rule-strong: #B7BBB1; --accent: #2C6E49; --accent-ink: #FFFFFF;
  --accent-soft: #D9E9DE; --accent-soft-ink: #1E4A31; --pending: #6B5B95; --pending-ink: #FFFFFF;
  --pending-soft: #E7E1F0; --pending-soft-ink: #4A3D6B; --score-track: #E2E4DD;
  --place2-soft: #D9E4F2; --place2-soft-ink: #1F4E79; --place3-soft: #F5EAB3; --place3-soft-ink: #6B5610;
  --tier-red-soft: #F3D9D3; --tier-red-soft-ink: #8A2E1E;
  --shadow: 0 1px 2px rgba(20,22,18,0.06), 0 6px 16px -10px rgba(20,22,18,0.18);
}

* { box-sizing: border-box; }
html { background: var(--bg); }
body {
  margin: 0; background: var(--bg); color: var(--ink); font-family: var(--sans);
  font-feature-settings: "palt"; line-height: 1.6; font-variant-numeric: tabular-nums;
}
main { max-width: 960px; margin: 0 auto; padding: 0 20px 64px; }

.jumpnav {
  position: sticky; top: 0; z-index: 20;
  background: color-mix(in srgb, var(--bg) 88%, transparent);
  backdrop-filter: blur(8px); border-bottom: 1px solid var(--rule);
}
.jumpnav-inner {
  max-width: 960px; margin: 0 auto; padding: 10px 20px;
  display: flex; align-items: center; gap: 14px; overflow-x: auto; scrollbar-width: thin;
}
.jumpnav-label { font: 700 11px/1 var(--sans); letter-spacing: 0.12em; color: var(--ink-faint); flex: none; }
.jumpnav-group {
  flex: none; font: 700 11px/1 var(--sans); letter-spacing: 0.1em; color: var(--ink-faint);
  padding-left: 8px; border-left: 1px solid var(--rule);
}
.jumpnav-group.is-pending { color: var(--pending); border-left-color: var(--pending); }
.jumpnav a {
  flex: none; font-size: 13px; color: var(--ink-muted); text-decoration: none;
  padding: 5px 10px; border-radius: 999px; border: 1px solid var(--rule);
  white-space: nowrap; transition: color .15s, border-color .15s, background .15s;
}
.jumpnav a:hover { color: var(--ink); border-color: var(--rule-strong); background: var(--bg-elev); }
.jumpnav a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.jumpnav a.is-pending { color: var(--pending); border-color: var(--pending); }
.jumpnav a.is-pending:hover { background: var(--pending-soft); }

.masthead { padding: 40px 0 28px; border-bottom: 3px double var(--rule-strong); margin-bottom: 8px; }
.eyebrow { font: 700 11px/1 var(--sans); letter-spacing: 0.16em; color: var(--accent); margin: 0 0 10px; }
.title {
  font-family: var(--serif); font-weight: 700; font-size: clamp(28px, 4.4vw, 40px);
  margin: 0 0 14px; text-wrap: balance; letter-spacing: 0.01em;
}
.lede { font-size: 15px; color: var(--ink-muted); max-width: 62ch; margin: 0 0 18px; }
.method { font-size: 13.5px; color: var(--ink-muted); }
.method summary { cursor: pointer; color: var(--ink); font-weight: 600; padding: 6px 0; }
.method ul { margin: 10px 0; padding-left: 1.2em; }
.method li { margin: 4px 0; }
.method-note { margin: 10px 0 0; max-width: 68ch; }
.method + .method { margin-top: 14px; padding-top: 14px; border-top: 1px dashed var(--rule); }
.box-method { margin-top: 14px; }
.box-method summary { font-size: 12.5px; }

.date-block { margin-top: 40px; }
.date-head {
  font-family: var(--serif); font-size: 24px; font-weight: 700; margin: 0 0 18px;
  padding-bottom: 8px; border-bottom: 1px solid var(--rule);
}
.course-block { margin-bottom: 30px; scroll-margin-top: 80px; }
.course-head {
  display: flex; align-items: baseline; gap: 8px;
  font-size: 15px; font-weight: 700; color: var(--ink); margin: 0 0 12px;
}
.course-count { font-size: 12px; font-weight: 400; color: var(--ink-faint); }

.race-grid { display: grid; grid-template-columns: 1fr; gap: 14px; }
@media (min-width: 720px) { .race-grid { grid-template-columns: 1fr 1fr; } }

.race-card {
  background: var(--bg-card); border: 1px solid var(--rule); border-radius: 10px;
  padding: 14px 14px 10px; box-shadow: var(--shadow); scroll-margin-top: 80px;
}
.race-head { display: flex; gap: 10px; align-items: baseline; margin-bottom: 10px; }
.race-num { font-family: var(--serif); font-weight: 700; font-size: 15px; color: var(--accent); flex: none; }
.race-head-text h3 { font-size: 14.5px; margin: 0; font-weight: 700; text-wrap: balance; }
.race-name-link { color: inherit; text-decoration: none; border-bottom: 1px solid transparent; }
.race-name-link:hover { border-bottom-color: var(--accent); }
.race-name-link:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.race-sub { margin: 2px 0 0; font-size: 11.5px; color: var(--ink-faint); }

.conf-badge {
  display: inline-flex; align-items: center; font: 700 10px/1 var(--sans);
  letter-spacing: 0.02em; color: var(--ink-faint); background: var(--score-track);
  border-radius: 999px; padding: 3px 8px; margin-left: 8px; vertical-align: middle; white-space: nowrap;
}
.conf-badge.is-high { background: var(--accent-soft); color: var(--accent-soft-ink); }

.model-badge {
  display: inline-flex; align-items: center; font: 700 10px/1 var(--sans);
  letter-spacing: 0.02em; color: var(--model-fg); background: var(--model-bg);
  border-radius: 4px; padding: 3px 8px; margin-left: 8px; vertical-align: middle; white-space: nowrap;
}

.pick-table-wrap { overflow-x: auto; }
.pick-table { width: 100%; border-collapse: collapse; font-size: 12.5px; }
.pick-table thead th {
  text-align: left; font-weight: 500; font-size: 10.5px; color: var(--ink-faint);
  letter-spacing: 0.04em; padding: 4px 6px; border-bottom: 1px solid var(--rule);
}
.pick-table td { padding: 5px 6px; border-bottom: 1px solid var(--rule); vertical-align: middle; }
.pick-table tbody tr:last-child td { border-bottom: none; }
.pick-table tbody tr.place-1 { background: var(--accent-soft); }
.pick-table tbody tr.place-1 td { color: var(--accent-soft-ink); }
.pick-table tbody tr.place-2 { background: var(--place2-soft); }
.pick-table tbody tr.place-2 td { color: var(--place2-soft-ink); }
.pick-table tbody tr.place-3 { background: var(--place3-soft); }
.pick-table tbody tr.place-3 td { color: var(--place3-soft-ink); }

.rank-badge {
  display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; border-radius: 50%; font-size: 11px; font-weight: 700;
  background: var(--score-track); color: var(--ink-muted);
}
.rank-1 .rank-badge { background: var(--accent); color: var(--accent-ink); }

.waku {
  display: inline-flex; align-items: center; justify-content: center;
  width: 18px; height: 18px; border-radius: 4px; font-size: 10.5px; font-weight: 700;
}
.c-uma { font-weight: 700; text-align: center; }
.c-name { font-weight: 600; white-space: nowrap; }
.mark { display: none; }
.mark.is-mark {
  display: inline-block; margin-left: 5px; font-size: 10px; font-weight: 700;
  color: var(--accent); border: 1px solid var(--accent); border-radius: 3px; padding: 0 3px;
}
.c-ninki, .c-odds, .c-weight { color: var(--ink-muted); white-space: nowrap; font-variant-numeric: tabular-nums; }
.c-score { min-width: 74px; }
.score-bar {
  display: inline-block; width: 42px; height: 6px; border-radius: 3px; background: var(--score-track);
  overflow: hidden; vertical-align: middle; margin-right: 6px;
}
.score-bar-fill { height: 100%; background: var(--accent); border-radius: 3px; }
.score-num { font-size: 11px; color: var(--ink-faint); }
.c-finish { text-align: center; white-space: nowrap; color: var(--ink-muted); font-weight: 700; }
.c-finish.is-won { color: var(--accent); }
.c-place { text-align: right; white-space: nowrap; color: var(--ink-muted); font-variant-numeric: tabular-nums; }

.conf-ladder {
  display: flex; flex-wrap: wrap; gap: 4px; margin-top: 8px;
}
.ladder-cell {
  font-size: 10.5px; color: var(--ink-muted); background: var(--score-track);
  border-radius: 6px; padding: 3px 7px; white-space: nowrap; font-variant-numeric: tabular-nums;
}
.ladder-cell b { color: var(--ink); font-weight: 700; margin-right: 3px; }
.ladder-cell.ladder-top20 { background: var(--tier-red-soft); color: var(--tier-red-soft-ink); }
.ladder-cell.ladder-top20 b { color: inherit; }
.ladder-cell.ladder-top50 { background: var(--place3-soft); color: var(--place3-soft-ink); }
.ladder-cell.ladder-top50 b { color: inherit; }

.quick-result {
  display: flex; flex-wrap: wrap; align-items: baseline; gap: 6px;
  margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--pending);
  font-size: 11.5px;
}
.quick-result-label {
  font: 700 10px/1 var(--sans); color: var(--pending); letter-spacing: 0.03em; flex: none;
}
.quick-result-body { color: var(--ink-muted); }

.result-strip {
  display: flex; flex-wrap: wrap; align-items: center; gap: 6px;
  margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--rule);
}
.result-strip-label { font: 700 10px/1 var(--sans); color: var(--ink-faint); letter-spacing: 0.03em; flex: none; }
.payout-chip {
  display: inline-flex; align-items: center; font-size: 11px; white-space: nowrap;
  padding: 3px 7px; border-radius: 6px; font-variant-numeric: tabular-nums;
}
.payout-chip.is-hit { background: var(--accent-soft); color: var(--accent-soft-ink); font-weight: 700; }
.payout-chip.is-miss { background: transparent; color: var(--ink-faint); border: 1px solid var(--rule); }

.status-badge {
  display: inline-flex; align-items: center; font: 700 10.5px/1 var(--sans);
  letter-spacing: 0.04em; padding: 3px 8px; border-radius: 999px; margin-left: 8px; vertical-align: middle;
}
.status-badge.is-pending { background: var(--pending-soft); color: var(--pending-soft-ink); border: 1px solid var(--pending); }
.status-badge.is-done { background: var(--score-track); color: var(--ink-muted); margin-left: 0; }
.status-badge.is-todo { background: transparent; color: var(--ink-faint); border: 1px solid var(--rule-strong); margin-left: 0; }

.pending-section {
  --accent: var(--pending); --accent-ink: var(--pending-ink);
  --accent-soft: var(--pending-soft); --accent-soft-ink: var(--pending-soft-ink);
  margin-top: 40px; padding: 20px 22px; border: 1px dashed var(--pending); border-radius: 12px;
  scroll-margin-top: 80px;
}
.pending-section .date-head { border-bottom-color: var(--pending); }
.pending-note { font-size: 13px; color: var(--ink-muted); max-width: 68ch; margin: 0 0 18px; }

.box-section, .fetch-board-section {
  margin: 32px 0 40px; padding: 20px 22px; background: var(--bg-elev);
  border: 1px solid var(--rule); border-radius: 12px; scroll-margin-top: 80px;
}
.box-head { font-family: var(--serif); font-size: 20px; font-weight: 700; margin: 0 0 8px; cursor: pointer; }
.box-lede { font-size: 13px; color: var(--ink-muted); max-width: 68ch; margin: 0 0 16px; }
.box-table-wrap { overflow-x: auto; }
.box-table { width: 100%; border-collapse: collapse; font-size: 13.5px; min-width: 480px; }
.box-table th {
  text-align: left; font-weight: 500; font-size: 11px; color: var(--ink-faint);
  letter-spacing: 0.04em; padding: 6px 10px; border-bottom: 1px solid var(--rule-strong);
}
.box-table td { padding: 7px 10px; border-bottom: 1px solid var(--rule); }
.box-table .bt-name { font-weight: 700; }
.box-table .num { text-align: right; }
.box-sub { color: var(--ink-faint); font-size: 11px; margin-left: 4px; }
.box-table .rate { font-weight: 700; font-family: var(--serif); }
.box-table .rate.is-plus { color: var(--accent); }
.box-table .rate.is-mid { color: var(--ink); }
.box-table .rate.is-minus { color: var(--ink-faint); }

.conf-tabs { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.conf-tab-input { position: absolute; opacity: 0; width: 1px; height: 1px; overflow: hidden; }
.conf-tab-label {
  cursor: pointer; font-size: 12.5px; color: var(--ink-muted); user-select: none;
  padding: 5px 12px; border-radius: 999px; border: 1px solid var(--rule);
  transition: color .15s, border-color .15s, background .15s;
}
.conf-tab-label:hover { border-color: var(--rule-strong); color: var(--ink); }
.conf-tab-input:checked + .conf-tab-label { background: var(--accent); color: var(--accent-ink); border-color: var(--accent); }
.conf-tab-input:focus-visible + .conf-tab-label { outline: 2px solid var(--accent); outline-offset: 2px; }
.conf-tabs-panels { flex-basis: 100%; margin-top: 14px; }
.conf-tab-panel { display: none; }
.conf-scope-label { font-size: 12px; color: var(--ink-faint); margin: 0 0 8px; }

.fetch-board-table .fb-date { font-weight: 700; white-space: nowrap; }
.fetch-board-table .fb-cell { white-space: nowrap; }
.fetch-board-table .fb-cell-muted { color: var(--ink-muted); font-size: 13px; white-space: nowrap; }
.fetch-board-note { font-size: 11.5px; color: var(--ink-faint); margin: 14px 0 0; }
.fetch-board-today {
  display: flex; align-items: center; justify-content: space-between; gap: 16px;
  background: var(--accent-soft); border: 1px solid var(--accent); border-radius: 10px;
  padding: 12px 16px; margin: 0 0 16px;
}
.fetch-board-today-text strong { font-size: 13.5px; }
.fetch-board-today-text p { margin: 3px 0 0; font-size: 11.5px; color: var(--ink-muted); max-width: 56ch; }
.copy-btn-today {
  font: 700 12px/1 var(--sans); color: var(--accent-ink, var(--bg-card)); background: var(--accent);
  border: 1px solid var(--accent); border-radius: 7px; padding: 9px 14px; margin: 0; flex: none;
}
.copy-btn-today:hover { filter: brightness(1.08); }
.copy-btn-today.is-copied { background: var(--ink); border-color: var(--ink); }
@media (max-width: 560px) {
  .fetch-board-today { flex-direction: column; align-items: stretch; }
}
.copy-btn {
  font: 600 11px/1 var(--sans); color: var(--ink-muted); background: var(--bg-card);
  border: 1px solid var(--rule-strong); border-radius: 6px; padding: 4px 8px;
  cursor: pointer; margin-left: 8px; white-space: nowrap;
}
.copy-btn:hover { border-color: var(--accent); color: var(--accent); }
.copy-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.copy-btn.is-copied { border-color: var(--accent); color: var(--accent); background: var(--accent-soft); }

.pagefoot { margin-top: 48px; padding-top: 16px; border-top: 1px solid var(--rule); }
.pagefoot p { font-size: 11.5px; color: var(--ink-faint); margin: 0; }

a:focus-visible, summary:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
"""

html_out = f"""<title>{esc(page_title)}</title>
<style>{CSS}
{conf_tabs_css}
{waku_css_rules()}
</style>
{jumpnav_html}

<main>
  <header class="masthead">
    <p class="eyebrow">NETKEIBA NEWSPAPER ANALYSIS - NAR</p>
    <h1 class="title">地方競馬 馬柱データによる着順予想</h1>
    <p class="lede">{lede_html}</p>
    <details class="method">
      <summary>通常戦モデルの内訳(予想5頭、{esc(MODEL_LABEL)})</summary>
      <ul>
        {''.join(weight_li)}
      </ul>
      <p class="method-note">
        <b>2026-08-01、box5(予想5頭表示)を独立モデル化。</b>2026-07-31までは予想4頭BOX
        モデル(winner_box4_nar.json)の重みをそのまま流用していたが、ユーザー依頼により
        box5にも独立の300パターン自由探索を実施した。プロのシステムエンジニア(サブエージェント)
        によるレビューの結果、探索は現在の標本規模(253レース)では統計的に成立しないと
        判定され(box4での診断: 300パターンの回収率の標準偏差3.3〜4.0ptに対しin-sampleの
        「勝ち幅」は説明可能な範囲、Dirichlet(1)サンプリングが等重み近傍を探索しない設計上の
        偏り等)、box5・box4・box3すべて生存{len(MODEL_ALIVE)}個のシグナルの等重みを採用している。
        測定結果・専門家レビューの詳細は「予想4頭BOXモデルの内訳」を参照。表示頭数のみ5頭の
        まま変えていない(4頭BOXは回収率検証専用に上位4頭で判定)。
        欠損値は当該項目を除いた残りの指標で重み再配分。出走取消(オッズ非公開)馬は事前に除外。
        あくまで公開データに基づく統計的な目安であり、的中を保証するものではありません。
        「穴」マークは人気6番人気以下でモデルが上位3着以内と評価した馬(人気は表示のみで
        スコア計算には不使用)。
      </p>
    </details>

    <details class="method">
      <summary>予想4頭BOXモデルの内訳({esc(BOX4_LABEL)})</summary>
      <ul>
        {''.join(box4_weight_li)}
      </ul>
      <p class="method-note" style="border-left:3px solid #2e7d32; padding-left:0.8em;">
        <b style="color:#2e7d32">2026-08-01: 300パターン自由探索を再検証した結果、等重みに
        差し戻しました。</b>2026-07-29にはユーザー判断で300パターン探索のパターン#157
        (差のある重み)を採用していましたが、データ増加(126→253レース)を受けた再探索を
        プロのシステムエンジニアがレビューした結果を踏まえ、ユーザーが等重みへの差し戻しを
        明示的に選択しました。
      </p>
      <p class="method-note">
        <b>経緯(1回目・2回目: 生存14シグナル等重み)。</b>当初は「重み探索は126レースでは
        有害」という実測(選ぶことの正味の価値{BOX4_BIAS.get('true_edge_pt', 0):+.2f}pt、
        標準偏差{BOX4_BIAS.get('true_edge_sd', 0):.2f}、平均を上回る確率
        {BOX4_BIAS.get('win_rate', 0) * 100:.0f}%)に基づき、生存14シグナルの等重みに
        固定していた。
        <br><br>
        <b>経緯(3回目・2026-07-29: 300パターン探索、pattern#157採用)。</b>ユーザーから
        「等重みをやめ、シグナルに重みを持たせ、300パターンで検証してほしい」との明示的な
        指示があり、新規3シグナル(timediff・class_ninki・weight、詳細は上のリスト参照)を
        加えた全17シグナルでDirichlet分布から300パターンの重みを生成し、全126レースで
        最良の1パターン(#157)を選んだ。学習データそのものでの市場差は+9.74ptと大幅改善に
        見えたが、12ブロックのNested LOBO OOFによる誠実な推定は市場差-11.40ptで現行の
        等重みモデル(+1.12pt)を大きく下回った。競馬専門家・シニアエンジニアの両レビューは
        不採用(現行の等重み維持)を推奨したが、ユーザーはこの測定結果を承知の上で
        「指示通り採用する」ことを明示的に選択し、pattern#157を本番採用した。
        <br><br>
        <b>経緯(4回目・2026-08-01: データ増加に伴う再探索、等重みへ差し戻し)。</b>
        検証済みレースが253レース(23ブロック)に増えたため、2026-07-29と同じ手法・同じ
        シードで300パターンを再探索した(box5にも新規に拡張)。全253レースでの
        in-sample最良は市場差{_b4_insample_excess:+.2f}ptと見えたが、23ブロックの
        Nested LOBO OOFによる誠実な推定は市場差{_b4_nested_excess:+.2f}ptにとどまり、
        現行の等重み({_b4_excess:+.2f}pt)を下回った。選択バイアス診断でも選ぶことの
        正味の価値は{_b4_edge8:+.2f}pt(標準偏差{_b4_edge_sd8:.2f})、未使用側で300パターン
        平均を上回る確率は{_b4_win8:.0f}%だった。
        ユーザーから明示的に依頼されたプロのシステムエンジニア(サブエージェント)による
        検証手法レビューでは、300パターンの回収率の標準偏差(3.3〜4.0pt)を実測した上で
        in-sampleの「勝ち幅」(+9.9〜+13.2pt)が純ノイズ(相関のある300個の最大値が生む
        期待上振れ6.6〜13.5pt)で説明可能であること、Dirichlet([1]*17)サンプリングが
        等重み近傍を一度も生成しない設計上の偏りにより<b>box4では現行の等重み自体が
        300パターン中97パーセンタイルに位置する</b>(=探索は「基準より悪いパターン群の
        最大値」を基準と比較していたに等しい)ことなどが指摘され、
        「{esc(_b4_review8.get('conclusion', ''))}」と結論づけた。
        レビューは「box5/4/3すべて300パターン探索は不採用、等重み17シグナルを維持する」
        ことを推奨し、ユーザーもこの推奨通りに進めることを選択したため、pattern#157は
        撤回し、box5(予想5頭)・box4・box3すべて等重みへ差し戻した。
        なお検証に使った253レースの77%(196レース)がレース終了後に馬柱データを取得した
        ものであることがレビュー過程で判明しており、本番(発走前取得)とは異なる情報状態で
        検証されている可能性がある(直接的な着順混入の証拠は見つからなかったが、コース別
        集計値がスナップショット依存であることは実証された)。この点への対応はユーザー判断
        により今回は保留とし、別タスクとして後日検討する。
        {_factor_v2_note(BOX4_FACTOR_V2, base_label="基準")}
      </p>
    </details>

    <details class="method">
      <summary>予想3頭BOXモデルの内訳({esc(BOX3_LABEL)})</summary>
      <ul>
        {''.join(box3_weight_li)}
      </ul>
      <p class="method-note" style="border-left:3px solid #2e7d32; padding-left:0.8em;">
        <b style="color:#2e7d32">2026-08-01: box4と同じ理由で等重みに差し戻しました。</b>
        詳細は「予想4頭BOXモデルの内訳」を参照してください。
      </p>
      <p class="method-note">
        <b>経緯(1回目・2回目: 生存14シグナル等重み)。</b>競馬専門家・シニアエンジニア
        2名のレビューを踏まえ、新規候補(休養日数適性・斤量帯適性)をLOBO OOFの二値採否で
        検証しつつ(いずれも基準を悪化させ不採用)、血統系統・馬場状態適性は実装・リーク
        リスクの懸念から見送り、探索は有害という知見(旧pattern6と同型の自由探索で楽観
        バイアス+27.7pt・選ぶことの正味の価値+0.67pt・sd7.11)に基づき生存14シグナルの
        等重みに固定していた。
        <br><br>
        <b>経緯(3回目・2026-07-29: 300パターン探索、pattern#157採用)。</b>ユーザーから
        「等重みをやめ、シグナルに重みを持たせ、300パターンで検証してほしい」との明示的な
        指示があり、新規3シグナル(timediff・class_ninki・weight、詳細は上のリスト参照)を
        加えた全17シグナルで300パターンの自由探索を実施し、4頭BOXと同じパターン#157の重みを
        採用した。学習データそのものでの市場差は+22.76ptだったが、12ブロックのNested LOBO
        OOFによる誠実な推定は市場差+13.11pt(現行の等重みは+12.17pt)で、改善幅は選択バイアス
        診断の標準偏差(8.47)に比べて小さく統計的に有意とは言えなかった。両レビューは
        「保留・実質現状維持」を推奨したが、ユーザーはこれを承知の上で採用を明示的に選択した。
        <br><br>
        <b>経緯(4回目・2026-08-01: データ増加に伴う再探索、等重みへ差し戻し)。</b>
        検証済みレースが253レース(23ブロック)に増えたため2026-07-29と同じ手法で300パターンを
        再探索したところ、全253レースでのin-sample最良は市場差{_b3_insample_excess:+.2f}ptと
        見えたが、23ブロックのNested LOBO OOFによる誠実な推定は市場差
        {_b3_nested_excess:+.2f}ptにとどまり(現行の等重みは{_b3_excess:+.2f}pt)、選択バイアス
        診断でも選ぶことの正味の価値は{_b3_edge8:+.2f}pt(標準偏差{_b3_edge_sd8:.2f})、未使用側で
        300パターン平均を上回る確率は{_b3_win8:.0f}%だった。
        ユーザー依頼によるプロのシステムエンジニア(サブエージェント)レビューでは、box4での
        診断(300パターンの回収率の標準偏差3.3〜4.0pt・等重みが300パターン中97パーセンタイル
        に位置する等)がbox3にも同様に当てはまるとして、box5/4/3すべて300パターン探索の
        不採用・等重み維持を推奨し、ユーザーもこれに従うことを選択したため、pattern#157は
        撤回し等重みへ差し戻した(詳細・検証データの信頼性に関する留意点は「予想4頭BOXモデル
        の内訳」参照)。
        旧pattern6(71レースのみで固定・一部in-sample)との比較を含む2回目の更新までの
        検証経緯は下記に残す。
        3連複はBOXした3頭がそのまま実際の1〜3着でないと当たらない1点賭けに退化するため、
        的中数が少なく回収率の振れ幅が非常に大きい点に注意。
        {_factor_v2_note(BOX3_FACTOR_V2, base_label="基準")}
      </p>
    </details>

    <details class="method">
      <summary>新馬戦モデルの内訳(JRA向け重みを暫定流用、pattern{SHINBA_PATTERN_ID})</summary>
      <ul>
        {''.join(shinba_weight_li)}
      </ul>
      <p class="method-note">
        NAR専用の新馬戦重み探索はまだ実施していない(検証済みの新馬戦レースが現状2レースのみで
        探索に足る量ではないため)。JRA向けに探索されたwinner_shinba.json(14レース、4-fold評価)の
        重みをそのまま暫定流用している。この重みはjt(騎手・厩舎)シグナルに89%集中する極端な配分で、
        JRAの母集団でも小標本への過学習の可能性が指摘されているもの。加えてNARの馬柱データには
        厩舎コメント欄・調教評価欄が存在しないため、train(1.8%)・comment(8.2%)の計約10%は
        常にNaNとなり他シグナルへ自動再配分される結果、NARでは実質ほぼjt一本の予想になっている。
        今後NARの新馬戦データが増えたら専用の重み探索に切り替えることを推奨する。
      </p>
    </details>
  </header>

  {fetch_board_section}

  {box_section or ""}

  {box_section_box4}

  {box_section_box3}

  {box_section_shinba or ""}

  {''.join(pending_sections)}

  {''.join(sections)}

  <footer class="pagefoot">
    <p>Source: netkeiba.com（馬柱・血統・コース分析等の各データを合成、地方競馬(NAR)対象） / 生成: netkeiba_pipeline</p>
  </footer>
</main>
<script>
function fallbackCopyText(text) {{
  var ta = document.createElement('textarea');
  ta.value = text;
  ta.style.position = 'fixed';
  ta.style.top = '0';
  ta.style.left = '0';
  ta.style.opacity = '0';
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  var ok = false;
  try {{ ok = document.execCommand('copy'); }} catch (e) {{ ok = false; }}
  document.body.removeChild(ta);
  return ok;
}}

document.querySelectorAll('.copy-btn').forEach(function (btn) {{
  btn.addEventListener('click', function () {{
    var template = btn.getAttribute('data-copy-template');
    var text;
    if (template) {{
      var now = new Date();
      var y = now.getFullYear();
      var m = String(now.getMonth() + 1).padStart(2, '0');
      var d = String(now.getDate()).padStart(2, '0');
      text = template.split('{{DATE}}').join(y + m + d);
    }} else {{
      text = btn.getAttribute('data-copy');
    }}
    var original = btn.textContent;
    function onCopied() {{
      btn.textContent = 'コピーしました';
      btn.classList.add('is-copied');
      setTimeout(function () {{
        btn.textContent = original;
        btn.classList.remove('is-copied');
      }}, 500);
    }}
    if (navigator.clipboard && navigator.clipboard.writeText) {{
      navigator.clipboard.writeText(text).then(onCopied).catch(function () {{
        if (fallbackCopyText(text)) onCopied();
      }});
    }} else if (fallbackCopyText(text)) {{
      onCopied();
    }}
  }});
}});

function openTargetDetails() {{
  var id = location.hash.slice(1);
  if (!id) return;
  var el = document.getElementById(id);
  if (!el) return;
  var det = el.tagName === 'DETAILS' ? el : el.querySelector(':scope > details');
  if (det && !det.open) det.open = true;
}}
window.addEventListener('hashchange', openTargetDetails);
openTargetDetails();
</script>
"""

(DATA_DIR / "prediction_report_nar.html").write_text(html_out, encoding="utf-8")
print("wrote", DATA_DIR / "prediction_report_nar.html", len(html_out), "chars")
print("total_races_scored:", total_races_scored, "total_races_verified:", total_races_verified)
print("verified_dates:", verified_dates)
print("pending_predictable_dates:", pending_predictable_dates)
