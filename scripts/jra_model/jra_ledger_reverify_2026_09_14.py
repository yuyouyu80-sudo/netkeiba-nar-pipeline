# -*- coding: utf-8 -*-
"""2026-09-07/09に生成された4台帳(単勝回収率10%未満・複勝回収率25%未満・ワイド回収率10%未満・
3連複回収率110%)を、Phase1a/1b(2026-09-13、履歴年範囲拡大・odds_final優先化)+
レーダー細分化(2026-09-14)を反映した現行factor_database.jsonに対して再スコアする。

背景: 4台帳を生成した元のビームサーチスクリプトはリポジトリに存在しない(2026-09-13の
監査で判明したout_marks_2026_09_11.jsonと同じ「別セッションscratchpadで実行され失われた」
パターンと推測される)。ただし各台帳のArtifact自体に「条件の中身(selected/conditions)」と
「生成当時のn_races/的中率/回収率」が埋め込まれているため、失われているのは探索アルゴリズム
だけで、既知パターンの再判定・再集計は可能(新規パターンの発見は行わない、ユーザー選択
「全パターン再スコア」)。

2方式の条件セット:
  - 3連複回収率110%台帳: 全条件がjra_factor_registry.FACTOR_GROUPSの既存id/option。
    horse_qualifies()をそのまま再利用できる。
  - 単勝10%未満・複勝25%未満・ワイド10%未満の3台帳: 「低い方」専用の169atom
    (例: ninki_low_ge13、career_form_low_lt5)を使っており、現行FACTOR_GROUPSには
    存在しない。各行のconditions(人間可読ラベル)から、対応する既存groupのsourceと
    向き(rank_le/threshold_geの反転)を機械的に導出する(_build_low_atom_evaluator）。
    唯一の例外はflop(凡走予測): 「凡走危険度(現行「凡走しにくさ」の逆)上位1〜N位」は
    既存flop_safety_rankをレース内で逆順位付けしたdanger_rankを使う特別処理。

決済はjra_backtest.settle()を再利用(可変長の「条件を満たした馬の集合」をそのまま
1レース分のcombos_for()に渡すだけで、単勝/複勝/ワイド/3連複のstake/returnが一括で出る)。
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

LIB_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = LIB_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"

sys.path.insert(0, str(LIB_DIR))
import jra_backtest as JB  # noqa: E402
import jra_dataset_wide as JDW  # noqa: E402
import jra_factor_registry as FR  # noqa: E402

TOOL_RESULTS_DIR = (Path.home() / ".claude" / "projects"
                    / "c--Users-yuyou-Desktop--------" / "904b9395-7511-4618-878e-3d211a238f9f"
                    / "tool-results")

LEDGERS = {
    "単勝回収率10%未満台帳": {
        "html": TOOL_RESULTS_DIR / "artifact-700851e2-1789046904-eac0.html",
        "bet_type": "単勝", "family": "low",
    },
    "複勝回収率25%未満台帳": {
        "html": TOOL_RESULTS_DIR / "artifact-20c86c67-1789050351-3ccb.html",
        "bet_type": "複勝", "family": "low",
    },
    "ワイド回収率10%未満台帳": {
        "html": TOOL_RESULTS_DIR / "artifact-d6b0babd-1789077338-c3cb.html",
        "bet_type": "ワイド", "family": "low",
    },
    "3連複回収率110%台帳": {
        "html": TOOL_RESULTS_DIR / "artifact-5aac2d4a-1788956850-dd7d.html",
        "bet_type": "3連複", "family": "existing",
    },
}


def load_data_blob(html_path: Path) -> dict:
    html = html_path.read_text(encoding="utf-8")
    m = re.search(r'<script type="application/json" id="data-blob">(.*?)</script>', html, re.S)
    return json.loads(m.group(1))


# --------------------------------------------------------------------- 「低い方」atom評価器

# 既存FACTOR_GROUPSのlabel(「(低い方)」除去後)→(gid, group)の逆引き。
_LABEL_TO_GROUP = {g["label"]: (gid, g) for gid, g in FR.FACTOR_GROUPS.items()}

_RANK_SUFFIX_RE = re.compile(r"(\d+)(位以下|番人気以下)$")
_PCT_SUFFIX_RE = re.compile(r"(\d+)%未満")  # ラベル全文中のどこかに出現(例: 「通算複勝率(全キャリア)が30%未満」)
_PCT_ZERO_RE = re.compile(r"0%\(該当実績なし\)")
_FLOP_DANGER_RE = re.compile(r"^凡走危険度.*上位1[〜~](\d+)位$")


def _strip_low_suffix(label: str) -> str:
    return label[:-5] if label.endswith("(低い方)") else label


def build_low_condition_fn(group_label: str, value_label: str):
    """「低い方」台帳の1条件(conditions内の{group, value})から、
    horse(dict) -> bool の判定関数を1つ作る。flop以外は既存FACTOR_GROUPSのsource/kindを
    そのまま使い、向きだけ反転させる(rank_le -> value>=N、threshold_ge -> value<N)。
    group_labelが「(低い方)」で終わらない場合(odds_band/grade_best/waku等、既存の
    category_in系グループをそのまま流用しているケース)は、既存options内のlabel一致で
    直接evaluate_option()に委譲する(2026-09-14、単勝オッズ帯等で発覚し追加)。"""
    # --- flop(凡走予測)の特別処理: 「凡走しにくさ」の逆順位(危険度)---
    m = _FLOP_DANGER_RE.match(value_label)
    if m:
        n = int(m.group(1))

        def fn(horse: dict) -> bool:
            dr = horse.get("_flop_danger_rank")
            return dr is not None and dr <= n
        return fn

    # --- 「(低い方)」が付いていない = 既存グループ・既存optionをそのまま流用 ---
    if not group_label.endswith("(低い方)"):
        if group_label not in _LABEL_TO_GROUP:
            raise ValueError(f"未知のgroup label(既存FACTOR_GROUPSに無い): {group_label!r}")
        gid, group = _LABEL_TO_GROUP[group_label]
        opt = next((o for o in group["options"] if o["label"] == value_label), None)
        if opt is None:
            raise ValueError(f"既存option内にlabel一致なし: group={group_label!r} value={value_label!r}")

        def fn(horse: dict) -> bool:
            return FR.evaluate_option(horse, group, opt)
        return fn

    base_label = _strip_low_suffix(group_label)
    if base_label not in _LABEL_TO_GROUP:
        raise ValueError(f"未知のgroup label(低い方対応表に無い): {group_label!r} (base={base_label!r})")
    gid, group = _LABEL_TO_GROUP[base_label]
    source, kind = group["source"], group["kind"]

    m = _RANK_SUFFIX_RE.search(value_label)
    if m and kind == "rank_le":
        n = int(m.group(1))

        def fn(horse: dict) -> bool:
            v = horse.get(source)
            return v is not None and v >= n
        return fn

    if kind == "threshold_ge":
        if _PCT_ZERO_RE.search(value_label):
            def fn(horse: dict) -> bool:
                v = horse.get(source)
                return v is not None and v < 1
            return fn
        m = _PCT_SUFFIX_RE.search(value_label)
        if m:
            n = int(m.group(1))

            def fn(horse: dict) -> bool:
                v = horse.get(source)
                return v is not None and v < n
            return fn

    raise ValueError(f"未対応のvalueパターン: group={group_label!r} value={value_label!r} kind={kind!r}")


# --------------------------------------------------------------------- レース側の前処理

def attach_flop_danger_rank(races: list) -> None:
    """既存flop_safety_rank(1=最も凡走しにくい)を各レース内で逆順位付けし、
    horse dictに"_flop_danger_rank"(1=最も凡走しやすい)として追加する(破壊的、
    再検証専用の一時フィールド)。"""
    for race in races:
        horses = race["horses"]
        vals = pd.Series([h.get("flop_safety_rank") for h in horses], dtype="float64")
        # flop_safety_rankは既に「1=最も安全」の昇順ランクなので、逆順(危険度)にするには
        # 同じ値をdescendingで順位付けし直す必要がある(ascendingだと実質no-opになるバグを
        # 2026-09-14の再検証テストで発見・修正)。
        danger = vals.rank(ascending=False, method="min", na_option="bottom")
        for h, d in zip(horses, danger):
            h["_flop_danger_rank"] = None if pd.isna(d) else int(d)


# --------------------------------------------------------------------- パターン再スコア

# 券種ごとに組合せを1つ作るのに必要な最少頭数(単勝/複勝は1頭、ワイドは2頭、3連複は3頭)。
# これを満たさないレースは「条件に該当する馬はいるが、その券種の買い目が組めない」ため
# 対象レース数(n_races)に含めない(2026-09-14、再検証時に発見: この下限を掛けないと
# 1〜2頭しか該当しないレースまでn_races・点数に混入し、台帳の再現性が大きく崩れる)。
_MIN_HORSES_FOR_BET = {"単勝": 1, "複勝": 1, "ワイド": 2, "3連複": 3}


def score_pattern(races: list, actual: dict, bet_type: str, family: str,
                  selected: dict, conditions: list) -> dict:
    bt_idx = JB.BET_TYPES.index(bet_type)
    min_horses = _MIN_HORSES_FOR_BET[bet_type]
    if family == "existing":
        sel = {g: {v} for g, v in selected.items()}
    else:
        fns = [build_low_condition_fn(c["group"], c["value"]) for c in conditions]

    n_races = 0
    total_stake = 0
    total_return = 0
    hit_races = 0
    for race in races:
        horses = race["horses"]
        if family == "existing":
            qualifying = [h for h in horses if FR.horse_qualifies(h, sel)]
        else:
            qualifying = [h for h in horses if all(fn(h) for fn in fns)]
        qualifying = [h for h in qualifying if h.get("umaban") is not None]
        if len(qualifying) < min_horses:
            continue
        n_races += 1
        umabans = [h["umaban"] for h in qualifying]
        wakus = [h["waku"] for h in qualifying if h.get("waku") is not None]
        a = actual.get(race["race_id"], {})
        stake_vec, return_vec = JB.settle(a, umabans, wakus)
        s, r = int(stake_vec[bt_idx]), int(return_vec[bt_idx])
        total_stake += s
        total_return += r
        if r > 0:
            hit_races += 1
    n_points = total_stake // JB.UNIT
    return {
        "n_races": n_races, "n_points": n_points, "hit_races": hit_races,
        "hit_rate_pct": round(hit_races / n_races * 100, 4) if n_races else 0.0,
        "return_rate_pct": round(total_return / total_stake * 100, 4) if total_stake else 0.0,
        "stake": total_stake, "return": total_return,
    }


def main():
    print("factor_database.json読み込み中...", flush=True)
    fdb = json.loads((DATA_DIR / "factor_database.json").read_text(encoding="utf-8"))
    races = fdb["races"]
    attach_flop_danger_rank(races)

    print("actual(払戻テーブル)読み込み中...", flush=True)
    wide = JDW.load(rebuild=False)
    actual = wide["actual"]

    results = {}
    for name, spec in LEDGERS.items():
        print(f"\n=== {name} ===", flush=True)
        blob = load_data_blob(spec["html"])
        rows = blob["rows"]
        out_rows = []
        for row in rows:
            new = score_pattern(races, actual, spec["bet_type"], spec["family"],
                                row["selected"], row["conditions"])
            out_rows.append({
                "rank_old": row["rank"], "selected": row["selected"], "conditions": row["conditions"],
                "old": {k: row[k] for k in
                       ("n_races", "n_points", "hit_races", "hit_rate_pct", "return_rate_pct",
                        "stake", "return")},
                "new": new,
            })
        results[name] = {"meta_old": blob["meta"], "rows": out_rows}
        print(f"  {len(rows)}パターン再スコア完了", flush=True)

    out_path = (Path(__file__).resolve().parent.parent.parent / "data" / "jra_pipeline"
               / "jra_ledger_reverify_2026_09_14_raw.json")
    out_path.write_text(json.dumps(results, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
