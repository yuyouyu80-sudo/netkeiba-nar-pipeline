# -*- coding: utf-8 -*-
"""Stage 0-D(3列目改良プラン、2026-09-16): 3連複フォーメーション列別検証ツールのHTMLを、
Stage 0-Cの基礎データ(jra_formation_base_2026_09_16.json)+
scripts/jra_model/templates/formation_tool_template.html から一発生成する。

外部キャッシュ(claude.aiのArtifact取得結果HTML)や過去のパッチチェーン
(assemble_formation_tool_v2.py→patch_formation_tool_zero.py)には一切依存しない。
テンプレート自体に、旧パッチチェーンが適用していたテキスト差分(タイトル・notes-block・
rebuild-note追記)をあらかじめ手作業で畳み込んである。

マーク(t/f/w/ur/sr)の計算方式はrebuild_formation_tool_marks.pyと同じ
(各台帳の的中率0.0%パターンをOR結合したベクトル化atom_mask、0.0%パターンが0件の券種は
全パターンにフォールバック)だが、母集団はStage 0-Cの314レース(障害戦除外・fail-fast結合
済み)を使う。
"""
import json
import pathlib
import sys

import numpy as np

PROJECT_ROOT = pathlib.Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
LIB_DIR = PROJECT_ROOT / "scripts" / "jra_model"
DATA_DIR = PROJECT_ROOT / "data" / "jra_pipeline"
TEMPLATE_PATH = LIB_DIR / "templates" / "formation_tool_template.html"
sys.path.insert(0, str(LIB_DIR))

import jra_factor_registry as FR  # noqa: E402

BASE_PATH = DATA_DIR / "jra_formation_base_2026_09_16.json"
# 検証用出力(再生成可能なので恒久保存しない)。scratchpadが無い実行環境ではDATA_DIR直下に出す。
SCRATCHPAD = pathlib.Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad"
)
OUT_HTML = (SCRATCHPAD if SCRATCHPAD.exists() else DATA_DIR) / "formation_tool_v4_onepass_VERIFY_ONLY.html"

LEDGER_FILES = {
    "t": (DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "単勝"),
    "w": (DATA_DIR / "jra_ledger_search_low_2026_09_15_result.json", "ワイド"),
    "f": (DATA_DIR / "jra_ledger_search_fukusho_hitrate_2026_09_15_result.json", "複勝"),
    "ur": (DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "馬連"),
    "sr": (DATA_DIR / "jra_ledger_search_low_hitrate_2026_09_15_result.json", "3連複"),
}
LEDGER_ARTIFACT_URLS = {
    "t": "https://claude.ai/code/artifact/700851e2-e4c6-44cf-a1a0-f08f2b52186b",
    "f": "https://claude.ai/code/artifact/20c86c67-c718-4039-97ce-88b4ab509ad3",
    "w": "https://claude.ai/code/artifact/d6b0babd-c8b4-404e-b6d7-6e59d90b6379",
    "ur": "https://claude.ai/code/artifact/e156a3dc-f78b-4be5-8679-c8a5615c2d44",
    "sr": "https://claude.ai/code/artifact/249e12c1-f2d9-4af7-8032-2a1897802eb6",
}
MARK_LABEL = {"t": "外し", "f": "複外", "w": "W外し", "ur": "馬連外し", "sr": "3連複外し"}

print("loading Stage 0-C base dataset...")
base = json.loads(BASE_PATH.read_text(encoding="utf-8"))
races = base["races"]
meta0c = base["meta"]
n_races = len(races)
dates = meta0c["dates"]
date_range = f"{dates[0][:4]}-{dates[0][4:6]}-{dates[0][6:]}〜{dates[-1][4:6]}-{dates[-1][6:]}"

flat_horses = []
horse_index = {}  # (race_id, umaban) -> index
for r in races:
    for h in r["horses"]:
        horse_index[(r["race_id"], h["umaban"])] = len(flat_horses)
        flat_horses.append(h["atoms"])
n = len(flat_horses)
print(f"races={n_races}  horses={n}")

print("building atom_masks (290atom)...")
atom_masks = {}
for gid, group in FR.FACTOR_GROUPS.items():
    for opt in group["options"]:
        key = (gid, opt["id"])
        atom_masks[key] = np.array([FR.evaluate_option(h, group, opt) for h in flat_horses], dtype=bool)
print(f"atoms: {len(atom_masks)}")

print("computing t/f/w/ur/sr marks from 2026-09-15/16 ledgers...")
mark_arrays = {}
mark_summary = {}
for mk, (path, bt) in LEDGER_FILES.items():
    ld = json.loads(path.read_text(encoding="utf-8"))
    all_rows = ld["bet_types"][bt]["rows"]
    zero_rows = [row for row in all_rows if row.get("hit_rate_pct", 0.0) == 0.0]
    rows = zero_rows if zero_rows else all_rows
    cum = np.zeros(n, dtype=bool)
    for row in rows:
        pat_mask = np.ones(n, dtype=bool)
        for gid, oid in row["selected"].items():
            pat_mask &= atom_masks[(gid, oid)]
        cum |= pat_mask
    mark_arrays[mk] = cum
    mark_summary[mk] = {
        "bet_type": bt, "total_patterns": len(all_rows), "used_patterns": len(rows),
        "used_is_zero_subset": bool(zero_rows), "n_marked": int(cum.sum()), "n": n,
    }
    print(f"  {mk}({bt}, 全{len(all_rows)}件中0%={len(zero_rows)}件使用): "
          f"該当馬 {int(cum.sum())}/{n} ({cum.sum()/n*100:.1f}%)")

print("assembling RACES json...")
races_out = []
for r in races:
    horses = []
    for h in r["horses"]:
        idx = horse_index[(r["race_id"], h["umaban"])]
        sc5 = h.get("score5")
        rank4 = h.get("rank4")
        horses.append({
            "u": h["umaban"], "n": h.get("horse_name"),
            "b": bool(rank4 is not None and rank4 <= 4),
            "t": bool(mark_arrays["t"][idx]), "f": bool(mark_arrays["f"][idx]),
            "w": bool(mark_arrays["w"][idx]), "ur": bool(mark_arrays["ur"][idx]),
            "sr": bool(mark_arrays["sr"][idx]),
            "s": sc5, "fp": h.get("finish_pos"),
        })
    races_out.append({
        "id": r["race_id"], "d": r["kaisai_date"], "c": r["racecourse"],
        "rn": r.get("race_number"), "name": r["race_name"], "blk": r["block_id"],
        "h": horses, "pay": r["pay_sanrenpuku"],
    })
RACES_JSON = json.dumps(races_out, ensure_ascii=False, separators=(",", ":"))
print(f"embedded {len(races_out)} races, json size {len(RACES_JSON):,} chars")

# ------------------------------------------------------------ 文言組み立て(v2/v3パッチの
# テキスト差分をテンプレートへ畳み込んだ上で、2026-09-16の基盤刷新(障害戦除外・fail-fast
# 結合への変更)を追記する)
TITLE = "3連複フォーメーション列別検証ツール(2026-09-15マーク更新版)"
DESCRIPTION = (
    "3連複フォーメーションの1・2・3列目をそれぞれ独立した基準+チェックボックス"
    "(外し/複外/W外し/馬連外し/3連複外し、計5種)で検証できるインタラクティブツール。"
    "1列目=BOX4上位4頭に該当マーク馬を追加、2列目=BOX4上位4頭から同マーク該当馬を除外、"
    "3列目=スコア5(モデル予測)が30以上の馬プールから同マーク該当馬を除外。"
    "3列とも起点は独立。2026-09-15、5種のマークを290atom候補プールで再構築した最新の"
    "消し材料台帳ベースへ更新。"
)
H1 = "3連複フォーメーション列別マーク検証ツール(マーク更新版)"

mark_links = "・\n    ".join(f'<a href="{LEDGER_ARTIFACT_URLS[k]}">{MARK_LABEL[k]}台帳</a>' for k in ("t", "f", "w", "ur", "sr"))
REBUILD_NOTE_BLOCK = (
    f"<b>2026-09-15マーク更新</b>: t/f/w/ur/srの5マークを、JRAファクター検証データベースと共通の\n"
    f"    290atom候補プールで再構築された最新の消し材料台帳へ差し替えました\n"
    f"    ({mark_links})。\n"
    f"    レース母集団・BOX4/BOX5モデルスコア・着順・払戻は旧版から変更していません({n_races}レース、\n"
    f"    {date_range}は同一)。マーク判定は各台帳のうち的中率0.0%のパターンのみを使用\n"
    f"    (複勝は該当パターンが0件だったため全{mark_summary['f']['total_patterns']}パターンを使用)、"
    f"該当馬はいずれか1パターンでも\n"
    f"    一致すればそのマークが付きます。290atomプールは「上位N位以内」系の広い条件が主体のため、\n"
    f"    馬連・3連複は該当馬が population の{mark_summary['ur']['n_marked']/n*100:.1f}%・"
    f"{mark_summary['sr']['n_marked']/n*100:.1f}%まで広がり、旧版(169atom専用「低い方」語彙)\n"
    f"    より該当率が高くなっている点にご注意ください。<br><br>\n"
    f"    <b>2026-09-16追記(マーク文言)</b>: 馬連的中率3%未満台帳・3連複的中率3%未満台帳は、\n"
    f"    このツールのマーク判定が元々使っていた「的中率0.0%のみ」基準"
    f"({mark_summary['ur']['used_patterns']:,}件・{mark_summary['sr']['used_patterns']:,}件)に台帳側も\n"
    f"    合わせて再構築され、それぞれ「馬連的中率0%台帳」「3連複的中率0%台帳」として同一URLで\n"
    f"    公開されています。このツールのマーク該当馬・的中率・回収率の計算自体に変更はありません。"
    f"<br><br>\n"
    f"    <b>2026-09-16追記(基盤刷新)</b>: レース母集団・生成パイプラインを、キャッシュ済みHTMLへの\n"
    f"    パッチ適用方式から一次データ(newspaper CSV・race_results・factor_database.json)からの\n"
    f"    独立ワンパス生成へ全面的に再設計しました。この過程で、旧版が対象にしていた319レースのうち\n"
    f"    5レース(「障害3歳以上OP」等の障害戦)がfactor_databaseと結合できず、5マークとも黙って\n"
    f"    「該当なし」扱いになっていた既知バグを発見・是正しています。障害戦は本ツールの対象外と\n"
    f"    明示的に決定した上で母集団から除外し、現在は{n_races}レース(該当馬{n}頭、全頭が\n"
    f"    factor_databaseとの結合に成功、fail-fast検証済み)を対象としています。BOX4/BOX5モデルの\n"
    f"    priorsはbuild_data2.py以来の方式(winner_v3.json/winner_box4.json内の固定priors)を明示的に\n"
    f"    踏襲し、旧版との連続性を保っています。"
)

WARNBAR_TEXT = (
    f"軸に使うBOX4モデル、比較に使う「外し」「複外」「W外し」「馬連外し」「3連複外し」の各マークは、\n"
    f"      いずれもこの{n_races}レース({date_range}、障害戦を除く通常戦)という同一母集団を対象にした網羅的探索で\n"
    f"      得られたものです。チェックボックスを変えるたびに違う「パターン」を試すことになり、\n"
    f"      多数の組み合わせを試すほど選択バイアスで良い数値が出やすくなります。表示される的中率・\n"
    f"      回収率は<b>参考値</b>としてご覧ください。"
)

def _mark_para_piece(mk):
    s = mark_summary[mk]
    pct = s["n_marked"] / n * 100
    base_txt = f"「{MARK_LABEL[mk]}」="
    if s["used_is_zero_subset"]:
        cond = f"{s['bet_type']}的中率0%台帳(的中率0.0%の{s['used_patterns']:,}パターン全件)"
    else:
        cond = f"{s['bet_type']}的中率5%未満台帳(的中率0.0%パターンが0件のため全{s['total_patterns']:,}パターンを使用)"
    return f"{base_txt}{cond}に一致、該当馬{s['n_marked']:,}/{n:,}頭({pct:.1f}%)"

MARK_MEANING_PARA = (
    "<b>マークの意味:</b> " + "。".join(_mark_para_piece(k) for k in ("t", "f", "w", "ur", "sr")) + "。"
    "いずれもJRAファクター検証データベースと共通の290atom候補プールで2026-09-15に再構築されたもので、"
    "本番score/pred_rankには使われていない参考の消し材料です。1列目ではこれらのマーク該当馬を追加、"
    "2・3列目では除外に使います。"
)

POPULATION_PARA = (
    f"<b>母集団:</b> 馬柱データ2と同じ通常戦のみ、ただし2026-09-16の基盤刷新で障害戦5レースを\n"
    f"    明示的に除外し{n_races}レース({date_range})が対象です(旧版は障害戦を含む319レースを\n"
    f"    対象としていましたが、当該5レースの馬は5マークとも実際には未結合のまま「該当なし」に\n"
    f"    デフォルトされていた既知バグがありました)。全ての計算はこのページの読み込み後、\n"
    f"    ブラウザ内でその場で行われます(サーバー通信なし)。"
)

FOOTER_TEXT = (
    "jra_formation_tool_build_2026_09_16.py(一次データからの独立ワンパス生成、旧build_formation_tool.py"
    "(2026-09-12)+rebuild_formation_tool_marks.py(2026-09-15)+patch_formation_tool_zero.py(2026-09-16)の"
    "パッチチェーンを置換) — 券種: 3連複フォーメーション固定・1列目はチェックで追加、2・3列目はそれぞれ"
    "独立の起点からチェックで除外"
)

template = TEMPLATE_PATH.read_text(encoding="utf-8")
html = (template
        .replace("__TITLE__", TITLE)
        .replace("__DESCRIPTION__", DESCRIPTION)
        .replace("__H1__", H1)
        .replace("__REBUILD_NOTE_BLOCK__", REBUILD_NOTE_BLOCK)
        .replace("__WARNBAR_TEXT__", WARNBAR_TEXT)
        .replace("__MARK_MEANING_PARA__", MARK_MEANING_PARA)
        .replace("__POPULATION_PARA__", POPULATION_PARA)
        .replace("__FOOTER_TEXT__", FOOTER_TEXT)
        .replace("__RACES_JSON__", RACES_JSON))

OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
OUT_HTML.write_text(html, encoding="utf-8")
print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size:,} bytes)")

summary_out = {
    "n_races": n_races, "n_horses": n, "date_range": date_range,
    "mark_summary": mark_summary,
    "old_tool_ur_true_count_for_reference": 2775,
    "old_tool_sr_true_count_for_reference": 3591,
}
(DATA_DIR / "jra_formation_tool_build_2026_09_16_result.json").write_text(
    json.dumps(summary_out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(summary_out, ensure_ascii=False, indent=2))
