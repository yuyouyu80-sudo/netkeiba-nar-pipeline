# -*- coding: utf-8 -*-
"""627a1a36(3連複フォーメーション列別検証ツール、旧版)のHTMLを土台に、RACESデータの
t/f/w/ur/srマークを2026-09-15版5台帳(単勝462/複勝的中率5%未満242/ワイド407/
馬連的中率3%未満2017/3連複的中率3%未満1786)ベースへ差し替えた新版HTMLを組み立てる。
旧版は一切変更せず、新版を別Artifactとして公開する前提の下ごしらえ。
"""
import json
import re
from pathlib import Path

ORIG_HTML = Path(r"C:\Users\yuyou\.claude\projects\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\tool-results\artifact-627a1a36-1789201126-7b04.html")
NEW_RACES = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\formation_tool_new_races.json")
SUMMARY = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\formation_tool_mark_summary.json")
OUT_HTML = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\formation_tool_v2.html")

html = ORIG_HTML.read_text(encoding="utf-8")
new_races = json.loads(NEW_RACES.read_text(encoding="utf-8"))
summary = json.loads(SUMMARY.read_text(encoding="utf-8"))

# 1) RACESデータ差し替え
new_races_json = json.dumps(new_races, ensure_ascii=False)
html, n_sub = re.subn(r"const RACES = \[.*?\];\n", f"const RACES = {new_races_json};\n", html,
                      count=1, flags=re.S)
assert n_sub == 1, "RACES置換に失敗"

# 2) title / meta description
OLD_TITLE = "3連複フォーメーション列別検証ツール"
NEW_TITLE = "3連複フォーメーション列別検証ツール(2026-09-15マーク更新版)"
html = html.replace(f"<title>{OLD_TITLE}</title>", f"<title>{NEW_TITLE}</title>")

OLD_DESC = ('3連複フォーメーションの1・2・3列目をそれぞれ独立した基準+チェックボックス'
           '(外し/複外/W外し/馬連外し/3連複外し、計5種)で検証できるインタラクティブツール。'
           '1列目=BOX4上位4頭に該当マーク馬を追加、2列目=BOX4上位4頭から同マーク該当馬を除外、'
           '3列目=スコア5(モデル予測)が30以上の馬プールから同マーク該当馬を除外。'
           '3列とも起点は独立。')
NEW_DESC = OLD_DESC + '2026-09-15、5種のマークを290atom候補プールで再構築した最新の消し材料台帳ベースへ更新。'
html = html.replace(f'<meta name="description" content="{OLD_DESC}">',
                    f'<meta name="description" content="{NEW_DESC}">')

# 3) h1
html = html.replace(
    "<h1>3連複フォーメーション列別マーク検証ツール</h1>",
    "<h1>3連複フォーメーション列別マーク検証ツール(マーク更新版)</h1>", 1)

# 4) dekの直後に旧版へのクロスリンク + 更新ノートを追加
OLD_CROSSLINKS = (
    '  <div class="cross-link"><a href="https://claude.ai/code/artifact/96de8129-f3f9-4676-833c-0a9a5393eafd">'
    '← 3連複フォーメーション探索(全1,050パターン一覧)</a></div>\n'
    '  <div class="cross-link"><a href="https://claude.ai/code/artifact/2e88cb0d-b081-4c68-aa95-e02795206ffe">'
    '← 馬柱データ2(元データ)</a></div>\n')
NEW_CROSSLINKS = (
    OLD_CROSSLINKS +
    '  <div class="cross-link"><a href="https://claude.ai/code/artifact/627a1a36-6038-450d-87c9-a198ed880db6">'
    '← 旧版(2026-09-12生成、169atom専用「低い方」語彙・回収率基準のマーク)</a></div>\n'
    '  <div class="rebuild-note">\n'
    '    <b>2026-09-15マーク更新</b>: t/f/w/ur/srの5マークを、JRAファクター検証データベースと共通の\n'
    '    290atom候補プールで再構築された最新の消し材料台帳へ差し替えました\n'
    '    (<a href="https://claude.ai/code/artifact/700851e2-e4c6-44cf-a1a0-f08f2b52186b">単勝回収率10%未満台帳</a>・\n'
    '    <a href="https://claude.ai/code/artifact/20c86c67-c718-4039-97ce-88b4ab509ad3">複勝的中率5%未満台帳</a>・\n'
    '    <a href="https://claude.ai/code/artifact/d6b0babd-c8b4-404e-b6d7-6e59d90b6379">ワイド回収率10%未満台帳</a>・\n'
    '    <a href="https://claude.ai/code/artifact/e156a3dc-f78b-4be5-8679-c8a5615c2d44">馬連的中率3%未満台帳</a>・\n'
    '    <a href="https://claude.ai/code/artifact/249e12c1-f2d9-4af7-8032-2a1897802eb6">3連複的中率3%未満台帳</a>)。\n'
    '    レース母集団・BOX4/BOX5モデルスコア・着順・払戻は旧版から変更していません(319レース、\n'
    '    2026-07-11〜09-06は同一)。マーク判定は各台帳のうち<b>的中率0.0%のパターンのみ</b>を使用\n'
    '    (複勝は該当パターンが0件だったため全242パターンを使用)、該当馬はいずれか1パターンでも\n'
    '    一致すればそのマークが付きます。290atomプールは「上位N位以内」系の広い条件が主体のため、\n'
    '    馬連・3連複は該当馬が population の68.0%・88.0%まで広がり、旧版(169atom専用「低い方」語彙)\n'
    '    より該当率が高くなっている点にご注意ください。\n'
    '  </div>\n')
assert OLD_CROSSLINKS in html
html = html.replace(OLD_CROSSLINKS, NEW_CROSSLINKS, 1)

# 5) rebuild-note用CSSを追加(この配色トークンに合わせる)
CSS_INSERT_MARK = "  .warnbar .icon { font-size: 16px; flex: none; }\n"
REBUILD_CSS = (
    CSS_INSERT_MARK +
    "  .rebuild-note {\n"
    "    background: var(--bg-elev); border: 1px solid var(--rule); border-radius: 10px;\n"
    "    padding: 12px 16px; margin: 0 0 16px; font-size: 12.5px; line-height: 1.7; color: var(--ink-muted);\n"
    "  }\n"
    "  .rebuild-note b { color: var(--ink); }\n"
    "  .rebuild-note a { color: var(--accent); }\n")
assert html.count(CSS_INSERT_MARK) == 1
html = html.replace(CSS_INSERT_MARK, REBUILD_CSS, 1)

# 6) notes-block内「マークの意味」段落を新しい定義に差し替え
OLD_MARK_PARA = (
    '    <p><b>マークの意味:</b> 「外し」=単勝回収率10%未満台帳(的中率0.0%・非冗長83パターン)に一致、\n'
    '    「複外」=複勝回収率25%未満台帳(的中率5%未満・非冗長54パターン)に一致、「W外し」=ワイド回収率\n'
    '    10%未満台帳(的中率0.0%・非冗長66パターン)に一致、「馬連外し」=馬連回収率10%未満台帳(的中率0.0%・\n'
    '    非冗長112パターン)に一致、「3連複外し」=3連複回収率10%未満台帳(的中率0.0%・非冗長193パターン)に\n'
    '    一致。5種とも本番score/pred_rankには使われていない参考の消し材料です。1列目ではこれらのマーク\n'
    '    該当馬を追加、2・3列目では除外に使います。</p>\n'
)
NEW_MARK_PARA = (
    '    <p><b>マークの意味(2026-09-15更新):</b> 「外し」=単勝回収率10%未満台帳(全462パターン中\n'
    '    的中率0.0%の153パターン)に一致、該当馬905/4081頭(22.2%)。「複外」=複勝的中率5%未満台帳\n'
    '    (的中率0.0%パターンが0件のため全242パターンを使用)に一致、該当馬1,092/4,081頭(26.8%)。\n'
    '    「W外し」=ワイド回収率10%未満台帳(全407パターン中的中率0.0%の257パターン)に一致、該当馬\n'
    '    1,949/4,081頭(47.8%)。「馬連外し」=馬連的中率3%未満台帳(全2,017パターン中的中率0.0%の\n'
    '    1,075パターン)に一致、該当馬2,775/4,081頭(68.0%)。「3連複外し」=3連複的中率3%未満台帳\n'
    '    (全1,786パターン中的中率0.0%の1,084パターン)に一致、該当馬3,591/4,081頭(88.0%)。いずれも\n'
    '    JRAファクター検証データベースと共通の290atom候補プールで2026-09-15に再構築されたもので、\n'
    '    本番score/pred_rankには使われていない参考の消し材料です。1列目ではこれらのマーク該当馬を追加、\n'
    '    2・3列目では除外に使います。</p>\n'
)
assert OLD_MARK_PARA in html
html = html.replace(OLD_MARK_PARA, NEW_MARK_PARA, 1)

# 7) footer に更新日を追記
OLD_FOOTER = ('<footer>build_formation_tool.py — 券種: 3連複フォーメーション固定・1列目はチェックで追加、'
             '2・3列目はそれぞれ独立の起点からチェックで除外</footer>')
NEW_FOOTER = ('<footer>build_formation_tool.py(2026-09-12生成) + '
             'rebuild_formation_tool_marks.py(2026-09-15、5マークを290atomプール最新台帳へ更新) — '
             '券種: 3連複フォーメーション固定・1列目はチェックで追加、2・3列目はそれぞれ独立の起点から'
             'チェックで除外</footer>')
assert OLD_FOOTER in html
html = html.replace(OLD_FOOTER, NEW_FOOTER, 1)

OUT_HTML.write_text(html, encoding="utf-8")
print(f"wrote {OUT_HTML} ({OUT_HTML.stat().st_size/1024/1024:.2f}MB)")
