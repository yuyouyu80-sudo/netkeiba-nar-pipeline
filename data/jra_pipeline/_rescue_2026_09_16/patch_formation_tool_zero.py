# -*- coding: utf-8 -*-
"""3連複フォーメーション列別検証ツール(KDTiAXxCKYAg2HReKgNb3C = 9381e1ae)の文言を、
馬連外し/3連複外し台帳が「的中率3%未満」→「的中率0%」に改名・再構築されたことに合わせて更新する。
このツールのur/srマーク自体は元々「各台帳のうち的中率0.0%のパターンのみ」を使って計算済み
(1,075件/1,084件、今回の台帳と同じ基準)なので、RACES データ(マーク割当・的中率・回収率)は
無変更。文言(リンクの表示テキスト・パターン数の説明)だけを差し替える。
"""
from pathlib import Path

SRC = Path(r"C:\Users\yuyou\.claude\projects\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\tool-results\artifact-9381e1ae-1789480347-d6b2.html")
OUT = Path(r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------\904b9395-7511-4618-878e-3d211a238f9f\scratchpad\formation_tool_v3.html")

html = SRC.read_text(encoding="utf-8")
orig_len = len(html)

replacements = [
    # cross-link近くのrebuild-note: リンク表示テキストを新タイトルへ
    (
        '<a href="https://claude.ai/code/artifact/e156a3dc-f78b-4be5-8679-c8a5615c2d44">馬連的中率3%未満台帳</a>・\n'
        '    <a href="https://claude.ai/code/artifact/249e12c1-f2d9-4af7-8032-2a1897802eb6">3連複的中率3%未満台帳</a>)。',
        '<a href="https://claude.ai/code/artifact/e156a3dc-f78b-4be5-8679-c8a5615c2d44">馬連的中率0%台帳</a>・\n'
        '    <a href="https://claude.ai/code/artifact/249e12c1-f2d9-4af7-8032-2a1897802eb6">3連複的中率0%台帳</a>)。',
    ),
    # rebuild-note末尾に2026-09-16追記を追加
    (
        '馬連・3連複は該当馬が population の68.0%・88.0%まで広がり、旧版(169atom専用「低い方」語彙)\n'
        '    より該当率が高くなっている点にご注意ください。\n'
        '  </div>',
        '馬連・3連複は該当馬が population の68.0%・88.0%まで広がり、旧版(169atom専用「低い方」語彙)\n'
        '    より該当率が高くなっている点にご注意ください。<br><br>\n'
        '    <b>2026-09-16追記</b>: 馬連的中率3%未満台帳・3連複的中率3%未満台帳は、このツールの'
        'マーク判定が元々使っていた「的中率0.0%のみ」基準(1,075件・1,084件)に台帳側も合わせて'
        '再構築され、それぞれ<b>「馬連的中率0%台帳」「3連複的中率0%台帳」</b>として同一URLで'
        '公開されています。このツールのマーク該当馬・的中率・回収率の計算自体に変更はありません。\n'
        '  </div>',
    ),
    # notes-block: 馬連外し・3連複外しの説明文(古い母数「全2,017/1,786パターン中」を除去)
    (
        '「馬連外し」=馬連的中率3%未満台帳(全2,017パターン中的中率0.0%の\n'
        '    1,075パターン)に一致、該当馬2,775/4,081頭(68.0%)。「3連複外し」=3連複的中率3%未満台帳\n'
        '    (全1,786パターン中的中率0.0%の1,084パターン)に一致、該当馬3,591/4,081頭(88.0%)。',
        '「馬連外し」=馬連的中率0%台帳(的中率0.0%の\n'
        '    1,075パターン全件、2026-09-16に3%未満基準2,017件から再構築)に一致、該当馬2,775/4,081頭(68.0%)。'
        '「3連複外し」=3連複的中率0%台帳\n'
        '    (的中率0.0%の1,084パターン全件、2026-09-16に3%未満基準1,786件から再構築)に一致、該当馬3,591/4,081頭(88.0%)。',
    ),
]

n_total = 0
for old, new in replacements:
    n = html.count(old)
    if n != 1:
        raise SystemExit(f"NOT FOUND or not unique (n={n}): {old[:80]!r}...")
    html = html.replace(old, new, 1)
    n_total += 1

print(f"applied {n_total} replacements, orig_len={orig_len} new_len={len(html)}")
OUT.write_text(html, encoding="utf-8")
print("wrote", OUT)
