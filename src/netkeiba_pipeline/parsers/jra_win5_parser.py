"""JRA公式サイト www.jra.go.jp/kouza/win5/result.html (「過去のキャリーオーバー」)のパース。

実データ確認(2026-09-02)で判明した構造: `<h2 class="heading-leftline h2">{日付}</h2>
<div class="txt">発生したキャリーオーバー<span class="carry"><strong class="red">{金額}</strong>
</span></div>`の繰り返し。**このページは「過去に発生した」事象のみを記録した疎な履歴
(2011年〜2026年で16件)であり、直近の掲載日は2026年2月1日 - 「今週のキャリーオーバーの
有無」を予想時点で判定できる速報ページではない**(WIN5は的中者が出ない週にのみ
キャリーオーバーが発生するため、そもそも毎週更新される類のデータではない)。
"""
import re

import pandas as pd

_ENTRY_RE = re.compile(
    r'<h2 class="heading-leftline h2">(.*?)</h2>\s*'
    r'<div class="txt">発生したキャリーオーバー<span class="carry"><strong class="red">(.*?)</strong>'
)
_DATE_RE = re.compile(r"(\d{4})年(\d{1,2})月(\d{1,2})日")

_COLUMNS = ["date", "carryover_amount_text"]


def parse_win5_carryover_history(html: str) -> pd.DataFrame:
    rows = []
    for date_text, amount_text in _ENTRY_RE.findall(html):
        match = _DATE_RE.search(date_text)
        if match is None:
            raise ValueError(f"could not parse date from {date_text!r}")
        year, month, day = match.groups()
        rows.append({"date": f"{year}-{int(month):02d}-{int(day):02d}", "carryover_amount_text": amount_text})
    return pd.DataFrame(rows, columns=_COLUMNS)
