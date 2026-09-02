"""JRA公式サイトのクッション値・含水率アーカイブPDF(1開催回=1ファイル)のパース。
pdfplumberで表構造を抽出する(このプロジェクトで唯一PDFを扱うパーサー)。

実データ確認(2026-09-02、2026年1回東京競馬のPDF)で判明した表構造: 1ページに
ヘッダ3行+データ行(開催日ごとに1行)の単一テーブル。列は左から
[開催日次(「第n日」、金曜計測分は空欄), 測定月日(「1月30日」), 曜日,
使用コース(芝の内外回りローテーション、A/B/C/D等), クッション値の測定時刻,
クッション値, 含水率の測定時刻, 芝ゴール前含水率(%), 芝4コーナー含水率(%),
ダートゴール前含水率(%), ダート4コーナー含水率(%)]。年はPDF内表題
(「2026年 1回東京競馬」)から取れるが、呼び出し側が既にkai/venue/yearを知っている
(URLを組み立てた時点で判明済み)ため、パーサーへの引数として渡す設計にする。
"""
import io
import re

import pandas as pd
import pdfplumber

_DATE_RE = re.compile(r"(\d{1,2})月\s*(\d{1,2})日")

_COLUMNS = [
    "year", "venue", "kai", "date", "weekday", "day_label", "turf_course_variant",
    "cushion_time", "cushion_value", "moisture_time",
    "moisture_turf_goal_pct", "moisture_turf_corner4_pct",
    "moisture_dirt_goal_pct", "moisture_dirt_corner4_pct",
]


def _find_data_table(pdf: "pdfplumber.PDF") -> list[list[str | None]]:
    """全ページを走査し、ヘッダ行(「開催日次」「測定月日」を含む行)を持つ表を探す。
    実データでは1ページ1表だが、複数ページ(1開催回が長期にわたる場合)にまたがる
    可能性があるため、該当する全表のデータ行を連結して返す。"""
    rows: list[list[str | None]] = []
    for page in pdf.pages:
        for table in page.extract_tables():
            header_text = " ".join(str(c) for c in (table[0] if table else []) if c)
            if "開催日次" not in header_text or "測定月日" not in header_text:
                continue
            # 実データでの先頭3行はヘッダ(タイトル結合セル・列名・「ゴール前/4コーナー」の
            # 2階建てヘッダ)。データ行は3行目以降、1列目(測定月日相当)が空でない行のみ。
            for row in table[3:]:
                if row and row[1]:  # index1 = 測定月日
                    rows.append(row)
    return rows


def parse_baba_pdf(pdf_bytes: bytes, year: str, venue: str, kai: str) -> pd.DataFrame:
    """year: 'YYYY'。venue: 場名(日本語、例: "新潟"、JRA_TRACK_CODESの値)。
    kai: 開催回(文字列でも整数でもよい、出力では2桁ゼロ埋め文字列に正規化)。"""
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        raw_rows = _find_data_table(pdf)

    kai_str = f"{int(kai):02d}"
    records = []
    for row in raw_rows:
        day_label, date_text, weekday, course_variant, cushion_time, cushion_value, \
            moisture_time, turf_goal, turf_c4, dirt_goal, dirt_c4 = (row + [None] * 11)[:11]

        date_match = _DATE_RE.search(date_text or "")
        if date_match is None:
            raise ValueError(f"venue={venue} kai={kai_str}: could not parse date from {date_text!r}")
        month, day = date_match.groups()
        date_iso = f"{year}-{int(month):02d}-{int(day):02d}"

        records.append({
            "year": year, "venue": venue, "kai": kai_str, "date": date_iso,
            "weekday": (weekday or "").strip(), "day_label": (day_label or "").strip(),
            "turf_course_variant": (course_variant or "").strip(),
            "cushion_time": (cushion_time or "").strip(), "cushion_value": (cushion_value or "").strip(),
            "moisture_time": (moisture_time or "").strip(),
            "moisture_turf_goal_pct": (turf_goal or "").strip(),
            "moisture_turf_corner4_pct": (turf_c4 or "").strip(),
            "moisture_dirt_goal_pct": (dirt_goal or "").strip(),
            "moisture_dirt_corner4_pct": (dirt_c4 or "").strip(),
        })

    return pd.DataFrame(records, columns=_COLUMNS)
