"""JRA公式サイト(www.jra.go.jp/kouza/win5/result.html、netkeiba外)のWIN5キャリーオーバー
履歴を取得し、data/jra_win5_carryover_history.csv(全期間分・単一ファイル)へ保存する
スクリプト。

**このページ自体が「過去に発生した」疎な履歴一覧**(的中者が出ない週にのみ発生する
ため、2026-09-02時点で2011年〜2026年の16件のみ)であり、「今週キャリーオーバーが
あるか」を予想時点で判定できる速報ページではない(予想ファクター充足度マップTier4
項目11、詳細はsrc/netkeiba_pipeline/parsers/jra_win5_parser.pyのdocstring参照)。

日付/race_id引数は無い(ページ自体が全期間分をまとめて返すため、毎回全件を取得して
丸ごと上書きするだけでよい)。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import JRA_WIN5_CARRYOVER_CSV
from src.netkeiba_pipeline.parsers.jra_win5_parser import parse_win5_carryover_history
from src.netkeiba_pipeline.scrapers.jra_official_win5 import fetch_win5_carryover_history_html


def main() -> None:
    html = fetch_win5_carryover_history_html()
    df = parse_win5_carryover_history(html)

    JRA_WIN5_CARRYOVER_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(JRA_WIN5_CARRYOVER_CSV, index=False, encoding="utf-8")
    print(f"jra_win5_carryover: saved {len(df)} entries to {JRA_WIN5_CARRYOVER_CSV}")


if __name__ == "__main__":
    main()
