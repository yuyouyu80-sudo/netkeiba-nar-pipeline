from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RACE_LIST_SUB_URL = "https://race.netkeiba.com/top/race_list_sub.html"
RACE_RESULT_URL = "https://db.netkeiba.com/race/{race_id}/"
RACE_API_URL = "https://race.netkeiba.com/race_api/"
# 複勝・馬連等オッズのJSON API(2026-09-02実データ確認、JRAのみ)。type=1で単勝+複勝が
# セットで返る(フロントエンドの「単勝・複勝」タブと同じグルーピング)、type=4で馬連。
# GETのみ・race.netkeiba.com固定ホスト(NAR側の同等APIは未確認、詳細は
# src/netkeiba_pipeline/scrapers/odds_api.pyのdocstring参照)。
JRA_ODDS_API_URL = "https://race.netkeiba.com/api/api_get_jra_odds.html"

# JRA公式サイト(netkeibaとは別ドメイン、ログイン不要、robots.txt全許可を確認済み
# 2026-09-02)。クッション値・含水率は開催回(「○回○○」)単位のPDFで、当該開催回が
# 完全に終了した後の翌平日に初めて公開される(実データ確認: 現在進行中の開催回は403、
# 完了済みの開催回は200)。よって事前予想向けの速報値ではなく、事後の傾向分析専用。
# URLは競馬場のローマ字名(01=sapporo〜10=kokura、実データで10場全て200を確認済み)。
JRA_OFFICIAL_BABA_PDF_URL = "https://www.jra.go.jp/keiba/baba/archive/{year}pdf/{venue}{kai:02d}.pdf"
# horse_id単位(race_id非依存)。5世代分の血統表、ログイン不要・EUC-JP(race_result.pyと同種)。
PEDIGREE_URL = "https://db.netkeiba.com/horse/ped/{horse_id}/"
# horse_id単位。血統表(/horse/ped/)とは別の、競走馬プロフィール本体ページ(生産者・馬主欄)。
# ログイン不要・EUC-JP(2026-09-02実データ確認)。
HORSE_PROFILE_URL = "https://db.netkeiba.com/horse/{horse_id}/"
# jockey_id/trainer_id単位。年度別成績・所属地・所属形態。ログイン不要・EUC-JP
# (2026-09-02実データ確認)。jockey_id/trainer_idはbias_parser.py経由でnewspaper CSVの
# bias_jockey_id/bias_trainer_id列から取得する(レース前に利用可能なIDソース)。
JOCKEY_PROFILE_URL = "https://db.netkeiba.com/jockey/{person_id}/"
TRAINER_PROFILE_URL = "https://db.netkeiba.com/trainer/{person_id}/"
# The rest of the race.netkeiba.com/race/*.html pages (newspaper/bias/data_list/
# data/speed/surf_summary/shutuba_past) are built via
# src.netkeiba_pipeline.discovery.tracks.race_url(), which switches the host to
# nar.netkeiba.com for NAR race_ids instead of using a fixed constant here.

REQUEST_DELAY_SECONDS = 1.5

DATA_DIR = PROJECT_ROOT / "data"
RAW_HTML_DIR = DATA_DIR / "raw_html"
RACE_RESULTS_DIR = DATA_DIR / "race_results"
PAYOUTS_DIR = DATA_DIR / "payouts"
LAP_TIMES_DIR = DATA_DIR / "lap_times"
COURSE_ANALYSIS_DIR = DATA_DIR / "course_analysis"
COURSE_RANKING_DIR = DATA_DIR / "course_ranking"
NEWSPAPER_DIR = DATA_DIR / "newspaper"
# horse_id単位(1ファイル1頭、race_id/circuitに非依存)。同一馬が複数レースに登場しても
# 再取得しないためのキャッシュ兼データストア。詳細はstorage/paths.pyのpedigree_csv_path参照。
PEDIGREE_DIR = DATA_DIR / "pedigree"
# horse_id単位。生産者・馬主(不変性が高いためpedigreeと同じ永続キャッシュ)。
HORSE_PROFILE_DIR = DATA_DIR / "horse_profile"
# jockey_id/trainer_id単位。年度別成績等の時系列データのため、pedigreeと異なり
# 永続キャッシュにはしない(常に上書き、詳細はfetch_person_profile.py参照)。
JOCKEY_PROFILE_DIR = DATA_DIR / "jockey_profile"
TRAINER_PROFILE_DIR = DATA_DIR / "trainer_profile"
# 複勝・馬連オッズ時系列(Tier2、2026-09-02〜)。kaisai_date単位、既存の単勝オッズ時系列
# (odds_history_{date}.csv、セッション固有scratchpad出力の予想パイプライン専用ファイル)
# とは別の、新規・恒久データとしての永続化先。race_result_csv_path/payout_csv_pathと同じ
# 「用途ごとに別ディレクトリ」の流儀に揃え、複勝・馬連で別ディレクトリにする。
# 詳細はwatch_odds.py参照。
ODDS_HISTORY_FUKU_DIR = DATA_DIR / "odds_history_fuku"
ODDS_HISTORY_UMAREN_DIR = DATA_DIR / "odds_history_umaren"
# JRA公式サイトのクッション値・含水率(Tier3、2026-09-02〜)。venue_code+kai単位
# (開催回全体で1ファイル、race_id/kaisai_date単位ではない点に注意)。
JRA_BABA_DIR = DATA_DIR / "jra_baba"
# JRA公式サイトのWIN5キャリーオーバー履歴(Tier4項目11、2026-09-02〜)。単一ファイル
# (全期間分をまとめて1ファイル、開催回単位ではない - 元データが疎な履歴のため)。
JRA_WIN5_CARRYOVER_CSV = DATA_DIR / "jra_win5_carryover_history.csv"
MANIFEST_PATH = DATA_DIR / "_manifest" / "scraped_race_ids.csv"

LOG_DIR = PROJECT_ROOT / "logs"
