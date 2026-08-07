from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RACE_LIST_SUB_URL = "https://race.netkeiba.com/top/race_list_sub.html"
RACE_RESULT_URL = "https://db.netkeiba.com/race/{race_id}/"
RACE_API_URL = "https://race.netkeiba.com/race_api/"
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
MANIFEST_PATH = DATA_DIR / "_manifest" / "scraped_race_ids.csv"

LOG_DIR = PROJECT_ROOT / "logs"
