import requests

from config.settings import HORSE_PROFILE_URL, JOCKEY_PROFILE_URL, REQUEST_DELAY_SECONDS, TRAINER_PROFILE_URL
from src.netkeiba_pipeline.scrapers.base import fetch


def fetch_horse_profile_html(session: requests.Session, horse_id: str) -> str:
    """db.netkeiba.com/horse/{horse_id}/ (競走馬プロフィール本体、生産者・馬主欄)。
    ログイン不要ページだが、fetch_horse_profile.pyは--date指定時のrace_id列挙で
    ログイン済みセッションを使い回す(fetch_pedigree.pyのfetch_pedigree_htmlと同じ事情)。"""
    url = HORSE_PROFILE_URL.format(horse_id=horse_id)
    return fetch(session, url, encoding="euc-jp", delay_seconds=REQUEST_DELAY_SECONDS)


def fetch_jockey_profile_html(session: requests.Session, jockey_id: str) -> str:
    """db.netkeiba.com/jockey/{jockey_id}/ (年度別成績・所属地・所属形態)。ログイン不要。"""
    url = JOCKEY_PROFILE_URL.format(person_id=jockey_id)
    return fetch(session, url, encoding="euc-jp", delay_seconds=REQUEST_DELAY_SECONDS)


def fetch_trainer_profile_html(session: requests.Session, trainer_id: str) -> str:
    """db.netkeiba.com/trainer/{trainer_id}/ (年度別成績・所属地)。ログイン不要。"""
    url = TRAINER_PROFILE_URL.format(person_id=trainer_id)
    return fetch(session, url, encoding="euc-jp", delay_seconds=REQUEST_DELAY_SECONDS)
