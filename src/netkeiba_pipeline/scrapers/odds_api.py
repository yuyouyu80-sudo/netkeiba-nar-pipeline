"""race.netkeiba.com/api/api_get_jra_odds.html (発走前オッズのJSON API)。

実データ確認(2026-09-02、race_id=202604030301)で判明した仕様: フロントエンド
(race.netkeiba.com/odds/index.html)は`odds/index.html`自体には実オッズ値を持たず
(`<span id="odds-1_01">---.-</span>`のようなプレースホルダのみ)、ページ内の
`$.oddsUpdate({apiUrl: '/api/api_get_jra_odds.html', raceId: ..., ...})`という
JSライブラリ呼び出しが、このJSON APIをポーリングして値を後から埋め込んでいる
(riot.js経由のholding_time.htmlと同じ「静的HTMLは骨組みだけ」パターン)。

`type`パラメータで返る内訳が変わる: `type=1`は単勝(キー"1")と複勝(キー"2")が
セットで返る(フロントエンドの「単勝・複勝」タブと同じグルーピング)。`type=4`は
馬連(キー"4"、馬番組合せ4桁がキー)のみ。他の券種(枠連=3/ワイド=5/馬単=6/
3連複=7/3連単=8)も同じAPIで取得できる可能性が高いが未検証(2026-09-02時点の
予想ファクター充足度マップTier2スコープが複勝・馬連のみのため、この2種類のみ
実装・検証済み)。

odds値の内訳(実データで確認): 単勝(type "1")は`[odds, "0.0", ninki_rank]`
(2番目の要素は常に"0.0"で未使用、複勝の下限/上限と揃えるための固定枠と推測)。
複勝(type "2")は`[odds_low, odds_high, ninki_rank]`。馬連(type "4")は
`[odds, "0.0", ninki_rank]`(単勝と同型)。odds値の文字列表現は"---.-"であれば
未確定(発走直前や取消馬)、通常は小数点付き文字列(3桁区切りカンマを含みうる、
例: "1,352.5")。

**NAR側は未確認**: NAR race_id(202654072501)の`nar.netkeiba.com/odds/index.html`
ページを実際に確認したところ、`api_get_jra_odds`への参照も`oddsUpdate`呼び出しも
一切無く(古いレースでは静的にオッズ値が直接埋め込まれた表示に切り替わっている
可能性がある)、この関数群は現時点でJRAのみを対象とする。NAR側で同等の仕組みが
あるかは、実際に発走前のNARレースで再確認する必要がある(将来課題として明記)。
"""
import requests

from config.settings import JRA_ODDS_API_URL, REQUEST_DELAY_SECONDS
from src.netkeiba_pipeline.scrapers.base import fetch_json_get


def fetch_jra_odds_json(session: requests.Session, race_id: str, bet_type: int) -> dict:
    """bet_type: 1=単勝+複勝セット, 4=馬連(他は未検証)。戻り値はAPIの生JSON
    (呼び出し側でstatus=="result"を確認し、["data"]["odds"][str(bet_type or関連キー)]
    を読む)。"""
    params = {"race_id": race_id, "type": str(bet_type), "housiki": "", "update": "1"}
    return fetch_json_get(session, JRA_ODDS_API_URL, params, delay_seconds=REQUEST_DELAY_SECONDS)
