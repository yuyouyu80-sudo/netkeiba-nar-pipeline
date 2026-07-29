# 場コード -> 場名。netkeiba公式のセレクタ(nar.netkeiba.com/top/calendar.html の
# <select id="select_place">)から実データで確定した値。帯広(ば)(65)はばんえい競走という
# 別方式のため対象外(ホワイトリスト方式で除外)。
NAR_TRACK_CODES: dict[str, str] = {
    "30": "門別",
    "35": "盛岡",
    "36": "水沢",
    "42": "浦和",
    "43": "船橋",
    "44": "大井",
    "45": "川崎",
    "46": "金沢",
    "47": "笠松",
    "48": "名古屋",
    "50": "園田",
    "51": "姫路",
    "54": "高知",
    "55": "佐賀",
}

# storage/writer.pyのmanifestはJRA/地方競馬でrace_idを共有する単一台帳のため、場コードが
# 重複しないことが安全性の前提になる。import時に固定で検証する。
assert not (set(NAR_TRACK_CODES) & {f"{i:02d}" for i in range(1, 11)}), (
    "NAR_TRACK_CODES contains a code that collides with a JRA venue code (01-10)"
)


def is_nar_race(race_id: str) -> bool:
    """race_id: 12桁。位置[4:6]の場コードで判別する(JRA=01〜10, 地方競馬=30〜55。
    残り桁の意味はJRA/地方競馬で異なるが、場コードの位置と桁数はどちらも共通)。"""
    if len(race_id) != 12 or not race_id.isdigit():
        raise ValueError(f"race_id={race_id!r}: expected a 12-digit race_id")
    return race_id[4:6] in NAR_TRACK_CODES


def race_site_domain(race_id: str) -> str:
    return "nar.netkeiba.com" if is_nar_race(race_id) else "race.netkeiba.com"


def race_url(race_id: str, path: str, **params: str | int) -> str:
    """race.netkeiba.com系(newspaper/bias/data_list/data/speed/surf_summary/
    shutuba_past)のURLをここ経由で組み立てる。地方競馬race_idなら自動的に
    nar.netkeiba.comへ切り替える(同一パス・同一クエリで実データ返却を確認済み)。
    db.netkeiba.com(race_result)・race_api/(holding_time)はドメイン非依存のため対象外。"""
    domain = race_site_domain(race_id)
    query = "&".join(f"{k}={v}" for k, v in params.items())
    url = f"https://{domain}/race/{path}?race_id={race_id}"
    return f"{url}&{query}" if query else url
