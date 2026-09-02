# このプロジェクトについて

netKeiba(JRA中央競馬・NAR地方競馬)のデータ収集・蓄積・予想モデリング基盤。

## Claude Codeセッションの役割分担(2026-08-23〜)

このプロジェクトでは、Claude Codeの会話(セッション)を**「データ収集用」と「データ予想用」の
2つに分けて運用する**。1つの会話の中で両方を混在させない。新しい会話が始まったら、ユーザーの
最初の依頼内容からどちらの役割かを判断し、その役割の作業に留める(他方の役割の作業は、
確立済みのパイプラインが内部で呼んでいる場合(例: `jra_daily_reflect.py`が内部で
`refresh_bias.py`を呼ぶ)を除き、ユーザーから明示的に依頼されない限り自発的に持ち込まない)。

### データ収集用セッション

netKeibaから生データを取得・蓄積するだけの会話。JRA・NAR両方を対象とする。

- `scripts/fetch_newspaper.py --date/--race-id [--circuit jra|nar]` — 馬柱(newspaper)データ取得。
  **2026-08-28〜、日次収集では取得後に必ず`fetch_corner_position.py`→`fetch_marks.py`を同じ
  `--date`/`--circuit`で続けて実行する(3本セットが標準ワークフロー、詳細は
  `netkeiba-fetch-newspaper`スキル参照)**
- `scripts/run_pilot.py --date [--circuit jra|nar]` — レース結果・払戻取得
- `scripts/fetch_course_analysis.py --date` — コース分析・リーディングデータ取得(JRAのみ対応)
- `scripts/fetch_horse_weight.py` — 馬体重取得
- `scripts/fetch_quick_result.py --date` — 当日速報(1〜5着)取得
- `scripts/watch_odds.py --date` — 発走前オッズの継続監視・取得(要: 事前に`predict_pattern29.py`で
  `race_names_{date}.csv`が生成済みであること)。単勝(セッション固有scratchpad、予想パイプライン
  専用)に加え、2026-09-02〜複勝・馬連もJRAのみ取得し`data/odds_history_fuku/`・
  `data/odds_history_umaren/`(Git管理下、永続)へ追記する(NAR未対応、詳細はスクリプト冒頭docstring参照)
- `scripts/refresh_bias.py --date/--race-id` — オッズ・人気・馬体重の再取得
- `scripts/fetch_marks.py --date/--race-id [--circuit jra|nar]` — 予想印(本紙・CP・その他)取得。
  Playwright使用、事前に`fetch_newspaper.py`でCSVが生成済みであることが前提
- `scripts/fetch_corner_position.py --date/--race-id [--circuit jra|nar]` — 3・4コーナー位置取り
  (AI展開)取得。requestsのみ(Playwright不要)、事前に`fetch_newspaper.py`でCSVが
  生成済みであることが前提
- `scripts/fetch_pedigree.py --date/--race-id [--circuit jra|nar] [--force]` — 5世代血統表
  (祖先系図)取得。requestsのみ(Playwright不要、ログイン不要ページ)、出走馬のhorse_id単位で
  `data/pedigree/{horse_id}.csv`に保存(newspaper CSVへの列追加はしない)。事前に
  `fetch_newspaper.py`でCSVが生成済みであることが前提
- `scripts/fetch_horse_profile.py --date/--race-id/--ids-file [--circuit jra|nar] [--force]` —
  生産者・馬主取得(2026-09-02〜、予想ファクター充足度マップTier1)。requestsのみ、ログイン不要
  ページ、出走馬のhorse_id単位で`data/horse_profile/{horse_id}.csv`に保存(不変データのため恒久
  キャッシュ、`fetch_pedigree.py`と同じ方式)。`--date/--race-id`は事前に`fetch_newspaper.py`で
  CSVが生成済みであることが前提。`--ids-file`(2026-09-02追記)は1行1horse_idのテキストファイルで
  直接指定する方式で、newspaper CSV非依存のためdata/race_results由来のIDでの過去分バックフィルに使う
- `scripts/fetch_person_profile.py --kind jockey|trainer --date/--race-id/--ids-file [--circuit jra|nar]`
  — 騎手・調教師の年度別成績/リーディング順位/所属地・所属形態取得(2026-09-02〜、同Tier1)。
  requestsのみ、ログイン不要ページ、`data/{jockey,trainer}_profile/{id}.csv`に保存。
  時系列データのため**恒久キャッシュにせず毎回上書き**(fetch_pedigree.py/
  fetch_horse_profile.pyとは方式が異なる点に注意)。`--date/--race-id`は事前に`fetch_newspaper.py`
  (2026-09-02のbias_parser.py拡張以降)でCSVが生成済みであることが前提(`bias_jockey_id`/
  `bias_trainer_id`列をIDソースとして使う)。`--ids-file`(2026-09-02追記)はfetch_horse_profile.py
  と同じ動機で、data/race_results由来のjockey_id/trainer_idでの過去分バックフィルに使う
- `scripts/fetch_jra_baba.py --date/--race-id [--force]` — クッション値・含水率取得
  (2026-09-02〜、Tier3、JRAのみ・netkeiba外のwww.jra.go.jp)。開催回(複数週にまたがる
  ブロック)単位のPDFで、**開催回が完全終了した翌平日にしか公開されない事後データ**
  (予想時点の速報値ではない点に注意)。`data/jra_baba/{year}/{venue_code}_{kai}.csv`に
  保存、恒久キャッシュ(fetch_pedigree.pyと同方式)。進行中の開催回は403として
  正常にskip扱いされる。**2026-09-02判明: 現行パーサーは2025年1月以降のPDFレイアウトのみ
  対応、2024年以前は別レイアウト(週末ブロック単位)で0行になるため「unparseable」として
  キャッシュせずskipする(2024年以前対応は別パーサーが必要な将来課題)**
- `scripts/fetch_jra_win5_carryover.py` — WIN5キャリーオーバー履歴取得(2026-09-02〜、
  Tier4、netkeiba外のwww.jra.go.jp)。日付/race_id引数無し(全期間分をまとめて毎回
  丸ごと上書き)。`data/jra_win5_carryover_history.csv`に保存。的中者が出ず繰り越しが
  発生した週のみの疎な履歴で、「今週の有無」を予想時点で判定できるものではない
- `data/jra_course_master.csv` — JRA10場のコース仕様(直線距離・高低差・周長・幅員・
  回り方向・設定距離、2026-09-02〜、Tier4項目8)。恒久・静的な参照表のため取得
  スクリプトは無し(詳細・出典は同ファイル併設の`jra_course_master.README.md`参照)
- 関連スキル: `netkeiba-fetch-date`、`netkeiba-fetch-newspaper`、`netkeiba-fetch-marks`、
  `netkeiba-fetch-corner-position`、`netkeiba-fetch-pedigree`

### データ予想用セッション

取得済みデータを使ったモデリング・予想生成・確信度較正・レポート作成・検証を行う会話。

- `scripts/predict_pattern29.py` / `scripts/predict_top5_nar.py` — 予想生成
- `scripts/jra_daily_reflect.py` / `scripts/nar_daily_reflect.py` — 予想反映チェーン
  (内部でrefresh_bias/fetch_quick_resultを呼ぶが、目的は予想・レポートの再構築)
- `scripts/jra_verify_results.py` — 結果照合・検証
- `scripts/build_artifact_jra_axis.py` / `scripts/build_artifact_nar.py` — レポート生成
- `scripts/refresh_race_display.py` — 発走前の予想表示更新(軽量・単一レース)
- `scripts/jra_model/` 配下全体 — JRA評価基盤・重み探索・確信度較正(`jra_eval.py`/
  `jra_signals.py`/`jra_search_*.py`/`jra_confidence_calibrate.py`等)
- `scripts/nar_model/` 配下全体 — NAR側の同種一式
- Artifact公開・更新(claude.ai上のレポート)

JRA本体の日次予想パイプライン(`predict_pattern29.py`の出力・`build_artifact.py`等)は
セッション固有のscratchpad配下にあり、Git管理外(会話終了で消える)。NAR側はGit管理下・
GitHub push済み。詳細はメモリ`project_netkeiba_prediction_pipeline_location`を参照。
