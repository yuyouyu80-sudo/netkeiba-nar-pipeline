---
name: netkeiba-fetch-date
description: 指定した開催日(YYYYMMDD)のnetKeibaレース結果・コース分析データ(枠順/脚質/騎手/調教師/種牡馬/母父/タイム指数)・コースリーディング(騎手/種牡馬/調教師TOP20)を一括取得してdata/配下にCSVで保存する。ユーザーが日付を指定してnetKeibaデータの取得・更新を依頼した際に使う。
---

# netkeiba-fetch-date

このワークスペース(netKeibaデータ収集・蓄積基盤プロジェクト)で、指定された開催日1日分のデータを
まとめて取得するスキル。実装の詳細は `C:\Users\yuyou\.claude\plans\lovely-honking-adleman.md` を参照。

## 前提条件

- `.env` に `NETKEIBA_EMAIL` / `NETKEIBA_PASSWORD` が設定済みであること。未設定の場合は
  ユーザー自身に `.env.example` をコピーして直接編集してもらう(**ID/パスワードをチャットに
  貼らせない・こちらも入力しない**)。
- `pip install -r requirements.txt` 済みであること。
- これは自動ログイン+スクレイピングであり、netKeibaの利用規約に抵触しうる行為。初回実行時は
  ユーザーの明示的な了承を得ること(既に了承済みなら再確認は不要)。

## 手順

日付は `YYYYMMDD` 形式(例: `20260719`)。ユーザーから日付が与えられたら、以下を順に実行する。

1. **レース結果の取得**
   ```
   python scripts/run_pilot.py --date {YYYYMMDD}
   ```
   出力:
   - `data/race_results/{YYYY}/{YYYYMMDD}.csv`(1頭1行)
   - `data/payouts/{YYYY}/{YYYYMMDD}.csv`(払い戻し結果。単勝/複勝/枠連/馬連/ワイド/馬単/
     3連複/3連単。race_id・bet_type・rank(複勝/ワイドの2-3組分)ごとに1行、
     combination/payout/popularity列を持つロング形式)
   - レース結果ページ(db.netkeiba.com)はレース確定後でないと存在しない。当日開催中など
     まだ結果が確定していないrace_idは `table.race_table_01 not found` で失敗するのが正常
     (バグではない)。数時間〜翌日以降に再実行すれば取得できる。
   - 既にmanifestで成功済みのrace_idは既定でスキップされる。払い戻し機能追加前に取得済みの
     日付を再取得(バックフィル)したい場合は `--force` を付ける。
   - `--circuit nar` を付けると、地方競馬14場(門別/盛岡/水沢/浦和/船橋/大井/川崎/金沢/笠松/
     名古屋/園田/姫路/高知/佐賀)を対象に取得する。出力は`data/race_results/nar/...`・
     `data/payouts/nar/...`とJRAとは別パスに保存される。`--circuit`省略時(既定`jra`)の
     挙動は従来と完全に同じ。地方競馬対応は現時点でrace_results/payoutsのみ
     (course_analysis/course_ranking/newspaperは未対応、`fetch_course_analysis.py`に
     `--circuit`は無い)。

2. **コース分析・リーディングデータの取得**(その日の全race_idに対して自動的にループする)
   ```
   python scripts/fetch_course_analysis.py --date {YYYYMMDD}
   ```
   出力:
   - `data/course_analysis/{race_id}.csv`(race_idごと。カテゴリ: waku/running_style/jockey/trainer/sire/broodmare_sire/speed_index)
   - `data/course_ranking/{race_id}.csv`(race_idごと。ranking_type: jockey/sire/trainer、各TOP20)

## 冪等性・リトライ

- 両スクリプトとも `data/_manifest/scraped_race_ids.csv` で取得済みrace_id/データ種別を管理しており、
  既に成功済みのものは自動的にスキップされる。
- 実行ログに `Failed` が出た場合(まれに一時的なネットワークエラーが発生する)、**同じコマンドを
  もう一度実行するだけでよい**。成功済みの分はスキップされ、失敗した分だけ再試行される。
- 実行後は必ずログの `Failed` 件数を確認し、0件になるまで(または同じ失敗が繰り返される場合は
  原因を調査した上で)再実行すること。

## 注意点

- `course_analysis` 系データ(cid=1〜3, coursedata, surf_summary, ranking)はnetKeibaの
  プレミアム会員限定コンテンツ。非プレミアムアカウントだと3行だけ返され
  `Premium_Regist_Box` というペイウォール表示に切り替わるため、パーサーは検知次第
  明示的に例外を出す(サイレントに不完全データを保存しない)。
- 実行完了後は、対象日の総レース数・総取得行数・manifestのfailed件数をユーザーに簡潔に報告する。
