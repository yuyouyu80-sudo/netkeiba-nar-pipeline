---
name: netkeiba-fetch-marks
description: 指定した開催日(YYYYMMDD)または単一race_idについて、netKeibaの予想印ページ(mark_list.html、本紙・CP・その他の専門家印)を取得し、既存のdata/newspaper/{race_id}.csvへmark_raw_*(専門家ごとの生データ)とmark_honshi/mark_cp/mark_other(集計済み3列)を追加・上書きする。ユーザーが「予想印」「本紙・CP・その他」「専門家印」の取得・更新を依頼した際に使う。事前にnetkeiba-fetch-newspaperで対象race_idのCSVが生成済みであることが前提。
---

# netkeiba-fetch-marks

このワークスペースで、`scripts/fetch_marks.py` を使って開催日1日分(または単一race_id)の
予想印(本紙・CP・その他)データを取得し、既存の馬柱(newspaper)CSVへ追加するスキル。

`mark_list.html`の印テーブルはJavaScriptがDOM操作で描画するため、`requests`では空のテンプレート
しか取れない。このプロジェクトで唯一Playwrightを使うスクレイパー(既存の`login()`で得た
Cookieをブラウザコンテキストに引き継いで認証状態を再現する)。

## 前提条件

- 対象race_idについて `scripts/fetch_newspaper.py` で `data/newspaper/{race_id}.csv` が
  **既に生成済み**であること。未生成のrace_idはこのスクリプトではスキップされる(新規行は作らず、
  既存行に列を追加・上書きするだけの軽量版、`refresh_bias.py`と同じ設計)。
- `.env` に `NETKEIBA_EMAIL` / `NETKEIBA_PASSWORD` が設定済みであること。未設定の場合はユーザー
  自身に `.env.example` をコピーして直接編集してもらう(**ID/パスワードをチャットに貼らせない・
  こちらも入力しない**)。
- `pip install -r requirements.txt` 済み、かつ **`python -m playwright install chromium`を
  一度だけ実行済み**であること(通常の`pip install`だけではブラウザ本体は入らない)。
- 自動ログイン+スクレイピングであり、netKeibaの利用規約に抵触しうる行為。初回実行時はユーザーの
  明示的な了承を得ること(既に了承済みなら再確認は不要)。

## 手順

1. **取得の実行**(Playwrightのブラウザ起動を含むため1レースあたり数秒〜十数秒。日付指定は
   バックグラウンド実行を推奨)
   - 開催日1日分(JRA、既定):
     ```
     python scripts/fetch_marks.py --date {YYYYMMDD}
     ```
   - 単一race_id:
     ```
     python scripts/fetch_marks.py --race-id {race_id}
     ```
   - NAR:
     ```
     python scripts/fetch_marks.py --date {YYYYMMDD} --circuit nar
     ```
   出力: 既存 `data/newspaper/{race_id}.csv` への列追加・上書き。
   - `mark_raw_{専門家名}`: 専門家ごとの生の印(◎○▲△☆、無印は空)。専門家の顔ぶれはレース・
     開催によって変動するため、列の集合はレースごとに異なる(正常な挙動)。
   - `mark_honshi` / `mark_cp`: 「本紙」「CP予想」という名前の専門家の印(該当専門家がその
     レースに掲載されていなければ空。特にNARは「本紙」という名の専門家が存在しないレースが
     多く、その場合`mark_honshi`は空になるのが正常)。
   - `mark_other`: 「本紙」「CP予想」を除く全専門家の印を、馬ごとに◎=6/○=4/▲=3/△=2/☆=0.5点
     でスコアリングして合計し、レース内で順位付け(同点は同順位)した結果。1〜4位に◎○▲△、
     5位以下は無印、0点(=本紙・CP予想以外の誰からも印をもらっていない馬)は★。

2. **結果確認**
   - `--date` 実行時は最後に `refreshed marks: {更新頭数}/{総頭数} horses across
     {成功レース数}/{全体レース数} races for {YYYYMMDD}` のサマリ行と、スキップがあれば
     `skipped (no existing newspaper csv or no marks): ...` が出力される。
   - Windowsコンソールは日本語がcp932で文字化けするため、ログはBashの `tail`/`grep` で確認するか
     Readツールで確認する。

3. **スキップの扱い**
   - 「既存newspaper CSVが無い」でスキップされたrace_idは、先に`netkeiba-fetch-newspaper`
     スキル(`fetch_newspaper.py`)を実行してから再実行する。
   - 「mark_list.html取得結果が空」(専門家印が1人も掲載されていない)は仕様通りの正常系
     (netKeiba側にそもそもデータが無い日)であり、再実行しても解消しない。

4. **完了報告**
   - 対象日の総レース数・更新頭数・スキップ件数を簡潔に報告する。

## 注意点

- ブラウザ(Chromium)はレース毎ではなく実行全体で1回だけ起動・使い回す設計(`fetch_marks.py`が
  内部で行う、意識する必要はない)。
- 「予想ビルダー」(`id="yoso_goods_seq_builder"`)という広告枠は実専門家ではないため自動的に
  除外される。
- 印記号は実データで◎○▲△☆の5種類のみ確認済み(Icon_Honmei/Taikou/Osae/Kurosan/Hoshi)。
  未知のアイコンクラスが出た場合はログに警告を出したうえで生のクラス名を値として残す
  (`mark_other`集計では0点扱い)。
- `mark_other`の配点・順位付けロジックの詳細は
  `src/netkeiba_pipeline/parsers/mark_list_parser.py` の `summarize_marks()` を参照。
