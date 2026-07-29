---
name: netkeiba-fetch-results
description: 指定した開催日(YYYYMMDD、複数可)のnetKeibaレース結果(1頭1行)と払い戻し結果(単勝/複勝/枠連/馬連/ワイド/馬単/3連複/3連単)を取得してdata/race_results・data/payouts配下にCSVで保存する。ユーザーが日付を指定して「レース結果」「払い戻し」の取得・更新を依頼した際に使う(コース分析/リーディングデータは含まない - それが必要な場合はnetkeiba-fetch-dateを使う)。
---

# netkeiba-fetch-results

このワークスペース(netKeibaデータ収集・蓄積基盤プロジェクト)で、`scripts/run_pilot.py` を
使って開催日1日分(複数日まとめて可)のレース結果+払い戻し結果を取得するスキル。

## 前提条件

- `.env` に `NETKEIBA_EMAIL` / `NETKEIBA_PASSWORD` が設定済みであること。未設定の場合はユーザー
  自身に `.env.example` をコピーして直接編集してもらう(**ID/パスワードをチャットに貼らせない・
  こちらも入力しない**)。
- `pip install -r requirements.txt` 済みであること。
- 自動ログイン+スクレイピングであり、netKeibaの利用規約に抵触しうる行為。初回実行時はユーザーの
  明示的な了承を得ること(既に了承済みなら再確認は不要)。

## 手順

1. **取得の実行**(対象日ごとに1コマンド。複数日を頼まれたら1つのバックグラウンドジョブでループさせる)
   ```
   python scripts/run_pilot.py --date {YYYYMMDD}
   ```
   出力(同じdb.netkeiba.comのレースページから1回のフェッチで両方取得する):
   - `data/race_results/{YYYY}/{YYYYMMDD}.csv` — 1頭1行(着順/枠/馬番/馬名/性齢/斤量/騎手/
     タイム/着差/通過順/上がり3F/オッズ/人気/馬体重/調教師/馬主/賞金など)
   - `data/payouts/{YYYY}/{YYYYMMDD}.csv` — 払い戻し結果。race_id・bet_type
     (単勝/複勝/枠連/馬連/ワイド/馬単/3連複/3連単)・rank(複勝/ワイドは着順分だけ2-3行、
     他は1行)ごとに1行、combination/payout/popularity列を持つロング形式

2. **既に取得済みの日付を再取得したい場合**は `--force` を付ける(manifestでの
   スキップを無視して上書きする)。例えば払い戻し機能を追加する前に結果だけ取得済みだった
   日付をバックフィルする場合など。

3. **結果確認**
   - 実行ログの最後に `Done. success=X skipped=Y failed=Z` が出る。`failed` が0になるまで
     (または同じ失敗が繰り返される場合は原因を調査した上で)再実行する。
   - Windowsコンソールは日本語がcp932で文字化けするため、ログはBashの `tail`/`grep` で確認するか
     Readツールで確認する(printで直接目視確認しない)。
   - **当日開催中など、まだレースが確定していないrace_idは結果ページ自体が存在せず
     `table.race_table_01 not found` で失敗するのが正常**(バグではない)。全レース失敗する
     場合はこれが原因である可能性が高く、その日の開催が終わってから(通常は翌日以降)
     再実行すれば取得できる。

4. **完了報告**
   - 対象日ごとの成功数・失敗数を簡潔に報告する。
   - 取得が完全に完了したら、ユーザーに明示的に頼まれていなくても以下のPowerShellコマンドで
     ブザー音を鳴らす(このプロジェクトの他の取得スキルと同様、既定動作とする):
     ```
     [console]::beep(880,300); [console]::beep(1046,300); [console]::beep(1318,400)
     ```

## 注意点

- `run_pilot.py` はmanifest(`data/_manifest/scraped_race_ids.csv`)でrace_idごとの成功/失敗を
  管理しており、`--force` を付けない限り成功済みのrace_idは自動的にスキップされる(冪等)。
- レース結果・払い戻しとも同一のHTML取得から両方をパースするため、どちらかのパースに失敗すると
  そのrace_id全体が失敗扱いになる。
- コース分析(枠順/脚質/騎手/調教師/種牡馬/母父/タイム指数)やコースリーディングTOP20が別途
  必要な場合は、このスキルではなく `netkeiba-fetch-date` スキルを使う(`fetch_course_analysis.py`
  はプレミアム会員限定コンテンツで別のペイウォール検知ロジックが要る)。
