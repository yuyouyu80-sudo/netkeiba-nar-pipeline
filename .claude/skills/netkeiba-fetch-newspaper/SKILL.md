---
name: netkeiba-fetch-newspaper
description: 指定した開催日(YYYYMMDD)または単一race_idについて、馬柱(新聞)ベースの1頭1行ワイドCSV(厩舎コメント/調教タイム/血統ビーム/コース分析/コースデータ/スピード指数/持続時間/data.html各種内訳など)を取得しdata/newspaper/{race_id}.csvに保存する。ユーザーが日付を指定して「新聞データ」「馬柱データ」の取得・更新を依頼した際に使う。完了時はブザー音で知らせる。
---

# netkeiba-fetch-newspaper

このワークスペース(netKeibaデータ収集・蓄積基盤プロジェクト)で、`scripts/fetch_newspaper.py` を
使って開催日1日分(または単一race_id)の馬柱(新聞)データを取得するスキル。

## 前提条件

- `.env` に `NETKEIBA_EMAIL` / `NETKEIBA_PASSWORD` が設定済みであること。未設定の場合はユーザー
  自身に `.env.example` をコピーして直接編集してもらう(**ID/パスワードをチャットに貼らせない・
  こちらも入力しない**)。
- `pip install -r requirements.txt` 済みであること。
- 自動ログイン+スクレイピングであり、netKeibaの利用規約に抵触しうる行為。初回実行時はユーザーの
  明示的な了承を得ること(既に了承済みなら再確認は不要)。

## 手順

1. **取得の実行**(1レースあたり約1分かかるため、日付指定は必ずバックグラウンド実行にする)
   - 開催日1日分:
     ```
     python scripts/fetch_newspaper.py --date {YYYYMMDD}
     ```
   - 単一race_id:
     ```
     python scripts/fetch_newspaper.py --race-id {race_id}
     ```
   出力: `data/newspaper/{race_id}.csv`(1頭1行、newspaper/shutuba_past/speed_index/holding_time/
   bias/course_analysis(cid0-3)/coursedata(cid1,4)/surf_summary既定+6種のkey1+key2組合せ/
   concerned/data.html内訳(distance/course/condition/others/cushion/baba_water)を全てumaban
   またはhorse_idで結合したワイドCSV)。

2. **結果確認**
   - `--date` 実行時は最後に `{成功数}/{全体数} races written for {YYYYMMDD}` のサマリ行と、
     失敗があれば `Failed (N): race_id1, race_id2, ...` が出力される。
   - Windowsコンソールは日本語がcp932で文字化けするため、ログはBashの `tail`/`grep` で確認するか
     Readツールで確認する(printで直接目視確認しない)。

3. **失敗レースの再取得**
   - `fetch_newspaper.py` にはmanifestベースのスキップ機構が無いため、失敗したrace_idだけを
     `--race-id` で個別に再実行する(1レースの出力は独立したCSV全体の書き込みなので、再実行しても
     安全)。
   - 典型的な失敗要因は一時的なネットワークエラー(DNS解決失敗・コネクションリセットなど)で、
     同じレースを再実行すればほぼ解消する。
   - 失敗が0件になるまで(または同じ失敗が繰り返される場合は原因を調査した上で)再実行する。

4. **完了報告**
   - 対象日の総レース数・成功数・(あれば)再取得した失敗レースの内訳を簡潔に報告する。
   - **取得が完全に完了したら、ユーザーに明示的に頼まれていなくても以下のPowerShellコマンドで
     ブザー音を鳴らす**(このワークフローではユーザーが毎回ブザー通知を求めているため、既定動作
     とする):
     ```
     [console]::beep(880,300); [console]::beep(1046,300); [console]::beep(1318,400)
     ```

## 注意点

- 各レースの取得には`newspaper.html`だけでなく計20以上のページへのリクエストが発生するため、
  1レースあたり約1分かかる。開催日1日(最大36レース程度)では30〜60分程度を見込む。
- `PredictRap_Table`(前半3F/後半3F予想ラップ)はこのアカウントの会員レベルでは閲覧不可のページが
  あり、その場合は警告ログを出したうえで出力から省く(仕様通りの動作でエラーではない)。
- 一部のテーブル(厩舎コメント/スピード指数/持続時間/クッション値など)はレースの性質上
  netKeiba側に元々データが存在しない場合があり、その際は空の該当列で出力される(パース失敗ではない)。
