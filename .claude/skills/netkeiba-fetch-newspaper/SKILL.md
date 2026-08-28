---
name: netkeiba-fetch-newspaper
description: 指定した開催日(YYYYMMDD)または単一race_idについて、馬柱(新聞)ベースの1頭1行ワイドCSV(厩舎コメント/調教タイム/血統ビーム/コース分析/コースデータ/スピード指数/持続時間/data.html各種内訳など)を取得しdata/newspaper/{race_id}.csvに保存する。取得後、同じ日付・circuitで3・4コーナー位置(AI展開)と予想印(本紙・CP・その他)も自動的に続けて取得し、同じCSVへ列追加する。ユーザーが日付を指定して「新聞データ」「馬柱データ」の取得・更新を依頼した際に使う。完了時はブザー音で知らせる。
---

# netkeiba-fetch-newspaper

このワークスペース(netKeibaデータ収集・蓄積基盤プロジェクト)で、`scripts/fetch_newspaper.py` を
使って開催日1日分(または単一race_id)の馬柱(新聞)データを取得し、続けて
`scripts/fetch_corner_position.py`(3・4コーナー位置/AI展開)・`scripts/fetch_marks.py`
(予想印: 本紙・CP・その他)を同じ対象に対して自動的に実行するスキル(2026-08-28〜、日次収集の
標準ワークフローとして3本セットで実行する)。

## 前提条件

- `.env` に `NETKEIBA_EMAIL` / `NETKEIBA_PASSWORD` が設定済みであること。未設定の場合はユーザー
  自身に `.env.example` をコピーして直接編集してもらう(**ID/パスワードをチャットに貼らせない・
  こちらも入力しない**)。
- `pip install -r requirements.txt` 済み、かつ **`python -m playwright install chromium`を
  一度だけ実行済み**であること(`fetch_marks.py`がPlaywrightを使うため)。
- 自動ログイン+スクレイピングであり、netKeibaの利用規約に抵触しうる行為。初回実行時はユーザーの
  明示的な了承を得ること(既に了承済みなら再確認は不要)。

## 手順

対象は開催日1日分(`--date {YYYYMMDD}`)または単一race_id(`--race-id {race_id}`)、
circuit(`--circuit jra`既定 / `--circuit nar`)は3コマンドで必ず揃える。

1. **馬柱データの取得**(1レースあたり約1分かかるため、日付指定は必ずバックグラウンド実行にする)
   ```
   python scripts/fetch_newspaper.py --date {YYYYMMDD} [--circuit nar]
   ```
   または単一race_id:
   ```
   python scripts/fetch_newspaper.py --race-id {race_id}
   ```
   出力: `data/newspaper/{race_id}.csv`(JRA)または`data/newspaper/nar/{race_id}.csv`(NAR)
   (1頭1行、newspaper/shutuba_past/speed_index/holding_time/bias/course_analysis(cid0-3)/
   coursedata(cid1,4)/surf_summary既定+6種のkey1+key2組合せ/concerned/data.html内訳
   (distance/course/condition/others/cushion/baba_water)を全てumabanまたはhorse_idで
   結合したワイドCSV)。

2. **3・4コーナー位置(AI展開)の取得**(手順1が完了してから、同じ`--date`/`--circuit`で実行。
   1で対象になったrace_idの`data/newspaper/`CSVへ列追加するだけの軽量版で、1レースあたり数秒)
   ```
   python scripts/fetch_corner_position.py --date {YYYYMMDD} [--circuit nar]
   ```
   出力列: `corner4_rank`/`corner4_gap_pct`/`corner4_gap_lengths`/`corner4_speedup`/
   `corner3_rank`/`corner3_gap_pct`/`corner3_gap_lengths`。「AI展開データが無い」でのskipは
   正常系(全レースにこの機能があるわけではない)。詳細は
   `.claude/skills/netkeiba-fetch-corner-position/SKILL.md` を参照。

3. **予想印(本紙・CP・その他)の取得**(手順1が完了してから、同じ`--date`/`--circuit`で実行。
   Playwrightのブラウザ起動を含むため1レースあたり数秒〜十数秒)
   ```
   python scripts/fetch_marks.py --date {YYYYMMDD} [--circuit nar]
   ```
   出力列: `mark_raw_{専門家名}`(専門家ごとの生データ)、`mark_honshi`/`mark_cp`/`mark_other`
   (集計済み3列)。NARは「本紙」という専門家が存在しないレースが多く、`mark_honshi`が空になるのは
   正常。詳細は `.claude/skills/netkeiba-fetch-marks/SKILL.md` を参照。

4. **結果確認**(各ステップごとに)
   - 手順1は最後に `{成功数}/{全体数} races written for {YYYYMMDD}` のサマリ行と、
     失敗があれば `Failed (N): race_id1, race_id2, ...` が出力される。
   - 手順2・3はそれぞれ `refreshed corner3/corner4 position: ...` / `refreshed marks: ...`
     のサマリ行が出力される。
   - Windowsコンソールは日本語がcp932で文字化けするため、ログはBashの `tail`/`grep` で確認するか
     Readツールで確認する(printで直接目視確認しない)。

5. **失敗レースの再取得**
   - `fetch_newspaper.py` にはmanifestベースのスキップ機構が無いため、失敗したrace_idだけを
     `--race-id` で個別に再実行する(1レースの出力は独立したCSV全体の書き込みなので、再実行しても
     安全)。典型的な失敗要因は一時的なネットワークエラーで、同じレースを再実行すればほぼ解消する。
   - 手順2・3で「既存newspaper CSVが無い」でスキップされたrace_idは、手順1がその
     race_idで失敗している証拠なので、まず手順1を再実行してから手順2・3を再実行する。
   - 失敗が0件になるまで(または同じ失敗が繰り返される場合は原因を調査した上で)再実行する。

6. **完了報告**
   - 対象日の総レース数・成功数・(あれば)再取得した失敗レースの内訳を、手順1〜3それぞれについて
     簡潔に報告する。
   - **3ステップすべてが完全に完了したら、ユーザーに明示的に頼まれていなくても以下の
     PowerShellコマンドでブザー音を鳴らす**(このワークフローではユーザーが毎回ブザー通知を
     求めているため、既定動作とする):
     ```
     [console]::beep(880,300); [console]::beep(1046,300); [console]::beep(1318,400)
     ```

## 注意点

- 各レースの取得には`newspaper.html`だけでなく計20以上のページへのリクエストが発生するため、
  手順1は1レースあたり約1分かかる。開催日1日(最大36レース程度)では手順1だけで30〜60分程度を
  見込む。手順2・3はそれぞれ軽量(1レース数秒〜十数秒)。
- `PredictRap_Table`(前半3F/後半3F予想ラップ)はこのアカウントの会員レベルでは閲覧不可のページが
  あり、その場合は警告ログを出したうえで出力から省く(仕様通りの動作でエラーではない)。
- 一部のテーブル(厩舎コメント/スピード指数/持続時間/クッション値など)はレースの性質上
  netKeiba側に元々データが存在しない場合があり、その際は空の該当列で出力される(パース失敗ではない)。
- 手順2・3は手順1のCSVに列を追加・上書きするだけなので、手順1が終わっていない(または一部の
  race_idが失敗している)状態で実行すると該当race_idはskipされる。3ステップは必ずこの順序で
  実行すること。
