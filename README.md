# netkeiba データ収集・蓄積基盤

netKeiba会員サイトからレース結果・馬/血統・騎手/調教師・オッズ情報を収集し、CSVとして
蓄積するためのパイロットプロジェクト。詳細な設計・方針は次を参照:

- 実装プラン: `C:\Users\yuyou\.claude\plans\lovely-honking-adleman.md`

## セットアップ

```
pip install -r requirements.txt
cp .env.example .env   # NETKEIBA_EMAIL / NETKEIBA_PASSWORD を自分で入力(チャットには貼らない)
```

## 注意事項

- 会員サイトへの自動ログイン・スクレイピングは netKeiba の利用規約に抵触する可能性があり、
  アカウント停止のリスクがある。自己責任・個人利用目的でのみ実行すること。
- `.env` は絶対にコミットしない(`.gitignore` 済み)。

## 現状のスコープ(パイロット)

まずはレース結果のみを対象に、指定した1日分のデータ取得を検証する。馬/騎手/調教師/オッズは
未実装(理由はプランを参照)。

```
python scripts/run_pilot.py --date YYYYMMDD
```
