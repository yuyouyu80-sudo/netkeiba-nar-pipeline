# -*- coding: utf-8 -*-
"""発走前の1レースだけ、人気・オッズ・馬体重を再取得し、既存のpredictions CSVへ
即時反映する軽量スクリプト。

refresh_bias.py --race-id と同じbias.html 1ページ取得(数秒)で済むため、1日分
まとめて取得する fetch_newspaper.py / refresh_bias.py --date より大幅に速い。
predict_pattern29.py 等のフル再実行(1日分の全レースを再スコアリング)とは違い、
スコア(_score)・順位(pred_rank)には一切触れない。単勝人気・オッズはスコアリングに
使っていないため(循環参照を避けるため元々分析軸から除外されている)、newspaper CSVの
最新値を該当predictions CSVの表示列(bias_ninki/bias_win_odds/bias_horse_weight)へ
コピーするだけで表示が最新化できる。

対象predictions CSVはrace_idを含むものを自動検出する(通常戦/新馬戦/未勝利戦、
top5版・全頭版の両方、JRA(セッションscratchpad)・NAR(data/nar_pipeline)の両方)。

最後にcircuitに応じてbuild_artifact.py(JRA)/build_artifact_nar.py(NAR)を実行し、
レポートHTMLを再構築する(Artifact公開は対話的セッションからしか呼べないため対象外。
実行後、Claude Codeセッション側で手動再公開すること)。

使い方:
    python scripts/refresh_race_display.py --race-id 202601010405
    python scripts/refresh_race_display.py --race-id 202654080201 --no-rebuild
"""
import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from dotenv import load_dotenv

from config.settings import LOG_DIR
from refresh_bias import refresh_one_race  # noqa: E402
from src.netkeiba_pipeline.auth.session import login
from src.netkeiba_pipeline.discovery.tracks import is_nar_race
from src.netkeiba_pipeline.storage.paths import newspaper_csv_path
from src.netkeiba_pipeline.utils.logging_conf import configure_logging

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# JRAレポート生成パイプライン(build_artifact.py等)はセッション固有scratchpadにしか
# 存在しない(predict_pattern29.pyのOUT_DIRと同じ場所に揃える)。
JRA_SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)
NAR_OUT_DIR = PROJECT_ROOT / "data" / "nar_pipeline"

BIAS_COLUMNS = ["bias_win_odds", "bias_ninki", "bias_horse_weight"]


def update_predictions_csv(path: Path, race_id: str, bias_by_umaban: pd.DataFrame) -> bool:
    df = pd.read_csv(path, dtype=str, encoding="utf-8")
    if "race_id" not in df.columns or "umaban" not in df.columns:
        return False
    mask = df["race_id"].astype(str) == race_id
    if not mask.any():
        return False
    umaban = df.loc[mask, "umaban"].astype(str)
    for col in BIAS_COLUMNS:
        if col not in df.columns or col not in bias_by_umaban.columns:
            continue
        new_vals = umaban.map(bias_by_umaban[col])
        df.loc[mask, col] = new_vals.where(new_vals.notna(), df.loc[mask, col])
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="1レースだけ最新の人気・オッズ・馬体重を再取得し、既存predictions CSVへ即時反映する。"
    )
    parser.add_argument("--race-id", required=True, help="対象race_id(12桁)")
    parser.add_argument("--no-rebuild", action="store_true", help="build_artifact(_nar).pyを実行しない")
    args = parser.parse_args()

    race_id = args.race_id
    nar = is_nar_race(race_id)

    load_dotenv()
    email = os.environ.get("NETKEIBA_EMAIL")
    password = os.environ.get("NETKEIBA_PASSWORD")
    if not email or not password:
        raise SystemExit(
            "NETKEIBA_EMAIL / NETKEIBA_PASSWORD not set. Copy .env.example to .env "
            "and fill them in yourself (never paste real credentials into chat)."
        )

    configure_logging(LOG_DIR / f"refresh_race_display_{race_id}.log")
    logger = logging.getLogger("refresh_race_display")

    session = login(email, password)
    updated, total = refresh_one_race(session, race_id, logger)
    if total == 0:
        raise SystemExit(f"race_id={race_id}: newspaper CSVが存在しません(先にfetch_newspaper.pyが必要です)")
    print(f"{race_id}: bias.html再取得 {updated}/{total}頭 更新")

    newspaper_df = pd.read_csv(newspaper_csv_path(race_id), dtype=str, encoding="utf-8")
    newspaper_df["umaban"] = newspaper_df["umaban"].astype(str)
    bias_by_umaban = newspaper_df.drop_duplicates("umaban").set_index("umaban")

    search_dir = NAR_OUT_DIR if nar else JRA_SCRATCHPAD
    touched = []
    for path in sorted(search_dir.glob("predictions*.csv")):
        if update_predictions_csv(path, race_id, bias_by_umaban):
            touched.append(path.name)

    if not touched:
        print(
            f"race_id={race_id}: 該当するpredictions CSVが見つかりませんでした"
            "(先にpredict_pattern29.py等で予想を生成してください)"
        )
    else:
        print(f"更新したpredictions CSV ({len(touched)}): {touched}")

    if not args.no_rebuild:
        if nar:
            subprocess.run(
                [sys.executable, str(PROJECT_ROOT / "scripts" / "build_artifact_nar.py")],
                cwd=PROJECT_ROOT, check=True,
            )
        else:
            subprocess.run([sys.executable, "build_artifact.py"], cwd=JRA_SCRATCHPAD, check=True)
        print("レポートを再構築しました。Artifact再公開はClaude Codeセッション側で行ってください。")


if __name__ == "__main__":
    main()
