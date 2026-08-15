# -*- coding: utf-8 -*-
"""JRAの検証待ち日付を1コマンドで通しで反映する。

refresh_bias(オッズ再取得) → predict_pattern29(通常戦予想再生成) →
fetch_quick_result(当日速報、1〜5着・失敗は非fatal) →
predict_shinba_pending(新馬戦、pending全日付を再生成) →
predict_mishoubi_pending(未勝利戦、pending全日付を再生成) →
confidence_per_race(確信度再計算) → build_artifact(レポート再構築)、の順に実行する。

このスクリプト自体はセッション固有scratchpad配下(build_artifact.py等と同じ場所)に置く。
refresh_bias.py / predict_pattern29.py はプロジェクト本体(git管理下のscripts/)を直接呼ぶ。

Artifact公開(claude.ai)は対話的セッションからしか呼び出せないためこのスクリプトの対象外。
実行後、生成された prediction_report.html をClaude Codeセッション側でArtifactとして再公開すること。

使い方:
    python jra_daily_reflect.py --date 20260802
    python jra_daily_reflect.py --date 20260802 --skip-shinba --skip-mishoubi
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
# predict_shinba_pending.py等の実体はこのセッション固有のscratchpadに存在する(scripts/ではない)。
SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)


def run(cmd: list[str], cwd: Path, allow_fail: bool = False) -> int:
    print(f"\n$ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JRAの検証待ち日付の反映チェーン(オッズ再取得→予想再生成→確信度再計算"
        "→レポート再構築)を通しで実行する。"
    )
    parser.add_argument("--date", required=True, help="対象日(YYYYMMDD)")
    parser.add_argument("--skip-shinba", action="store_true", help="predict_shinba_pending.pyをスキップする")
    parser.add_argument("--skip-mishoubi", action="store_true", help="predict_mishoubi_pending.pyをスキップする")
    args = parser.parse_args()

    date = args.date
    py = sys.executable

    run([py, "scripts/refresh_bias.py", "--date", date], cwd=PROJECT_ROOT)
    run([py, "scripts/predict_pattern29.py", "--date", date], cwd=PROJECT_ROOT)
    # db.netkeiba.com(run_pilot.py)は翌日反映のため、発走済みレースの1〜5着を当日中に
    # 見たい場合はrace.netkeiba.com/result.htmlから取れる簡易結果を使う(2026-08-15追加)。
    run([py, "scripts/fetch_quick_result.py", "--date", date], cwd=PROJECT_ROOT, allow_fail=True)
    if not args.skip_shinba:
        run([py, "predict_shinba_pending.py", "--date", date], cwd=SCRATCHPAD)
    if not args.skip_mishoubi:
        run([py, "predict_mishoubi_pending.py", "--date", date], cwd=SCRATCHPAD)
    # 2026-08-12、JRA/NAR確信度統一に伴いconfidence_per_race.pyはscripts/jra_model/へ
    # 昇格した(git管理下、predict.pyへのscratchpad依存を解消済み)。
    run([py, "scripts/jra_model/confidence_per_race.py"], cwd=PROJECT_ROOT)
    run([py, "build_artifact.py"], cwd=SCRATCHPAD)

    print("\n完了。レポート: prediction_report.html (scratchpad)")
    print("Artifact再公開はこのスクリプトの対象外(Claude Codeセッション側で手動再公開すること)。")


if __name__ == "__main__":
    main()
