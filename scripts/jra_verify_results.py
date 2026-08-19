# -*- coding: utf-8 -*-
"""JRAのレース結果検証チェーンを1コマンドで通しで実行する。

run_pilot.py(確定結果・払戻取得) → box_return.py(通常戦5頭BOX集計) →
box_return_box4.py(4頭BOX集計) → box_return_shinba.py(新馬戦集計) →
box_return_mishoubi.py(未勝利戦集計) → box_return_by_date.py(日付別5頭BOX集計) →
confidence_sweep_v2.py(5頭BOX確信度タブ) → confidence_sweep_box4.py(4頭BOX確信度タブ) →
confidence_sweep_box3.py(3頭BOX集計+確信度タブ) → confidence_sweep_shinba.py(新馬戦確信度) →
confidence_per_race.py(レースカード確信度バッジ) → build_artifact.py(レポート再構築)、
の順に実行する。

run_pilot.pyだけがdb.netkeiba.com側の反映待ちで失敗しうる(結果がまだ掲載されていない場合、
success=0で正常終了する仕様のため非fatal)。他のステップは全て検証済み日付を自動検出する
既存スクリプトで、日付引数は不要。

Claude CodeのArtifact公開機能(claude.ai)は対話的セッションからしか呼び出せないため
このスクリプトの対象外。実行後、生成された prediction_report.html をClaude Codeセッション側で
Artifactとして再公開すること。

使い方:
    python jra_verify_results.py --date 20260801
    python jra_verify_results.py --date 20260801 --skip-fetch
"""
import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(r"c:\Users\yuyou\Desktop\新しい作業場所")
# box_return.py等の実体はこのセッション固有のscratchpadに存在する(scripts/ではない)。
SCRATCHPAD = Path(
    r"C:\Users\yuyou\AppData\Local\Temp\claude\c--Users-yuyou-Desktop--------"
    r"\394156ad-fb7a-45bf-94f3-cbe5b6a82b5e\scratchpad"
)

VERIFY_STEPS = [
    "box_return.py",
    "box_return_box4.py",
    "box_return_shinba.py",
    "box_return_mishoubi.py",
    "box_return_by_date.py",
    "confidence_sweep_v2.py",
    "confidence_sweep_box4.py",
    "confidence_sweep_box3.py",
    "confidence_sweep_shinba.py",
    "confidence_per_race.py",
    "build_artifact.py",
]


def run(cmd: list[str], cwd: Path, allow_fail: bool = False) -> int:
    print(f"\n$ (cwd={cwd}) {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if result.returncode != 0 and not allow_fail:
        raise SystemExit(f"failed (exit {result.returncode}): {' '.join(cmd)}")
    return result.returncode


def main() -> None:
    parser = argparse.ArgumentParser(
        description="JRAのレース結果検証チェーン(結果取得→回収率集計→確信度再計算"
        "→レポート再構築)を通しで実行する。"
    )
    parser.add_argument("--date", required=True, help="結果取得の対象日(YYYYMMDD)")
    parser.add_argument("--skip-fetch", action="store_true", help="run_pilot.pyをスキップする")
    args = parser.parse_args()

    py = sys.executable

    if not args.skip_fetch:
        # 結果がまだnetkeiba側に掲載されていない(反映待ち)のは正常系のため非fatal。
        run([py, "scripts/run_pilot.py", "--date", args.date], cwd=PROJECT_ROOT, allow_fail=True)

    for step in VERIFY_STEPS:
        if step == "confidence_per_race.py":
            # 2026-08-12、JRA/NAR確信度統一に伴いscripts/jra_model/へ昇格した
            # (git管理下、predict.pyへのscratchpad依存を解消済み)。
            run([py, "scripts/jra_model/confidence_per_race.py"], cwd=PROJECT_ROOT)
        else:
            run([py, step], cwd=SCRATCHPAD)

    print("\n完了。レポート: prediction_report.html (scratchpad)")
    print("Artifact再公開はこのスクリプトの対象外(Claude Codeセッション側で手動再公開すること)。")


if __name__ == "__main__":
    main()
