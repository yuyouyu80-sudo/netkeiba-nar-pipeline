# -*- coding: utf-8 -*-
"""試行回数カウンター(2026-09-06、改善計画Phase 4-項目12)。

複数パターンを探索するたびに`log_trial()`を1行呼ぶだけで、
`data/jra_pipeline/trial_counter_log.jsonl`(JSONL、追記のみ)に記録が残る。
選択バイアスの説明責任を果たすための**ガバナンス用メタデータ**であり、多重比較補正の
自動入力には使わない(単純累積nをBonferroniにそのまま入れると、探索するほど未来永劫
採用不可能になる誤ったルールになるため。レビュー指摘、[[project_jra_improvement_review_2026_09_06]])。

**既存の`jra_search_*.py`群への遡及改修はしない**(各スクリプトが自身のdocstringで
「既存コードは無改造で参照するのみ」と宣言している凍結済み検証成果物のため)。
今後の新規スクリプトにのみ適用する。
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LOG_PATH = PROJECT_ROOT / "data" / "jra_pipeline" / "trial_counter_log.jsonl"


def log_trial(script_name: str, n_candidates: int, description: str,
              outcome: str | None = None, log_path: Path = LOG_PATH) -> dict:
    """1件のログをJSONLに追記する。

    Args:
        script_name: 記録元スクリプトのファイル名(例: "jra_odds_movement_signals.py")。
        n_candidates: このスクリプトが試した候補パターン数(重み・閾値の組み合わせ数等)。
        description: 何を探索したかの短い説明(日本語可)。
        outcome: 分かっていれば"採用"/"不採用"/"スクリーニングで見送り"等。未確定ならNone。
    """
    entry = {
        "logged_at": datetime.now(timezone.utc).isoformat(),
        "script_name": script_name,
        "n_candidates": int(n_candidates),
        "description": description,
        "outcome": outcome,
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return entry


def cumulative_n_candidates(log_path: Path = LOG_PATH) -> int:
    """記録済みの試行(候補パターン)数の累積合計を返す(参考値。多重比較補正の
    自動入力には使わないこと、モジュールdocstring参照)。"""
    if not log_path.exists():
        return 0
    total = 0
    with log_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total += json.loads(line).get("n_candidates", 0)
    return total


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--script-name", required=True)
    parser.add_argument("--n-candidates", type=int, required=True)
    parser.add_argument("--description", required=True)
    parser.add_argument("--outcome", default=None)
    args = parser.parse_args()

    entry = log_trial(args.script_name, args.n_candidates, args.description, args.outcome)
    print(f"logged: {entry}")
    print(f"cumulative n_candidates (参考値、補正には未使用): {cumulative_n_candidates()}")


if __name__ == "__main__":
    main()
