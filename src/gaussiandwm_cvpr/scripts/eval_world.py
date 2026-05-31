from __future__ import annotations

import argparse
from pathlib import Path

from gaussiandwm_cvpr.eval.world_metrics import evaluate_world_from_config_dir


DEFAULT_OUTPUT_DIR = "outputs/gaussiandwm_cvpr/eval"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "world"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate public CVPR world-generation predictions.")
    parser.add_argument("--config_dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--predictions_path", required=True)
    parser.add_argument("--output_dir", default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    metrics_path = evaluate_world_from_config_dir(
        config_dir=args.config_dir,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
    )
    print(metrics_path)


if __name__ == "__main__":
    main()
