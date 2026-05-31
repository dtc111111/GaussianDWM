from __future__ import annotations

import argparse
from pathlib import Path

from gaussiandwm_cvpr.infer.qa import run_qa_inference_from_config_dir


DEFAULT_MODEL_ID = "dtc111/GaussianDWM"
DEFAULT_REVISION = "main"
DEFAULT_RUN_DIR = "outputs/gaussiandwm_cvpr"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "qa"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run public CVPR QA inference.")
    parser.add_argument("--config_dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--run_dir", default=DEFAULT_RUN_DIR)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--annotation_path", default=None)
    parser.add_argument("--gauss_cache_root", default=None)
    args = parser.parse_args()

    predictions_path = run_qa_inference_from_config_dir(
        config_dir=args.config_dir,
        run_dir=args.run_dir,
        model_id=args.model_id,
        revision=args.revision,
        data_root=args.data_root,
        annotation_path=args.annotation_path,
        gauss_cache_root=args.gauss_cache_root,
    )
    print(predictions_path)


if __name__ == "__main__":
    main()
