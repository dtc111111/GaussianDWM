from __future__ import annotations

import argparse
from pathlib import Path

from gaussiandwm_cvpr.train.trainer import run_training_from_config_dir


DEFAULT_MODEL_ID = "dtc111/GaussianDWM"
DEFAULT_REVISION = "main"
DEFAULT_OUTPUT_DIR = "outputs/gaussiandwm_cvpr"
DEFAULT_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs" / "world"


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune the public CVPR world-generation chain.")
    parser.add_argument("--config_dir", default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--run_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--data_root", default=None)
    parser.add_argument("--annotation_path", default=None)
    parser.add_argument("--gauss_cache_root", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif args.run_dir is None:
        output_dir = DEFAULT_OUTPUT_DIR
    else:
        output_dir = None
    run_dir = run_training_from_config_dir(
        task="world",
        config_dir=args.config_dir,
        run_dir=args.run_dir,
        output_dir=output_dir,
        model_id=args.model_id,
        revision=args.revision,
        data_root=args.data_root,
        annotation_path=args.annotation_path,
        gauss_cache_root=args.gauss_cache_root,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
