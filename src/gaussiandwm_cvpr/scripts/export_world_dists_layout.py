from __future__ import annotations

import argparse
from pathlib import Path

from gaussiandwm_cvpr.eval.export_dists_layout import export_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Export world predictions into the DiST FID/FVD layout.")
    parser.add_argument("--annotation_path", required=True)
    parser.add_argument("--predictions_json_paths", nargs="+", required=True)
    parser.add_argument("--output_root", required=True)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--cameras", nargs="+", default=None)
    parser.add_argument("--layout_name", default=None)
    parser.add_argument("--shifts", nargs="+", default=["1", "2", "4"])
    parser.add_argument("--missing_prediction", choices=["error", "skip"], default="error")
    parser.add_argument("--overwrite", choices=["error", "replace"], default="error")
    args = parser.parse_args()

    manifest = export_layout(
        annotation_path=args.annotation_path,
        predictions_json_paths=args.predictions_json_paths,
        output_root=args.output_root,
        camera=args.camera,
        cameras=args.cameras,
        layout_name=args.layout_name,
        shifts={str(shift) for shift in args.shifts},
        missing_prediction=args.missing_prediction,
        overwrite=args.overwrite,
    )
    manifest_path = Path(str(manifest["output_root"])) / "export_manifest.json"
    print(manifest_path)


if __name__ == "__main__":
    main()
