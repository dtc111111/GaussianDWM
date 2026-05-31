# GaussianDWM CVPR Release Package

This directory is the CVPR-release code package for GaussianDWM. It covers only
the conference-closed QA and world generation paths.

## Included Scope

- QA train, inference, and evaluation.
- World generation train, inference, and ordinary image/depth evaluation.
- DiST-style export layout for FID/FVD calculation.
- Identity and CLIP-similarity Gaussian fine selection.
- Processed annotation contracts and synthetic dummy examples.

## Excluded Scope

- Trajectory train, inference, and evaluation.
- VGGDrive journal work.
- Gaussian-with-ViT-feature experiments.
- Attention-based fine selection.
- Raw NuScenes, LangSplat, or DiST preprocessing.
- Historical initialization from Qwen base plus private component weights.

These are documented non-goals for this release package, not hidden runtime
options.

## Weights

The default model id is
[`dtc111/GaussianDWM`](https://huggingface.co/dtc111/GaussianDWM), loaded from
the Hugging Face repo root. The default revision is `main`.

The loader expects root-level release files such as `config.json` and
`model.safetensors`. It does not assume task subfolders or separate QA/world
repositories.

## Data

This package expects processed QA and world annotation files. It does not
preprocess raw NuScenes, LangSplat, or DiST assets.

`examples/dummy_data/` is synthetic data for contract tests and smoke checks
only. It is not paper-quality data and should not be used to reproduce paper
metrics.

Gaussian payloads are normalized at load time. Raw payload fields are packed as
`xyz(3)+scaling(3)+rotation(4)+opacity(1)+language_feature(3)`. If
`gauss.pose_dir` is set in `data.yaml`, `xyz` is transformed by the configured
pose row for `scene_idx/frame_idx`; `scaling` is clamped to `[-7.5, 7.5]`, and
`opacity` logits are passed through sigmoid. The conference corrected-pose
`normalizer_version: v2` setup should point `gauss.pose_dir` at the matching
`world2ego` pose directory.

See the repository root README for public train/infer entrypoints. The processed
data contract is represented by the config files and synthetic examples in this
package.

## Selector Contracts

QA uses CLIP-similarity fine selection:

```yaml
coarse_method: voxel_topk
coarse_k: 4096
fine_method: similarity
fine_k:
  qa: 512
normalizer_version: v2
```

World generation uses identity fine selection:

```yaml
coarse_method: voxel_topk
coarse_k: 512
fine_method: identity
fine_k:
  world: 512
normalizer_version: v2
```

## DiST FID/FVD

The repository exports a DiST-compatible layout for world-generation FID/FVD
calculation. Metric calculation itself uses external tools from
[`royalmelon0505/dist4d`](https://github.com/royalmelon0505/dist4d).

This package does not vendor DiST metric code.
