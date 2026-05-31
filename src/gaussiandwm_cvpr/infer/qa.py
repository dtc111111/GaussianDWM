from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from transformers import AutoProcessor, GenerationConfig, GenerationMixin

from gaussiandwm_cvpr.data.collator import GaussianDWMCollator
from gaussiandwm_cvpr.data.dataset import GaussianDWMDataset
from gaussiandwm_cvpr.data.gauss_cache import GaussFeatureCache
from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer
from gaussiandwm_cvpr.data.records import load_records
from gaussiandwm_cvpr.data.taxonomy import load_taxonomy
from gaussiandwm_cvpr.models.unified_model import UnifiedGaussianDWM
from gaussiandwm_cvpr.utils import dump_json, load_yaml, resolve_path, timestamp_now


class QAGenerationWrapper(torch.nn.Module, GenerationMixin):
    main_input_name = "input_ids"
    _is_stateful = False

    def __init__(self, model: UnifiedGaussianDWM) -> None:
        super().__init__()
        self.model = model
        self.config = model.backbone.qwen.config
        generation_config = getattr(model.backbone.qwen, "generation_config", None)
        self.generation_config = generation_config or GenerationConfig.from_model_config(self.config)

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    def can_generate(self) -> bool:
        return True

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.Tensor,
        past_key_values: Any = None,
        attention_mask: torch.Tensor | None = None,
        cache_position: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        coarse_gauss_values: torch.Tensor | None = None,
        coarse_gauss_mask: torch.Tensor | None = None,
        clip_text_embed: torch.Tensor | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if cache_position is not None and int(cache_position[0].item()) != 0:
            pixel_values = None
            image_grid_thw = None
            coarse_gauss_values = None
            coarse_gauss_mask = None
            clip_text_embed = None
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "cache_position": cache_position,
            "pixel_values": pixel_values,
            "image_grid_thw": image_grid_thw,
            "coarse_gauss_values": coarse_gauss_values,
            "coarse_gauss_mask": coarse_gauss_mask,
            "clip_text_embed": clip_text_embed,
            "task_type": "qa",
        }

    def forward(self, **kwargs: Any) -> Any:
        return self.model.backbone.forward_for_generation(**kwargs)

    def _reorder_cache(self, past_key_values: Any, beam_idx: torch.Tensor) -> Any:
        qwen_model = self.model.backbone.qwen
        if hasattr(qwen_model, "_reorder_cache"):
            return qwen_model._reorder_cache(past_key_values, beam_idx)
        return past_key_values


def run_qa_inference_from_config_dir(
    config_dir: str | Path,
    run_dir: str | Path | None = None,
    model_id: str | Path | None = None,
    revision: str | None = None,
    data_root: str | Path | None = None,
    annotation_path: str | Path | None = None,
    gauss_cache_root: str | Path | None = None,
) -> Path:
    """Run QA inference and write a public predictions JSON file."""
    config_root = Path(config_dir).resolve()
    if run_dir is None:
        run_root = Path("outputs/gaussiandwm_cvpr").resolve()
    else:
        run_root = Path(run_dir).resolve()

    data_cfg = load_yaml(config_root / "data.yaml")
    gaussian_cfg = load_yaml(config_root / "gaussian_token.yaml")
    infer_cfg = load_yaml(config_root / "infer.yaml")
    model_cfg = load_yaml(config_root / "model.yaml")
    taxonomy = load_taxonomy(config_root / "budgets.yaml")

    model_name = str(model_id if model_id is not None else model_cfg["model_id"])
    model_revision = str(revision if revision is not None else model_cfg.get("revision", "main"))
    split = str(infer_cfg.get("split", "val"))
    dataset_names = [str(name) for name in infer_cfg.get("dataset_names", [])]
    if not dataset_names:
        raise ValueError("infer.yaml must provide at least one dataset name.")

    # Direct CLI/API overrides are applied in the public entry flow so the
    # exact data roots used for a run are visible at the call site.
    if data_root is not None:
        root_text = str(Path(data_root).expanduser())
        for dataset_name in dataset_names:
            dataset_cfg = data_cfg["datasets"][dataset_name]
            dataset_cfg["annotation_root"] = root_text
            dataset_cfg["image_root"] = root_text
            dataset_cfg["gauss_root"] = root_text
        clip_cfg = data_cfg.setdefault("clip_text_feature", {})
        if isinstance(clip_cfg, dict) and clip_cfg.get("root") is not None:
            clip_cfg["root"] = root_text
    if annotation_path is not None:
        for dataset_name in dataset_names:
            data_cfg["datasets"][dataset_name]["annotation_path"] = str(Path(annotation_path).expanduser())
    if gauss_cache_root is not None:
        gaussian_cfg["cache_root"] = str(Path(gauss_cache_root).expanduser())

    package_root = _package_root_for_config(config_root)
    processor = AutoProcessor.from_pretrained(model_name, revision=model_revision)
    records = load_records(
        data_cfg,
        taxonomy=taxonomy,
        split=split,
        dataset_names=dataset_names,
        task_filter=["qa"],
        package_root=package_root,
    )
    gauss_data_cfg = data_cfg.get("gauss", {})
    if not isinstance(gauss_data_cfg, dict):
        raise TypeError("data.yaml must define gauss as a mapping when present.")
    pose_dir = gauss_data_cfg.get("pose_dir")
    gauss_cache = GaussFeatureCache(
        cache_root=gaussian_cfg["cache_root"],
        coarse_method=str(gaussian_cfg["coarse_method"]),
        coarse_k=int(gaussian_cfg["coarse_k"]),
        fine_k_by_task={str(k): int(v) for k, v in gaussian_cfg["fine_k"].items()},
        normalizer_version=str(gaussian_cfg.get("normalizer_version", "v2")),
        normalizer=GaussNormalizer(
            pose_dir=None if not pose_dir else str(resolve_path(str(pose_dir), base=package_root)),
            pose_template=str(gauss_data_cfg.get("pose_template", "{scene_id}.txt")),
        ),
    )
    dataset = GaussianDWMDataset(
        records=records,
        processor=processor,
        gauss_cache=gauss_cache,
        gaussian_config=gaussian_cfg,
        mode="infer",
    )
    dataloader = DataLoader(
        dataset,
        batch_size=int(infer_cfg.get("batch_size", 1)),
        shuffle=False,
        collate_fn=GaussianDWMCollator(processor.tokenizer, mode="infer"),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UnifiedGaussianDWM.from_pretrained(model_name, revision=model_revision)
    model.to(device)
    model.eval()

    wrapper = QAGenerationWrapper(model).to(device)
    predictions: list[dict[str, Any]] = []
    max_new_tokens = int(infer_cfg.get("qa", {}).get("max_new_tokens", 1024))
    for batch in dataloader:
        qwen_inputs = _move_tensors_to_device(batch["qwen_inputs"], device)
        with _generation_autocast_context(device):
            generated = wrapper.generate(
                **qwen_inputs,
                max_new_tokens=max_new_tokens,
                use_cache=False,
            )
        prompt_len = qwen_inputs["input_ids"].shape[1]
        pred_texts = processor.batch_decode(
            generated[:, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        for index, pred_text in enumerate(pred_texts):
            predictions.append(
                {
                    "task_type": "qa",
                    "sample_uid": batch["sample_uids"][index],
                    "source_index": batch["meta"].get("source_index", [None])[index],
                    "dataset_name": batch["dataset_names"][index],
                    "qa_group": batch["meta"].get("qa_group", [None])[index],
                    "qa_subtask": batch["meta"].get("qa_subtask", [None])[index],
                    "pred": pred_text,
                    "gt": batch["meta"].get("gt_text", [None])[index],
                    "meta": {"source": "gaussiandwm_cvpr.qa"},
                }
            )

    infer_id = str(infer_cfg.get("infer_id", "auto"))
    if infer_id == "auto":
        infer_id = timestamp_now()
    output_path = run_root / "infer" / f"qa_predictions_{infer_id}.json"
    dump_json(
        {
            "task_type": "qa",
            "model_id": model_name,
            "revision": model_revision,
            "config_dir": str(config_root),
            "split": split,
            "dataset_names": dataset_names,
            "count": len(predictions),
            "items": predictions,
        },
        output_path,
    )
    return output_path


def _move_tensors_to_device(payload: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in payload.items()}


def _generation_autocast_context(device: torch.device):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _package_root_for_config(config_root: Path) -> Path:
    if config_root.parent.name == "configs":
        return config_root.parent.parent
    return config_root
