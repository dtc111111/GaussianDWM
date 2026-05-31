from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest
import torch

from gaussiandwm_cvpr.data.clip_text_features import load_clip_text_feature
from gaussiandwm_cvpr.data.collator import GaussianDWMCollator
from gaussiandwm_cvpr.data.dataset import GaussianDWMDataset
from gaussiandwm_cvpr.data.gauss_cache import GaussFeatureCache
from gaussiandwm_cvpr.data.gauss_normalizer import GaussNormalizer
from gaussiandwm_cvpr.data.records import ProcessedRecord, load_records
from gaussiandwm_cvpr.data.taxonomy import load_taxonomy
from gaussiandwm_cvpr.utils import load_json_or_jsonl, load_yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PACKAGE_ROOT / "configs"


class DummyTokenizer:
    pad_token_id = 0

    def __init__(self) -> None:
        self._token_to_id = {
            "<|im_start|>": 1,
            "<|im_end|>": 2,
            "system": 3,
            "user": 4,
            "assistant": 5,
            "\n": 6,
            "<|gaussian_start|>": 7,
            "<|gaussian_pad|>": 8,
            "<|gaussian_end|>": 9,
        }
        self._id_to_token = {value: key for key, value in self._token_to_id.items()}
        self._next_id = 10

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._token_id(token)

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self._id_to_token[int(token_id)]

    def encode_prompt(self, prompt: str) -> list[int]:
        ids: list[int] = []
        index = 0
        special_tokens = [
            "<|im_start|>",
            "<|im_end|>",
            "<|gaussian_start|>",
            "<|gaussian_pad|>",
            "<|gaussian_end|>",
        ]
        while index < len(prompt):
            matched = None
            for token in special_tokens:
                if prompt.startswith(token, index):
                    matched = token
                    break
            if matched is not None:
                ids.append(self._token_id(matched))
                index += len(matched)
                continue
            char = prompt[index]
            if char.isspace():
                index += 1
                continue
            end = index + 1
            while end < len(prompt) and not prompt[end].isspace() and not any(
                prompt.startswith(token, end) for token in special_tokens
            ):
                end += 1
            ids.append(self._token_id(prompt[index:end]))
            index = end
        return ids

    def _token_id(self, token: str) -> int:
        if token not in self._token_to_id:
            token_id = self._next_id
            self._next_id += 1
            self._token_to_id[token] = token_id
            self._id_to_token[token_id] = token
        return self._token_to_id[token]


class DummyProcessor:
    def __init__(self) -> None:
        self.tokenizer = DummyTokenizer()

    def apply_chat_template(
        self,
        messages: list[dict[str, object]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        parts: list[str] = []
        for message in messages:
            parts.append(f"<|im_start|>{message['role']}\n")
            content = message["content"]
            assert isinstance(content, list)
            for item in content:
                assert isinstance(item, dict)
                if item["type"] == "text":
                    parts.append(str(item["text"]))
                elif item["type"] == "gauss":
                    parts.append("<|gaussian_start|><|gaussian_pad|><|gaussian_end|>")
                elif item["type"] == "image":
                    parts.append("<image>")
            parts.append("<|im_end|>\n")
        if add_generation_prompt:
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def __call__(
        self,
        *,
        text: list[str],
        images: list[object] | None,
        return_tensors: str,
        padding: bool,
    ) -> dict[str, torch.Tensor]:
        assert return_tensors == "pt"
        assert padding is False
        input_ids = torch.tensor([self.tokenizer.encode_prompt(text[0])], dtype=torch.long)
        encoded = {
            "input_ids": input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }
        if images:
            encoded["pixel_values"] = torch.zeros((len(images), 3, 448, 768), dtype=torch.float32)
            encoded["image_grid_thw"] = torch.tensor([[1, 28, 48]] * len(images), dtype=torch.long)
        return encoded


def _gauss_cache(tmp_path: Path, *, fine_k: int) -> GaussFeatureCache:
    return GaussFeatureCache(
        cache_root=tmp_path / "cache",
        coarse_method="voxel_topk",
        coarse_k=fine_k,
        fine_k_by_task={"qa": fine_k, "world": fine_k},
        normalizer=GaussNormalizer(),
    )


def _world_dataset_for_records(
    records: list[ProcessedRecord],
    tmp_path: Path,
) -> GaussianDWMDataset:
    data_config = load_yaml(CONFIG_ROOT / "world" / "data.yaml")
    gaussian_config = load_yaml(CONFIG_ROOT / "world" / "gaussian_token.yaml")
    return GaussianDWMDataset(
        records=records,
        processor=DummyProcessor(),
        gauss_cache=_gauss_cache(tmp_path, fine_k=512),
        gaussian_config=gaussian_config,
        world_config=data_config["world"],
        mode="train",
    )


def _world_val_records() -> list[ProcessedRecord]:
    data_config = load_yaml(CONFIG_ROOT / "world" / "data.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "world" / "budgets.yaml")
    return load_records(
        data_config,
        taxonomy=taxonomy,
        split="val",
        dataset_names=["world_val"],
        package_root=PACKAGE_ROOT,
    )


def test_processed_record_public_fields_match_plan() -> None:
    assert [field.name for field in fields(ProcessedRecord)] == [
        "sample_uid",
        "dataset_name",
        "task",
        "scene_idx",
        "frame_idx",
        "views",
        "image_paths",
        "gauss_paths",
        "source_index",
        "conversations",
        "qa_group",
        "qa_subtask",
        "clip_text_embed_path",
        "clip_text_embed_row",
        "ref_image_path",
        "pseudo_video_paths",
        "pseudo_depth_paths",
        "target_latent_paths",
        "target_latent_pack_path",
        "target_latent_indices",
        "gt_rgb_frame_paths",
        "gt_depth_npz_paths",
        "pseudo_source_name",
        "pseudo_source_kind",
        "fps",
        "motion_bucket_id",
    ]


def test_qa_val_record_contract() -> None:
    data_config = load_yaml(CONFIG_ROOT / "qa" / "data.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "qa" / "budgets.yaml")

    records = load_records(
        data_config,
        taxonomy=taxonomy,
        split="val",
        dataset_names=["qa_val"],
        package_root=PACKAGE_ROOT,
    )

    assert len(records) == 1
    record = records[0]
    assert record.task == "qa"
    assert record.dataset_name == "qa_val"
    assert record.sample_uid == "dummy_qa_000001"
    assert record.scene_idx == 0
    assert record.frame_idx == 0
    assert record.views == ["CAM_FRONT"]
    assert len(record.image_paths) == 1
    assert str(record.image_paths[0]).endswith("images/qa_image.png")
    assert len(record.gauss_paths) == 1
    assert str(record.gauss_paths[0]).endswith(
        "gauss/output-full-6v/0_CAM_FRONT/langsplat_3/per_frame/00000.pth"
    )
    assert record.source_index == 0
    assert record.conversations[-1]["role"] == "assistant"
    assert record.qa_group == "NuInteract_RD&P"
    assert record.qa_subtask == "prediction"
    assert str(record.clip_text_embed_path).endswith("clip_text/features_000000.npy")
    assert record.clip_text_embed_row == 0


def test_dataset_spec_requires_major_task() -> None:
    data_config = load_yaml(CONFIG_ROOT / "qa" / "data.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "qa" / "budgets.yaml")
    del data_config["datasets"]["qa_val"]["major_task"]

    with pytest.raises(TypeError):
        load_records(
            data_config,
            taxonomy=taxonomy,
            split="val",
            dataset_names=["qa_val"],
            package_root=PACKAGE_ROOT,
        )


def test_annotation_record_requires_explicit_major_task(tmp_path: Path) -> None:
    data_config = load_yaml(CONFIG_ROOT / "qa" / "data.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "qa" / "budgets.yaml")
    record = load_json_or_jsonl(PACKAGE_ROOT / "examples/dummy_data/qa_val.jsonl")[0]
    del record["major_task"]
    annotation_path = tmp_path / "qa_missing_major_task.jsonl"
    annotation_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    data_config["datasets"]["qa_val"]["annotation_path"] = str(annotation_path)

    with pytest.raises(TypeError):
        load_records(
            data_config,
            taxonomy=taxonomy,
            split="val",
            dataset_names=["qa_val"],
            package_root=PACKAGE_ROOT,
        )


def test_annotation_source_index_is_generated_from_row_order(tmp_path: Path) -> None:
    data_config = load_yaml(CONFIG_ROOT / "qa" / "data.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "qa" / "budgets.yaml")
    record = load_json_or_jsonl(PACKAGE_ROOT / "examples/dummy_data/qa_val.jsonl")[0]
    record.pop("source_index", None)
    annotation_path = tmp_path / "qa_without_source_index.jsonl"
    annotation_path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    data_config["datasets"]["qa_val"]["annotation_path"] = str(annotation_path)

    records = load_records(
        data_config,
        taxonomy=taxonomy,
        split="val",
        dataset_names=["qa_val"],
        package_root=PACKAGE_ROOT,
    )

    assert records[0].source_index == 0


def test_repo_data_example_matches_public_cvpr_scope() -> None:
    repo_root = PACKAGE_ROOT.parents[1]
    data_example_root = repo_root / "data_example"
    if not data_example_root.exists():
        pytest.skip("repo-level data_example is not present")

    qa_records = load_records(
        load_yaml(data_example_root / "configs" / "qa_nuinteract" / "data.yaml"),
        taxonomy=load_taxonomy(CONFIG_ROOT / "qa" / "budgets.yaml"),
        split="train",
        package_root=repo_root,
    )
    world_records = load_records(
        load_yaml(data_example_root / "configs" / "world" / "data.yaml"),
        taxonomy=load_taxonomy(CONFIG_ROOT / "world" / "budgets.yaml"),
        split="train",
        package_root=repo_root,
    )

    assert len(qa_records) == 10
    assert [record.source_index for record in qa_records] == list(range(10))
    assert {record.task for record in qa_records} == {"qa"}
    assert len(world_records) == 1
    assert world_records[0].task == "world"
    assert world_records[0].source_index == 0


def test_clip_text_feature_returns_copy_not_cached_view() -> None:
    feature_path = PACKAGE_ROOT / "examples/dummy_data/clip_text/features_000000.npy"
    first = load_clip_text_feature(feature_path, 0)
    original_value = first[0].item()
    first[0] = original_value + 100.0

    second = load_clip_text_feature(feature_path, 0)

    assert second[0].item() == original_value


def test_world_val_record_contract() -> None:
    data_config = load_yaml(CONFIG_ROOT / "world" / "data.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "world" / "budgets.yaml")

    records = load_records(
        data_config,
        taxonomy=taxonomy,
        split="val",
        dataset_names=["world_val"],
        package_root=PACKAGE_ROOT,
    )

    assert len(records) == 1
    record = records[0]
    assert record.task == "world"
    assert record.dataset_name == "world_val"
    assert record.sample_uid == "dummy_world_000001"
    assert record.scene_idx == 0
    assert record.frame_idx == 0
    assert record.source_index == 0
    assert record.conversations == [{"role": "user", "content": "Generate the next six frames."}]
    assert str(record.ref_image_path).endswith("images/world_ref.png")
    assert len(record.image_paths) == 1
    assert len(record.gauss_paths) == 1
    assert len(record.pseudo_video_paths) == 6
    assert len(record.pseudo_depth_paths) == 6
    assert len(record.gt_rgb_frame_paths) == 6
    assert len(record.gt_depth_npz_paths) == 6
    assert str(record.pseudo_video_paths[0]).endswith("images/world_pseudo_000.png")
    assert str(record.gt_rgb_frame_paths[0]).endswith("images/world_gt_000.png")
    assert all(str(path).endswith(".png") for path in record.pseudo_video_paths)
    assert all(str(path).endswith(".npz") for path in record.pseudo_depth_paths)
    assert str(record.target_latent_pack_path).endswith("latents/world_pack.dat")
    assert record.target_latent_indices == [0, 1, 2, 3, 4, 5]
    assert record.pseudo_source_name == "dummy"
    assert record.pseudo_source_kind == "contract"
    assert record.fps == 5
    assert record.motion_bucket_id == 127


def test_qa_dataset_and_collator_shape_contracts(tmp_path: Path) -> None:
    fine_k = 512
    data_config = load_yaml(CONFIG_ROOT / "qa" / "data.yaml")
    gaussian_config = load_yaml(CONFIG_ROOT / "qa" / "gaussian_token.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "qa" / "budgets.yaml")
    records = load_records(
        data_config,
        taxonomy=taxonomy,
        split="val",
        dataset_names=["qa_val"],
        package_root=PACKAGE_ROOT,
    )
    processor = DummyProcessor()
    dataset = GaussianDWMDataset(
        records=records,
        processor=processor,
        gauss_cache=_gauss_cache(tmp_path, fine_k=fine_k),
        gaussian_config=gaussian_config,
        mode="train",
    )

    sample = dataset[0]
    assert sample.task == "qa"
    assert sample.qwen_inputs["coarse_gauss_values"].shape == (512, 14)
    assert sample.qwen_inputs["clip_text_embed"].shape == (512,)

    collator = GaussianDWMCollator(processor.tokenizer, mode="train")
    batch = collator([sample])
    assert batch["qwen_inputs"]["coarse_gauss_values"].shape == (1, 512, 14)
    assert batch["qwen_inputs"]["clip_text_embed"].shape == (1, 512)


def test_world_dataset_and_collator_shape_contracts(tmp_path: Path) -> None:
    fine_k = 512
    data_config = load_yaml(CONFIG_ROOT / "world" / "data.yaml")
    gaussian_config = load_yaml(CONFIG_ROOT / "world" / "gaussian_token.yaml")
    taxonomy = load_taxonomy(CONFIG_ROOT / "world" / "budgets.yaml")
    records = load_records(
        data_config,
        taxonomy=taxonomy,
        split="val",
        dataset_names=["world_val"],
        package_root=PACKAGE_ROOT,
    )
    processor = DummyProcessor()
    dataset = GaussianDWMDataset(
        records=records,
        processor=processor,
        gauss_cache=_gauss_cache(tmp_path, fine_k=fine_k),
        gaussian_config=gaussian_config,
        world_config=data_config["world"],
        mode="train",
    )

    sample = dataset[0]
    assert sample.task == "world"
    assert sample.task_inputs["ref_pixel_values"].shape == (3, 448, 768)
    assert sample.task_inputs["pseudo_pixel_values"].shape == (6, 3, 448, 768)
    assert sample.task_inputs["pseudo_depth_values"].shape == (6, 3, 448, 768)
    assert sample.targets["target_latents"]["rgb_latents"].shape == (6, 4, 56, 96)

    collator = GaussianDWMCollator(processor.tokenizer, mode="train")
    batch = collator([sample])
    assert batch["pseudo_pixel_values"].shape == (1, 6, 3, 448, 768)
    assert batch["target_latents"]["rgb_latents"].shape == (1, 6, 4, 56, 96)


def test_world_frame_target_latents_reject_bad_shape(tmp_path: Path) -> None:
    records = _world_val_records()
    latent_paths = []
    for index in range(6):
        latent_path = tmp_path / f"frame_latent_{index}.pt"
        torch.save(
            {
                "rgb_latent": torch.zeros((5, 56, 96), dtype=torch.float32),
                "depth_latent": torch.zeros((5, 56, 96), dtype=torch.float32),
            },
            latent_path,
        )
        latent_paths.append(str(latent_path))
    bad_record = replace(
        records[0],
        target_latent_paths=latent_paths,
        target_latent_pack_path=None,
        target_latent_indices=None,
    )
    dataset = _world_dataset_for_records([bad_record], tmp_path)

    with pytest.raises(ValueError, match=r"\[4,56,96\]"):
        dataset[0]


def test_world_packed_target_latents_reject_bad_shape(tmp_path: Path) -> None:
    records = _world_val_records()
    pack_path = tmp_path / "bad_world_pack.dat"
    shape = (6, 2, 5, 56, 96)
    pack = np.memmap(pack_path, mode="w+", dtype=np.float32, shape=shape)
    pack[:] = 0.0
    pack.flush()
    (tmp_path / "meta.json").write_text(
        json.dumps(
            {
                "dtype": "float32",
                "shape": list(shape),
                "streams": ["rgb", "depth"],
                "cache_granularity": "scene_camera_pack",
            }
        ),
        encoding="utf-8",
    )
    bad_record = replace(
        records[0],
        target_latent_pack_path=str(pack_path),
        target_latent_indices=[0, 1, 2, 3, 4, 5],
        target_latent_paths=[],
    )
    dataset = _world_dataset_for_records([bad_record], tmp_path)

    with pytest.raises(ValueError, match=r"\[N,2,4,56,96\]"):
        dataset[0]
