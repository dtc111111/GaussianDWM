from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from gaussiandwm_cvpr.data.clip_text_features import load_clip_text_feature
from gaussiandwm_cvpr.data.records import ProcessedRecord


IGNORE_INDEX = -100
WORLD_SIZE_WH = (768, 448)
DEPTH_MIN = 0.5
DEPTH_MAX = 100.0
TARGET_LATENT_CHW = (4, 56, 96)
GAUSS_TEMPLATE_SENTINEL = "<|gaussian_start|><|gaussian_pad|><|gaussian_end|>"


@dataclass
class GaussianDWMSample:
    sample_uid: str
    dataset_name: str
    task: str
    qwen_inputs: dict[str, Any]
    task_inputs: dict[str, Any]
    targets: dict[str, Any]
    meta: dict[str, Any]


class GaussianDWMDataset(Dataset[GaussianDWMSample]):
    def __init__(
        self,
        *,
        records: list[ProcessedRecord],
        processor: Any,
        gauss_cache: Any,
        gaussian_config: dict[str, Any],
        mode: str,
        world_config: dict[str, Any] | None = None,
    ) -> None:
        if mode not in {"train", "infer"}:
            raise ValueError(f"mode must be 'train' or 'infer', got {mode!r}")
        if not records:
            raise ValueError("records must not be empty")
        self.records = list(records)
        self.processor = processor
        self.gauss_cache = gauss_cache
        self.mode = mode
        self.world_config = dict(world_config or {})

        fine_k = gaussian_config.get("fine_k")
        if not isinstance(fine_k, dict):
            raise TypeError("gaussian_config['fine_k'] must be a mapping")
        self.fine_k_by_task = {str(task): int(value) for task, value in fine_k.items()}
        self.fine_method = str(gaussian_config.get("fine_method", "identity"))

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> GaussianDWMSample:
        record = self.records[index]
        if record.task == "qa":
            return self._build_qa_sample(record)
        if record.task == "world":
            return self._build_world_sample(record)
        raise ValueError(f"Unsupported record task: {record.task!r}")

    def _build_qa_sample(self, record: ProcessedRecord) -> GaussianDWMSample:
        fine_k = self._fine_k(record)
        gt_text = _last_assistant_text(record.conversations)
        targets: dict[str, Any] = {}

        if self.mode == "infer":
            prompt_messages = _build_qwen_messages(
                _drop_last_assistant_turn(record.conversations),
                record.image_paths,
            )
            qwen_inputs = _tokenize_messages(
                self.processor,
                messages=prompt_messages,
                gauss_token_count=fine_k,
                add_generation_prompt=True,
            )
        else:
            prompt_messages = _build_qwen_messages(record.conversations, record.image_paths)
            qwen_inputs = _tokenize_messages(
                self.processor,
                messages=prompt_messages,
                gauss_token_count=fine_k,
                add_generation_prompt=False,
            )
            targets["labels"] = _build_assistant_only_labels(
                self.processor.tokenizer,
                qwen_inputs["input_ids"],
            )

        qwen_inputs["coarse_gauss_values"] = self.gauss_cache.load_for_record(record)
        if self.fine_method == "similarity":
            if record.clip_text_embed_path is None:
                raise ValueError(f"QA record {record.sample_uid} is missing clip_text_embed_path")
            if record.clip_text_embed_row is None:
                raise ValueError(f"QA record {record.sample_uid} is missing clip_text_embed_row")
            qwen_inputs["clip_text_embed"] = load_clip_text_feature(
                record.clip_text_embed_path,
                record.clip_text_embed_row,
            )

        return GaussianDWMSample(
            sample_uid=record.sample_uid,
            dataset_name=record.dataset_name,
            task="qa",
            qwen_inputs=qwen_inputs,
            task_inputs={},
            targets=targets,
            meta={
                "source_index": record.source_index,
                "qa_group": record.qa_group,
                "qa_subtask": record.qa_subtask,
                "gt_text": gt_text,
            },
        )

    def _build_world_sample(self, record: ProcessedRecord) -> GaussianDWMSample:
        fine_k = self._fine_k(record)
        with_valid_mask = self.world_config.get("with_valid_mask", True)
        if not isinstance(with_valid_mask, bool):
            raise TypeError(f"world_config['with_valid_mask'] must be bool, got {type(with_valid_mask)!r}")
        require_target_latents = self.world_config.get("require_target_latents", True)
        if not isinstance(require_target_latents, bool):
            raise TypeError(
                "world_config['require_target_latents'] must be bool, "
                f"got {type(require_target_latents)!r}"
            )

        prompt_messages = _build_qwen_messages(record.conversations, record.image_paths)
        qwen_inputs = _tokenize_messages(
            self.processor,
            messages=prompt_messages,
            gauss_token_count=fine_k,
            add_generation_prompt=False,
        )
        qwen_inputs["coarse_gauss_values"] = self.gauss_cache.load_for_record(record)

        targets: dict[str, Any] = {}
        if require_target_latents:
            rgb_latents, depth_latents = _load_target_latents(record)
            targets["target_latents"] = {
                "rgb_latents": rgb_latents,
                "depth_latents": depth_latents,
            }

        return GaussianDWMSample(
            sample_uid=record.sample_uid,
            dataset_name=record.dataset_name,
            task="world",
            qwen_inputs=qwen_inputs,
            task_inputs={
                "ref_pixel_values": _load_ref_rgb(record.ref_image_path),
                "pseudo_pixel_values": _load_pseudo_rgb(record.pseudo_video_paths),
                "pseudo_depth_values": _load_pseudo_depth(
                    record.pseudo_depth_paths,
                    with_valid_mask=with_valid_mask,
                ),
            },
            targets=targets,
            meta={
                "source_index": record.source_index,
                "gt_rgb_frame_paths": list(record.gt_rgb_frame_paths or []),
                "gt_depth_npz_paths": list(record.gt_depth_npz_paths or []),
                "fps": record.fps,
                "motion_bucket_id": record.motion_bucket_id,
                "pseudo_source_name": record.pseudo_source_name,
                "pseudo_source_kind": record.pseudo_source_kind,
            },
        )

    def _fine_k(self, record: ProcessedRecord) -> int:
        if record.task not in self.fine_k_by_task:
            raise KeyError(f"No gaussian fine_k configured for task {record.task!r}")
        fine_k = self.fine_k_by_task[record.task]
        if fine_k <= 0:
            raise ValueError(f"fine_k for task {record.task!r} must be positive, got {fine_k}")
        return fine_k


def _build_qwen_messages(
    conversations: list[dict[str, str]],
    image_paths: list[str],
) -> list[dict[str, Any]]:
    if not conversations:
        raise ValueError("conversations must not be empty")
    if not image_paths:
        raise ValueError("image_paths must not be empty")

    messages: list[dict[str, Any]] = []
    injected_first_user = False
    for idx, message in enumerate(conversations):
        role = message.get("role")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"conversations[{idx}].role must be system/user/assistant, got {role!r}")
        content = message.get("content")
        if not isinstance(content, str):
            raise TypeError(f"conversations[{idx}].content must be str, got {type(content)!r}")

        turn_items: list[dict[str, Any]] = []
        if role == "user" and not injected_first_user:
            injected_first_user = True
            for path in image_paths:
                turn_items.append(
                    {
                        "type": "image",
                        "image": path,
                        "resized_height": WORLD_SIZE_WH[1],
                        "resized_width": WORLD_SIZE_WH[0],
                    }
                )
            turn_items.append({"type": "gauss", "gauss": "gaussian_placeholder"})
        turn_items.append({"type": "text", "text": content})
        messages.append({"role": role, "content": turn_items})

    if not injected_first_user:
        raise ValueError("At least one user turn is required for image and gaussian inputs")
    return messages


def _tokenize_messages(
    processor: Any,
    *,
    messages: list[dict[str, Any]],
    gauss_token_count: int,
    add_generation_prompt: bool,
) -> dict[str, torch.Tensor]:
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
    )
    if prompt.count(GAUSS_TEMPLATE_SENTINEL) != 1:
        raise ValueError("Prompt must contain exactly one gaussian placeholder block")
    gauss_block = "<|gaussian_start|>" + ("<|gaussian_pad|>" * gauss_token_count) + "<|gaussian_end|>"
    prompt = prompt.replace(GAUSS_TEMPLATE_SENTINEL, gauss_block)

    encoded = processor(
        text=[prompt],
        images=_image_inputs_from_messages(messages),
        return_tensors="pt",
        padding=False,
    )
    return {key: value.squeeze(0) if key != "image_grid_thw" else value for key, value in encoded.items()}


def _image_inputs_from_messages(messages: list[dict[str, Any]]) -> list[Image.Image] | None:
    images: list[Image.Image] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image":
                continue
            image_path = item.get("image")
            if not isinstance(image_path, str):
                raise TypeError(f"Image item must carry a string path, got {type(image_path)!r}")
            with Image.open(image_path) as image:
                image = image.convert("RGB").resize(WORLD_SIZE_WH)
                images.append(image.copy())
    return images or None


def _build_assistant_only_labels(tokenizer: Any, input_ids: torch.Tensor) -> torch.Tensor:
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={tuple(input_ids.shape)}")

    labels = input_ids.clone()
    keep_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    im_start = tokenizer.convert_tokens_to_ids("<|im_start|>")
    im_end = tokenizer.convert_tokens_to_ids("<|im_end|>")
    ids = input_ids.tolist()

    i = 0
    while i < len(ids):
        if ids[i] != im_start or i + 2 >= len(ids):
            i += 1
            continue
        role_token = tokenizer.convert_ids_to_tokens(ids[i + 1])
        if role_token != "assistant":
            i += 1
            continue
        start = i + 3
        end = None
        for j in range(start, len(ids)):
            if ids[j] == im_end:
                end = j
                break
        if end is None:
            keep_mask[start:] = True
            break
        keep_mask[start : end + 1] = True
        i = end + 1

    labels[~keep_mask] = IGNORE_INDEX
    return labels


def _last_assistant_text(conversations: list[dict[str, str]]) -> str | None:
    for message in reversed(conversations):
        if message.get("role") == "assistant":
            content = message.get("content")
            if not isinstance(content, str):
                raise TypeError(f"Assistant content must be str, got {type(content)!r}")
            return content
    return None


def _drop_last_assistant_turn(conversations: list[dict[str, str]]) -> list[dict[str, str]]:
    output = [dict(message) for message in conversations]
    if output and output[-1].get("role") == "assistant":
        output.pop()
    return output


def _load_ref_rgb(image_path: str | None) -> torch.Tensor:
    if image_path is None:
        raise ValueError("ref_image_path must not be empty for world samples")
    with Image.open(image_path) as image:
        image = image.convert("RGB").resize(WORLD_SIZE_WH)
        array = np.asarray(image, dtype=np.float32)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0


def _load_pseudo_rgb(image_paths: list[str] | None) -> torch.Tensor:
    if not image_paths:
        raise ValueError("pseudo_video_paths must not be empty for world samples")
    frames = []
    for path in image_paths:
        with Image.open(path) as image:
            image = image.convert("RGB").resize(WORLD_SIZE_WH)
            array = np.asarray(image, dtype=np.float32)
        image_tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous() / 255.0
        frames.append(_to_minus_one_one(image_tensor))
    return torch.stack(frames, dim=0)


def _load_pseudo_depth(
    depth_paths: list[str] | None,
    *,
    with_valid_mask: bool,
) -> torch.Tensor:
    if not depth_paths:
        raise ValueError("pseudo_depth_paths must not be empty for world samples")
    frames = []
    for path in depth_paths:
        with np.load(path) as payload:
            if "depth_proj" not in payload:
                raise KeyError(f"World pseudo depth npz must contain depth_proj: {path}")
            depth_proj = np.array(payload["depth_proj"], copy=True)
        frames.append(_prepare_depth(depth_proj, with_valid_mask=with_valid_mask))
    return torch.stack(frames, dim=0)


def _prepare_depth(depth_hw: np.ndarray, *, with_valid_mask: bool) -> torch.Tensor:
    if depth_hw.ndim != 2:
        raise ValueError(f"Expected HW depth array, got shape={tuple(depth_hw.shape)}")
    depth = np.asarray(depth_hw, dtype=np.float32)
    valid_mask = ((depth > DEPTH_MIN) & np.isfinite(depth)).astype(np.float32)
    depth = np.clip(depth, DEPTH_MIN, DEPTH_MAX) / DEPTH_MAX
    if with_valid_mask:
        channels = [depth, depth, valid_mask]
    else:
        channels = [depth, depth, depth]
    resized = [
        np.asarray(
            Image.fromarray(channel, mode="F").resize(
                WORLD_SIZE_WH,
                resample=Image.Resampling.NEAREST,
            ),
            dtype=np.float32,
        )
        for channel in channels
    ]
    return _to_minus_one_one(torch.from_numpy(np.stack(resized, axis=0)).contiguous())


def _to_minus_one_one(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(dtype=torch.float32) * 2.0 - 1.0


def _load_target_latents(record: ProcessedRecord) -> tuple[torch.Tensor, torch.Tensor]:
    if record.target_latent_pack_path and record.target_latent_indices:
        return _load_target_latents_from_pack(record)
    if record.target_latent_paths:
        return _load_target_latents_from_paths(record)
    raise ValueError(f"World record {record.sample_uid} must define target latent paths or a packed latent source")


def _validate_target_latent_length(record: ProcessedRecord, expected_len: int) -> None:
    for label, paths in (
        ("pseudo_video_paths", record.pseudo_video_paths),
        ("pseudo_depth_paths", record.pseudo_depth_paths),
    ):
        if paths is not None and len(paths) != expected_len:
            raise ValueError(
                f"World record {record.sample_uid} has {label} length {len(paths)} "
                f"but target latent length {expected_len}"
            )


def _load_target_latents_from_paths(record: ProcessedRecord) -> tuple[torch.Tensor, torch.Tensor]:
    if not record.target_latent_paths:
        raise ValueError(f"World record {record.sample_uid} must define target_latent_paths")
    _validate_target_latent_length(record, len(record.target_latent_paths))

    rgb_latents = []
    depth_latents = []
    expected_shape = None
    for path in record.target_latent_paths:
        payload = torch.load(Path(path), map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"World latent payload must be a mapping: {path}")
        rgb = torch.as_tensor(payload.get("rgb_latent"), dtype=torch.float32)
        depth = torch.as_tensor(payload.get("depth_latent"), dtype=torch.float32)
        if rgb.ndim != 3 or depth.ndim != 3 or rgb.shape != depth.shape:
            raise ValueError(f"World latent payload must contain matching [C,H,W] rgb/depth tensors: {path}")
        if tuple(rgb.shape) != TARGET_LATENT_CHW:
            raise ValueError(
                f"World frame latent payload at {path} must have rgb/depth shape [4,56,96], "
                f"got {tuple(rgb.shape)}"
            )
        if expected_shape is None:
            expected_shape = tuple(rgb.shape)
        elif tuple(rgb.shape) != expected_shape:
            raise ValueError(f"World latent shape mismatch in {path}: {tuple(rgb.shape)} != {expected_shape}")
        rgb_latents.append(rgb)
        depth_latents.append(depth)
    return torch.stack(rgb_latents, dim=0), torch.stack(depth_latents, dim=0)


def _load_target_latents_from_pack(record: ProcessedRecord) -> tuple[torch.Tensor, torch.Tensor]:
    if record.target_latent_pack_path is None:
        raise ValueError(f"World record {record.sample_uid} must define target_latent_pack_path")
    if not record.target_latent_indices:
        raise ValueError(f"World record {record.sample_uid} must define target_latent_indices")
    indices = [int(index) for index in record.target_latent_indices]
    _validate_target_latent_length(record, len(indices))
    pack, _meta = _open_target_latent_pack(record.target_latent_pack_path)
    if pack.ndim != 5 or pack.shape[1] != 2 or tuple(pack.shape[2:]) != TARGET_LATENT_CHW:
        raise ValueError(
            f"World target latent pack must have shape [N,2,4,56,96], got {tuple(pack.shape)}: "
            f"{record.target_latent_pack_path}"
        )
    if min(indices) < 0 or max(indices) >= pack.shape[0]:
        raise IndexError(
            f"World record {record.sample_uid} target_latent_indices out of bounds for "
            f"{record.target_latent_pack_path}: {indices}"
        )
    if indices == list(range(indices[0], indices[0] + len(indices))):
        window = pack[indices[0] : indices[0] + len(indices)]
    else:
        window = pack[indices]
    tensor = torch.from_numpy(np.array(window, copy=True)).to(dtype=torch.float32)
    return tensor[:, 0].contiguous(), tensor[:, 1].contiguous()


@lru_cache(maxsize=128)
def _open_target_latent_pack(pack_path: str) -> tuple[np.memmap, dict[str, Any]]:
    path = Path(pack_path)
    meta_path = path.with_name("meta.json")
    if not path.exists():
        raise FileNotFoundError(f"World target latent pack not found: {path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"World target latent pack metadata not found: {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    streams = meta.get("streams")
    if streams not in (["rgb", "depth"], ["rgb_latent", "depth_latent"]):
        raise ValueError(f"World target latent pack metadata has unsupported streams={streams!r}: {meta_path}")
    shape = tuple(int(value) for value in meta["shape"])
    if len(shape) != 5 or shape[1] != 2 or tuple(shape[2:]) != TARGET_LATENT_CHW:
        raise ValueError(f"World target latent pack metadata must define shape [N,2,4,56,96]: {meta_path}")
    dtype = np.dtype(str(meta["dtype"]))
    return np.memmap(path, mode="r", dtype=dtype, shape=shape), meta
