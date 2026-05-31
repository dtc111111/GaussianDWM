from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from gaussiandwm_cvpr.data.taxonomy import Taxonomy
from gaussiandwm_cvpr.utils import load_json_or_jsonl, resolve_path


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    major_task: str
    split: str
    annotation_path: str
    annotation_root: str
    image_root: str
    gauss_root: str
    clip_text_root: str | None = None


@dataclass(frozen=True)
class ProcessedRecord:
    sample_uid: str
    dataset_name: str
    task: str
    scene_idx: int
    frame_idx: int
    views: list[str]
    image_paths: list[str]
    gauss_paths: list[str]
    source_index: int
    conversations: list[dict[str, str]]
    qa_group: str | None = None
    qa_subtask: str | None = None
    clip_text_embed_path: str | None = None
    clip_text_embed_row: int | None = None
    ref_image_path: str | None = None
    pseudo_video_paths: list[str] | None = None
    pseudo_depth_paths: list[str] | None = None
    target_latent_paths: list[str] | None = None
    target_latent_pack_path: str | None = None
    target_latent_indices: list[int] | None = None
    gt_rgb_frame_paths: list[str] | None = None
    gt_depth_npz_paths: list[str] | None = None
    pseudo_source_name: str | None = None
    pseudo_source_kind: str | None = None
    fps: int | None = None
    motion_bucket_id: int | None = None


def load_records(
    data_config: dict[str, Any],
    *,
    taxonomy: Taxonomy,
    split: str,
    dataset_names: list[str] | None = None,
    task_filter: list[str] | None = None,
    package_root: str | Path | None = None,
    require_world_targets: bool = True,
) -> list[ProcessedRecord]:
    datasets = data_config.get("datasets")
    if not isinstance(datasets, dict):
        raise TypeError("data_config['datasets'] must be a mapping")

    allowed_tasks: set[str] | None = None
    if task_filter is not None:
        if not isinstance(task_filter, list) or not task_filter:
            raise TypeError("task_filter must be a non-empty list of task names or None")
        allowed_tasks = set()
        for task in task_filter:
            if not isinstance(task, str):
                raise TypeError("task_filter entries must be strings")
            allowed_tasks.add(taxonomy.validate_task(task))

    root = Path(package_root).resolve() if package_root is not None else Path.cwd()
    selected_names = dataset_names if dataset_names is not None else list(datasets)
    records: list[ProcessedRecord] = []
    for dataset_name in selected_names:
        dataset_payload = datasets.get(dataset_name)
        if not isinstance(dataset_payload, dict):
            raise KeyError(f"Unknown dataset: {dataset_name!r}")

        dataset_task = _require_str(
            dataset_payload.get("major_task"),
            f"datasets[{dataset_name}].major_task",
        )
        dataset_split = _require_str(dataset_payload.get("split"), f"datasets[{dataset_name}].split")
        taxonomy.validate_task(dataset_task)
        if allowed_tasks is not None and dataset_task not in allowed_tasks:
            continue
        if dataset_split != split:
            continue

        annotation_root = _resolve_config_path(dataset_payload["annotation_root"], root)
        image_root = _resolve_config_path(dataset_payload["image_root"], root)
        gauss_root = _resolve_config_path(dataset_payload["gauss_root"], root)
        clip_text_root = None
        clip_config = data_config.get("clip_text_feature")
        if isinstance(clip_config, dict) and clip_config.get("root") is not None:
            clip_text_root = _resolve_config_path(clip_config["root"], root)
        if dataset_payload.get("clip_text_root") is not None:
            clip_text_root = _resolve_config_path(dataset_payload["clip_text_root"], root)

        spec = DatasetSpec(
            name=dataset_name,
            major_task=dataset_task,
            split=dataset_split,
            annotation_path=str(resolve_path(dataset_payload["annotation_path"], base=annotation_root)),
            annotation_root=str(annotation_root),
            image_root=str(image_root),
            gauss_root=str(gauss_root),
            clip_text_root=None if clip_text_root is None else str(clip_text_root),
        )

        for source_index, raw_record in enumerate(load_json_or_jsonl(spec.annotation_path)):
            if not isinstance(raw_record, dict):
                raise TypeError(f"Annotation record must be a mapping: {spec.annotation_path}")
            records.append(
                _parse_record(
                    raw_record,
                    spec,
                    taxonomy,
                    source_index=source_index,
                    require_world_targets=require_world_targets,
                )
            )

    return records


def _parse_record(
    raw: dict[str, Any],
    spec: DatasetSpec,
    taxonomy: Taxonomy,
    *,
    source_index: int,
    require_world_targets: bool,
) -> ProcessedRecord:
    task = _require_str(raw.get("major_task"), "major_task")
    taxonomy.validate_task(task)
    if task != spec.major_task:
        raise ValueError(f"Record major_task {task!r} does not match dataset major_task {spec.major_task!r}")

    scene_idx = _require_non_negative_int(raw.get("scene_idx"), "scene_idx")
    frame_idx = _require_non_negative_int(raw.get("frame_idx"), "frame_idx")
    views = _require_str_list(raw.get("views"), "views")
    sample_uid = raw.get("sample_uid")
    if sample_uid is None:
        sample_uid = _make_sample_uid(spec.name, task, scene_idx, frame_idx, views)
    else:
        sample_uid = _require_str(sample_uid, "sample_uid")
    image_paths = _resolve_path_list(raw.get("image_paths"), spec.image_root, "image_paths")
    gauss_paths = _resolve_path_list(raw.get("gauss_paths"), spec.gauss_root, "gauss_paths")
    conversations = _validate_conversations(raw.get("conversations"))

    if task == "qa":
        qa_group = _require_str(raw.get("qa_group"), "qa_group")
        qa_subtask = _require_str(raw.get("qa_subtask"), "qa_subtask")
        taxonomy.validate_qa(qa_group, qa_subtask)
        clip_text_embed_path = _resolve_required_path(
            raw.get("clip_text_embed_path"), spec.clip_text_root, "clip_text_embed_path"
        )
        clip_text_embed_row = _require_non_negative_int(
            raw.get("clip_text_embed_row"), "clip_text_embed_row"
        )
        return ProcessedRecord(
            sample_uid=sample_uid,
            dataset_name=spec.name,
            task=task,
            scene_idx=scene_idx,
            frame_idx=frame_idx,
            views=views,
            image_paths=image_paths,
            gauss_paths=gauss_paths,
            source_index=source_index,
            conversations=conversations,
            qa_group=qa_group,
            qa_subtask=qa_subtask,
            clip_text_embed_path=clip_text_embed_path,
            clip_text_embed_row=clip_text_embed_row,
        )

    if task != "world":
        raise ValueError(f"Unsupported record task: {task!r}")

    target_latent_paths = _resolve_optional_path_list(
        raw.get("target_latent_paths"), spec.annotation_root, "target_latent_paths"
    )
    target_latent_pack_path = None
    target_latent_indices = None
    if raw.get("target_latent_pack_path") is not None:
        target_latent_pack_path = str(
            resolve_path(_require_str(raw["target_latent_pack_path"], "target_latent_pack_path"), base=spec.annotation_root)
        )
        raw_indices = _require_list(raw.get("target_latent_indices"), "target_latent_indices")
        target_latent_indices = [
            _require_non_negative_int(item, "target_latent_indices") for item in raw_indices
        ]
        if not target_latent_indices:
            raise ValueError("target_latent_indices must be non-empty")
    if require_world_targets and not target_latent_paths and target_latent_pack_path is None:
        raise ValueError(
            "World records require target_latent_paths or target_latent_pack_path "
            "when require_world_targets=True"
        )

    return ProcessedRecord(
        sample_uid=sample_uid,
        dataset_name=spec.name,
        task=task,
        scene_idx=scene_idx,
        frame_idx=frame_idx,
        views=views,
        image_paths=image_paths,
        gauss_paths=gauss_paths,
        source_index=source_index,
        conversations=conversations,
        ref_image_path=_resolve_required_path(raw.get("ref_image_path"), spec.image_root, "ref_image_path"),
        pseudo_video_paths=_resolve_path_list(
            raw.get("pseudo_video_paths"), spec.image_root, "pseudo_video_paths"
        ),
        pseudo_depth_paths=_resolve_path_list(
            raw.get("pseudo_depth_paths"), spec.image_root, "pseudo_depth_paths"
        ),
        target_latent_paths=target_latent_paths,
        target_latent_pack_path=target_latent_pack_path,
        target_latent_indices=target_latent_indices,
        gt_rgb_frame_paths=_resolve_optional_path_list(
            raw.get("gt_rgb_frame_paths"), spec.image_root, "gt_rgb_frame_paths"
        ),
        gt_depth_npz_paths=_resolve_optional_path_list(
            raw.get("gt_depth_npz_paths"), spec.image_root, "gt_depth_npz_paths"
        ),
        pseudo_source_name=_optional_str(raw.get("pseudo_source_name"), "pseudo_source_name"),
        pseudo_source_kind=_optional_str(raw.get("pseudo_source_kind"), "pseudo_source_kind"),
        fps=_require_non_negative_int(raw.get("fps"), "fps"),
        motion_bucket_id=_require_non_negative_int(raw.get("motion_bucket_id"), "motion_bucket_id"),
    )


def _resolve_config_path(value: str | Path, package_root: Path) -> Path:
    return resolve_path(value, base=package_root)


def _resolve_required_path(value: Any, base: str | None, name: str) -> str:
    if base is None:
        raise ValueError(f"{name} needs a configured root")
    return str(resolve_path(_require_str(value, name), base=base))


def _resolve_path_list(value: Any, base: str, name: str) -> list[str]:
    return [str(resolve_path(path, base=base)) for path in _require_str_list(value, name)]


def _resolve_optional_path_list(value: Any, base: str, name: str) -> list[str]:
    if value is None:
        return []
    return _resolve_path_list(value, base, name)


def _validate_conversations(value: Any) -> list[dict[str, str]]:
    conversations = _require_list(value, "conversations")
    if not conversations:
        raise ValueError("conversations must be non-empty")
    parsed: list[dict[str, str]] = []
    for index, item in enumerate(conversations):
        if not isinstance(item, dict):
            raise TypeError(f"conversations[{index}] must be a mapping")
        role = _require_str(item.get("role"), f"conversations[{index}].role")
        content = _require_str(item.get("content"), f"conversations[{index}].content")
        if role not in {"system", "user", "assistant"}:
            raise ValueError(f"Unsupported conversation role: {role!r}")
        parsed.append({"role": role, "content": content})
    return parsed


def _make_sample_uid(
    dataset_name: str,
    task: str,
    scene_idx: int,
    frame_idx: int,
    views: list[str],
) -> str:
    key = "|".join([dataset_name, task, str(scene_idx), str(frame_idx), ",".join(views)])
    digest = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    return f"{dataset_name}_{task}_{digest}"


def _require_list(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be a list")
    return value


def _require_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{name} must be a non-empty string")
    return value


def _optional_str(value: Any, name: str) -> str | None:
    if value is None:
        return None
    return _require_str(value, name)


def _require_str_list(value: Any, name: str) -> list[str]:
    items = _require_list(value, name)
    if not items or not all(isinstance(item, str) and item for item in items):
        raise TypeError(f"{name} must be a non-empty list of strings")
    return list(items)


def _require_non_negative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise TypeError(f"{name} must be a non-negative int")
    return value
