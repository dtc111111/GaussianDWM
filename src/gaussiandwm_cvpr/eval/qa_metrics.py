from __future__ import annotations

import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from gaussiandwm_cvpr.utils import dump_json, load_json_or_jsonl, load_yaml, timestamp_now


METRIC_PROTOCOL = "public_approximate"
METRIC_NOTE = (
    "Public approximate QA metrics: text groups use exact match and token-overlap F1; "
    "2D/3D visual grounding use simplified local parsers and success approximations. "
    "These are not official paper BLEU/ROUGE/CIDEr/mAP metrics."
)


def evaluate_qa_from_config_dir(
    config_dir: str | Path,
    predictions_path: str | Path,
    output_dir: str | Path,
) -> Path:
    config_root = Path(config_dir).resolve()
    output_root = Path(output_dir).resolve()
    eval_cfg = load_yaml(config_root / "eval.yaml")
    items = load_json_or_jsonl(predictions_path)
    task_filter = set(str(task) for task in eval_cfg.get("task_filter", ["qa"]))
    dataset_filter = set(str(name) for name in eval_cfg.get("dataset_filter", []))

    filtered = []
    for item in items:
        if str(item.get("task_type")) not in task_filter:
            continue
        if dataset_filter and str(item.get("dataset_name")) not in dataset_filter:
            continue
        filtered.append(item)

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in filtered:
        groups[str(item.get("qa_group") or "unknown")].append(item)

    group_summaries: dict[str, Any] = {}
    per_items: list[dict[str, Any]] = []
    enabled_groups = eval_cfg.get("qa", {}).get("enabled_groups")
    enabled = set(str(group) for group in enabled_groups) if isinstance(enabled_groups, list) else None
    for group, group_items in sorted(groups.items()):
        if enabled is not None and group not in enabled:
            continue
        summary, details = evaluate_qa_group(group, group_items)
        summary = {
            "metric_protocol": METRIC_PROTOCOL,
            "metric_note": METRIC_NOTE,
            **summary,
        }
        group_summaries[group] = summary
        per_items.extend(details)

    exact_scores = [item["metrics"].get("exact_match") for item in per_items if "exact_match" in item["metrics"]]
    output_path = output_root / f"qa_metrics_{timestamp_now()}.json"
    dump_json(
        {
            "task_type": "qa",
            "metric_protocol": METRIC_PROTOCOL,
            "metric_note": METRIC_NOTE,
            "config_dir": str(config_root),
            "predictions_path": str(predictions_path),
            "count": len(per_items),
            "summary": {
                "groups": group_summaries,
                "mean_exact_match": _mean([float(value) for value in exact_scores if value is not None]),
            },
            "items": per_items,
        },
        output_path,
    )
    return output_path


def evaluate_qa_group(group: str, items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if group == "NuInteract_Planning":
        return _evaluate_planning(items)
    if group == "NuInteract_2DVG":
        return _evaluate_2dvg(items)
    if group == "NuInteract_3DVG":
        return _evaluate_3dvg(items)
    return _evaluate_text_group(group, items)


def _evaluate_planning(items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    correct = 0
    for item in items:
        pred_label = _parse_planning_label(item.get("pred"))
        gt_label = _parse_planning_label(item.get("gt"))
        is_correct = pred_label is not None and pred_label == gt_label
        correct += int(is_correct)
        details.append(_detail(item, "NuInteract_Planning", not is_correct, 0.0 if is_correct else 1.0, {
            "pred_label": pred_label,
            "gt_label": gt_label,
            "is_correct": is_correct,
        }))
    return {"count": len(items), "accuracy": correct / len(items) if items else 0.0}, details


def _evaluate_text_group(group: str, items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    exact = []
    token_f1 = []
    for item in items:
        pred = str(item.get("pred") or "")
        gt = str(item.get("gt") or "")
        metrics = {
            "exact_match": float(_normalize_text(pred) == _normalize_text(gt)),
            "token_overlap_f1": _token_overlap_f1(pred, gt),
            "pred_token_count": len(_tokens(pred)),
            "gt_token_count": len(_tokens(gt)),
        }
        exact.append(metrics["exact_match"])
        token_f1.append(metrics["token_overlap_f1"])
        details.append(_detail(item, group, metrics["token_overlap_f1"] < 0.2, 1.0 - metrics["token_overlap_f1"], metrics))
    return {
        "count": len(items),
        "exact_match": _mean(exact),
        "token_overlap_f1": _mean(token_f1),
    }, details


def _evaluate_2dvg(items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    f1_values = []
    success_values = []
    for item in items:
        pred_boxes = _parse_2d_boxes(item.get("pred"))
        gt_boxes = _parse_2d_boxes(item.get("gt"))
        f1 = _f1_at_iou50(pred_boxes, gt_boxes)
        parse_ok = bool(pred_boxes) and bool(gt_boxes)
        success = float(parse_ok and f1 >= 0.5)
        f1_values.append(f1)
        success_values.append(success)
        details.append(_detail(item, "NuInteract_2DVG", success < 1.0, 1.0 - f1, {
            "parse_ok": parse_ok,
            "pred_box_count": len(pred_boxes),
            "gt_box_count": len(gt_boxes),
            "f1_iou50": f1,
            "success_iou50": success,
        }))
    return {"count": len(items), "f1_iou50": _mean(f1_values), "success_iou50": _mean(success_values)}, details


def _evaluate_3dvg(items: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    details: list[dict[str, Any]] = []
    success_1m = []
    success_2m = []
    distances = []
    for item in items:
        pred_centers = _parse_3d_centers(item.get("pred"))
        gt_centers = _parse_3d_centers(item.get("gt"))
        distance = _min_center_distance(pred_centers, gt_centers)
        parse_ok = math.isfinite(distance)
        metric_distance = distance if parse_ok else 20.0
        distances.append(metric_distance)
        success_1m.append(float(parse_ok and distance <= 1.0))
        success_2m.append(float(parse_ok and distance <= 2.0))
        details.append(_detail(item, "NuInteract_3DVG", not parse_ok or distance > 2.0, metric_distance, {
            "parse_ok": parse_ok,
            "pred_center_count": len(pred_centers),
            "gt_center_count": len(gt_centers),
            "min_center_distance_m": metric_distance,
            "success_1m": float(parse_ok and distance <= 1.0),
            "success_2m": float(parse_ok and distance <= 2.0),
        }))
    return {
        "count": len(items),
        "min_center_distance_m": _mean(distances),
        "success_1m": _mean(success_1m),
        "success_2m": _mean(success_2m),
    }, details


def _parse_planning_label(text: object) -> str | None:
    normalized = _normalize_text(str(text or ""))
    left_terms = {"left", "turn left", "turn_left", "left turn"}
    right_terms = {"right", "turn right", "turn_right", "right turn"}
    straight_terms = {"straight", "go straight", "forward", "keep straight", "go forward"}
    if any(term in normalized for term in left_terms):
        return "left"
    if any(term in normalized for term in right_terms):
        return "right"
    if any(term in normalized for term in straight_terms):
        return "straight"
    return None


def _parse_2d_boxes(value: object) -> list[list[float]]:
    boxes: list[list[float]] = []
    parsed = _parse_structured_or_numbers(value)
    for entry in _walk_objects(parsed):
        if isinstance(entry, dict):
            for key in ("bbox_2d", "bbox", "box", "boxes"):
                if key in entry:
                    boxes.extend(_numbers_to_boxes(_flat_numbers(entry[key]), width=4))
        elif isinstance(entry, (list, tuple)):
            boxes.extend(_numbers_to_boxes(_flat_numbers(entry), width=4))
    return [_normalize_xyxy(box) for box in boxes]


def _parse_3d_centers(value: object) -> list[list[float]]:
    centers: list[list[float]] = []
    parsed = _parse_structured_or_numbers(value)
    for entry in _walk_objects(parsed):
        if isinstance(entry, dict):
            for key in ("bbox_3d", "center", "location", "position"):
                if key in entry:
                    centers.extend(_numbers_to_centers(_flat_numbers(entry[key])))
        elif isinstance(entry, (list, tuple)):
            centers.extend(_numbers_to_centers(_flat_numbers(entry)))
    return centers


def _parse_structured_or_numbers(value: object) -> object:
    if not isinstance(value, str):
        return value
    text = value.strip()
    if not text:
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            return parser(text)
        except Exception:
            pass
    return _flat_numbers(text)


def _walk_objects(value: object) -> list[object]:
    objects = [value]
    if isinstance(value, dict):
        for child in value.values():
            objects.extend(_walk_objects(child))
    elif isinstance(value, (list, tuple)):
        for child in value:
            objects.extend(_walk_objects(child))
    return objects


def _flat_numbers(value: object) -> list[float]:
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, str):
        return [float(match) for match in re.findall(r"[-+]?(?:\d+\.\d+|\d+|\.\d+)(?:[eE][-+]?\d+)?", value)]
    if isinstance(value, dict):
        values: list[float] = []
        for child in value.values():
            values.extend(_flat_numbers(child))
        return values
    if isinstance(value, (list, tuple)):
        values = []
        for child in value:
            values.extend(_flat_numbers(child))
        return values
    return []


def _numbers_to_boxes(numbers: list[float], *, width: int) -> list[list[float]]:
    return [numbers[index : index + width] for index in range(0, len(numbers) - width + 1, width)]


def _numbers_to_centers(numbers: list[float]) -> list[list[float]]:
    if len(numbers) >= 7:
        return [numbers[index : index + 3] for index in range(0, len(numbers) - 2, 7)]
    if len(numbers) >= 3:
        return [numbers[index : index + 3] for index in range(0, len(numbers) - 2, 3)]
    return []


def _normalize_xyxy(box: list[float]) -> list[float]:
    x1, y1, x2, y2 = [float(value) for value in box[:4]]
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _f1_at_iou50(pred_boxes: list[list[float]], gt_boxes: list[list[float]]) -> float:
    if not pred_boxes and not gt_boxes:
        return 1.0
    if not pred_boxes or not gt_boxes:
        return 0.0
    matched_gt: set[int] = set()
    matched_pred = 0
    for pred in pred_boxes:
        best_gt = -1
        best_iou = 0.0
        for gt_idx, gt in enumerate(gt_boxes):
            if gt_idx in matched_gt:
                continue
            iou = _box_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_idx
        if best_gt >= 0 and best_iou >= 0.5:
            matched_gt.add(best_gt)
            matched_pred += 1
    precision = matched_pred / max(len(pred_boxes), 1)
    recall = matched_pred / max(len(gt_boxes), 1)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    denom = area_a + area_b - inter
    return 0.0 if denom <= 0 else inter / denom


def _min_center_distance(pred_centers: list[list[float]], gt_centers: list[list[float]]) -> float:
    if not pred_centers or not gt_centers:
        return math.inf
    return float(min(math.dist(pred[:3], gt[:3]) for pred in pred_centers for gt in gt_centers))


def _token_overlap_f1(pred: str, gt: str) -> float:
    pred_tokens = _tokens(pred)
    gt_tokens = _tokens(gt)
    if not pred_tokens and not gt_tokens:
        return 1.0
    if not pred_tokens or not gt_tokens:
        return 0.0
    remaining = list(gt_tokens)
    overlap = 0
    for token in pred_tokens:
        if token in remaining:
            remaining.remove(token)
            overlap += 1
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gt_tokens)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _normalize_text(text: str) -> str:
    return " ".join(_tokens(text))


def _detail(
    item: dict[str, Any],
    group: str,
    is_failure: bool,
    failure_score: float,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "task_type": "qa",
        "qa_group": group,
        "sample_uid": item.get("sample_uid"),
        "dataset_name": item.get("dataset_name"),
        "is_failure": is_failure,
        "failure_score": float(failure_score),
        "metrics": metrics,
        "pred": item.get("pred"),
        "gt": item.get("gt"),
        "meta": item.get("meta") if isinstance(item.get("meta"), dict) else {},
    }


def _mean(values: list[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0
