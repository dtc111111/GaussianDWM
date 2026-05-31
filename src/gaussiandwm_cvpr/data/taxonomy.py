from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from gaussiandwm_cvpr.utils import load_yaml


@dataclass(frozen=True)
class Taxonomy:
    major_tasks: tuple[str, ...]
    qa_groups: tuple[str, ...]
    qa_subtasks: dict[str, tuple[str, ...]]

    def validate_task(self, task: str) -> str:
        if task not in self.major_tasks:
            raise ValueError(f"Unsupported major task: {task!r}")
        return task

    def validate_qa(self, qa_group: str, qa_subtask: str) -> tuple[str, str]:
        if qa_group not in self.qa_groups:
            raise ValueError(f"Unsupported QA group: {qa_group!r}")
        if qa_subtask not in self.qa_subtasks[qa_group]:
            raise ValueError(
                f"Unsupported QA subtask {qa_subtask!r} for group {qa_group!r}"
            )
        return qa_group, qa_subtask


def load_taxonomy(path: str | Path) -> Taxonomy:
    payload = load_yaml(path)
    major_tasks = payload.get("major_tasks")
    qa_group_budgets = payload.get("qa_group_budgets")
    qa_subtask_budgets = payload.get("qa_subtask_budgets")
    if not isinstance(major_tasks, list) or not all(
        isinstance(item, str) for item in major_tasks
    ):
        raise TypeError("major_tasks must be a list of strings")
    if not isinstance(qa_group_budgets, dict) or not qa_group_budgets:
        raise TypeError("qa_group_budgets must be a non-empty mapping")
    if not isinstance(qa_subtask_budgets, dict):
        raise TypeError("qa_subtask_budgets must be a mapping")

    qa_groups = tuple(str(name) for name in qa_group_budgets.keys())
    qa_subtasks: dict[str, tuple[str, ...]] = {}
    for group in qa_groups:
        subtasks = qa_subtask_budgets.get(group)
        if not isinstance(subtasks, dict) or not subtasks:
            raise TypeError(f"qa_subtask_budgets[{group!r}] must be a non-empty mapping")
        qa_subtasks[group] = tuple(str(name) for name in subtasks.keys())

    return Taxonomy(
        major_tasks=tuple(major_tasks),
        qa_groups=qa_groups,
        qa_subtasks=qa_subtasks,
    )
