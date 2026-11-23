"""SWE-bench Lite loader and deterministic patch ground-truth parser."""
from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SWEBenchTask:
    """Single SWE-bench Lite instance with parsed patch ground truth."""

    instance_id: str
    repo: str
    base_commit: str
    problem_statement: str
    patch: str
    ground_truth_files: list[str]
    ground_truth_hunks: dict[str, list[tuple[int, int]]]
    patch_added_lines: int
    patch_removed_lines: int
    num_gt_files: int


def extract_ground_truth_files(patch: str) -> list[str]:
    """Extract sorted unique modified file paths from a unified git diff patch."""
    files = re.findall(r"^\+\+\+ b/(.+)$", patch, flags=re.MULTILINE)
    return sorted({f.strip() for f in files if f.strip() and f.strip() != "/dev/null"})


def extract_patch_line_ranges(patch: str) -> dict[str, list[tuple[int, int]]]:
    """Parse modified line ranges (start, end) in the base commit per file."""
    hunks: dict[str, list[tuple[int, int]]] = {}
    current_file: str | None = None
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            candidate = line[len("+++ b/") :].strip()
            current_file = None if candidate == "/dev/null" else candidate
            if current_file and current_file not in hunks:
                hunks[current_file] = []
        elif line.startswith("@@ ") and current_file is not None:
            match = re.search(r"@@ -(\d+)(?:,(\d+))? \+", line)
            if match:
                start = int(match.group(1))
                length = int(match.group(2)) if match.group(2) is not None else 1
                end = start + max(1, length) - 1
                hunks[current_file].append((start, end))
    return hunks


def count_patch_churn(patch: str) -> tuple[int, int]:
    """Return (added_lines, removed_lines) excluding unified diff file headers."""
    added = 0
    removed = 0
    for line in patch.splitlines():
        if line.startswith(("+++ ", "--- ")):
            continue
        if line.startswith("+"):
            added += 1
        elif line.startswith("-"):
            removed += 1
    return added, removed


def load_swebench_lite(
    limit: int = 20,
    split: str = "test",
    repos: list[str] | None = None,
    dataset_name: str = "princeton-nlp/SWE-bench_Lite",
) -> list[SWEBenchTask]:
    """Load SWE-bench Lite tasks lazily from Hugging Face and extract patch ground truth."""
    from datasets import load_dataset  # Lazy import keeps offline unit tests fast

    ds = load_dataset(dataset_name, split=split)
    repo_filter = set(repos) if repos is not None else None

    tasks: list[SWEBenchTask] = []
    for row in ds:
        if repo_filter is not None and row["repo"] not in repo_filter:
            continue
        patch = row["patch"]
        gt_files = extract_ground_truth_files(patch)
        hunks = extract_patch_line_ranges(patch)
        added, removed = count_patch_churn(patch)
        tasks.append(
            SWEBenchTask(
                instance_id=row["instance_id"],
                repo=row["repo"],
                base_commit=row["base_commit"],
                problem_statement=row["problem_statement"],
                patch=patch,
                ground_truth_files=gt_files,
                ground_truth_hunks=hunks,
                patch_added_lines=added,
                patch_removed_lines=removed,
                num_gt_files=len(gt_files),
            )
        )
        if len(tasks) >= limit:
            break
    return tasks


def tasks_to_dataframe(tasks: list[SWEBenchTask]) -> pd.DataFrame:
    """Convert SWEBenchTask objects into a tabular Pandas DataFrame."""
    records = [
        {
            "instance_id": t.instance_id,
            "repo": t.repo,
            "base_commit": t.base_commit,
            "problem_statement": t.problem_statement,
            "ground_truth_files": t.ground_truth_files,
            "num_gt_files": t.num_gt_files,
            "patch_added_lines": t.patch_added_lines,
            "patch_removed_lines": t.patch_removed_lines,
        }
        for t in tasks
    ]
    return pd.DataFrame(records)
