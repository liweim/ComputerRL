#!/usr/bin/env python3

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


def load_dataset(path: Path) -> Dict[str, List[str]]:
    data = json.loads(path.read_text())
    return {domain: list(task_ids) for domain, task_ids in data.items()}


def compute_domain_quotas(dataset: Dict[str, List[str]], target_size: int) -> Dict[str, int]:
    counts = {domain: len(task_ids) for domain, task_ids in dataset.items()}
    total = sum(counts.values())
    raw = {domain: counts[domain] * target_size / total for domain in counts}
    quotas = {domain: math.floor(value) for domain, value in raw.items()}
    remaining = target_size - sum(quotas.values())
    order = sorted(counts, key=lambda domain: (raw[domain] - quotas[domain], counts[domain]), reverse=True)
    for domain in order[:remaining]:
        quotas[domain] += 1
    return quotas


def load_method_scores(results_root: Path, all_tasks: Sequence[str]) -> Dict[str, Dict[str, float]]:
    task_set = set(all_tasks)
    per_method: Dict[str, Dict[str, float]] = defaultdict(dict)
    for result_path in results_root.glob("*/*/*/result.txt"):
        method = result_path.parts[-4]
        task_id = result_path.parts[-2]
        if task_id not in task_set:
            continue
        text = result_path.read_text().strip()
        if not text:
            continue
        try:
            score = float(text)
        except ValueError:
            continue
        per_method[method][task_id] = score
    return {method: dict(scores) for method, scores in per_method.items()}


def filter_complete_methods(
    method_scores: Dict[str, Dict[str, float]],
    all_tasks: Sequence[str],
) -> Dict[str, Dict[str, float]]:
    target = set(all_tasks)
    return {
        method: scores
        for method, scores in method_scores.items()
        if target.issubset(scores.keys())
    }


def build_task_vectors(
    complete_methods: Dict[str, Dict[str, float]],
    all_tasks: Sequence[str],
) -> Tuple[List[str], Dict[str, List[float]], List[float]]:
    methods = sorted(complete_methods)
    task_vectors = {
        task_id: [complete_methods[method][task_id] for method in methods]
        for task_id in all_tasks
    }
    target_means = []
    total = len(all_tasks)
    for method in methods:
        target_means.append(sum(complete_methods[method][task_id] for task_id in all_tasks) / total)
    return methods, task_vectors, target_means


def objective(
    selected_tasks: Sequence[str],
    task_vectors: Dict[str, List[float]],
    target_means: Sequence[float],
) -> float:
    size = len(selected_tasks)
    dims = len(target_means)
    sums = [0.0] * dims
    for task_id in selected_tasks:
        vec = task_vectors[task_id]
        for idx in range(dims):
            sums[idx] += vec[idx]
    err = 0.0
    for idx, target in enumerate(target_means):
        diff = sums[idx] / size - target
        err += diff * diff
    return err


def initial_selection(
    dataset: Dict[str, List[str]],
    quotas: Dict[str, int],
    rng: random.Random,
) -> List[str]:
    chosen: List[str] = []
    for domain, task_ids in dataset.items():
        pool = list(task_ids)
        rng.shuffle(pool)
        chosen.extend(pool[: quotas[domain]])
    return chosen


def improve_selection(
    dataset: Dict[str, List[str]],
    quotas: Dict[str, int],
    task_vectors: Dict[str, List[float]],
    target_means: Sequence[float],
    seed: int,
    restarts: int,
    steps_per_restart: int,
) -> Tuple[List[str], float]:
    rng = random.Random(seed)
    by_domain = {domain: list(task_ids) for domain, task_ids in dataset.items()}
    best_selection: List[str] = []
    best_score = float("inf")

    for restart in range(restarts):
        local_rng = random.Random(rng.randint(0, 10**9) + restart)
        selection = initial_selection(dataset, quotas, local_rng)
        selected_by_domain = {
            domain: set(task_id for task_id in selection if task_id in by_domain[domain])
            for domain in by_domain
        }
        current_score = objective(selection, task_vectors, target_means)
        stagnant = 0

        for _ in range(steps_per_restart):
            domain = local_rng.choice(list(by_domain))
            if quotas[domain] == 0 or quotas[domain] == len(by_domain[domain]):
                continue
            out_task = local_rng.choice(list(selected_by_domain[domain]))
            candidates = [task_id for task_id in by_domain[domain] if task_id not in selected_by_domain[domain]]
            in_task = local_rng.choice(candidates)

            trial = [in_task if task_id == out_task else task_id for task_id in selection]
            trial_score = objective(trial, task_vectors, target_means)

            if trial_score < current_score:
                selection = trial
                selected_by_domain[domain].remove(out_task)
                selected_by_domain[domain].add(in_task)
                current_score = trial_score
                stagnant = 0
            else:
                stagnant += 1
                if stagnant > 2000:
                    break

        if current_score < best_score:
            best_selection = list(selection)
            best_score = current_score

    return sorted(best_selection), best_score


def selection_report(
    selected_tasks: Sequence[str],
    dataset: Dict[str, List[str]],
    methods: Sequence[str],
    complete_methods: Dict[str, Dict[str, float]],
    target_means: Sequence[float],
) -> Dict[str, object]:
    selected_set = set(selected_tasks)
    sampled_dataset = {
        domain: [task_id for task_id in task_ids if task_id in selected_set]
        for domain, task_ids in dataset.items()
    }
    per_method = {}
    for idx, method in enumerate(methods):
        full_mean = target_means[idx]
        sample_mean = sum(complete_methods[method][task_id] for task_id in selected_tasks) / len(selected_tasks)
        per_method[method] = {
            "full_mean": full_mean,
            "sample_mean": sample_mean,
            "abs_diff": abs(sample_mean - full_mean),
        }

    abs_diffs = [stats["abs_diff"] for stats in per_method.values()]
    return {
        "sample_size": len(selected_tasks),
        "domain_counts": {domain: len(task_ids) for domain, task_ids in sampled_dataset.items()},
        "methods_used": list(methods),
        "mae": sum(abs_diffs) / len(abs_diffs),
        "max_abs_diff": max(abs_diffs) if abs_diffs else 0.0,
        "per_method": per_method,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("evaluation_examples/test_medium.json"))
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--output-json", type=Path, default=Path("evaluation_examples/test_medium_sampled_30pct.json"))
    parser.add_argument("--output-report", type=Path, default=Path("evaluation_examples/test_medium_sampled_30pct_report.json"))
    parser.add_argument("--ratio", type=float, default=0.3)
    parser.add_argument("--seed", type=int, default=20260407)
    parser.add_argument("--restarts", type=int, default=200)
    parser.add_argument("--steps-per-restart", type=int, default=20000)
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    all_tasks = [task_id for task_ids in dataset.values() for task_id in task_ids]
    target_size = round(len(all_tasks) * args.ratio)
    quotas = compute_domain_quotas(dataset, target_size)

    method_scores = load_method_scores(args.results_root, all_tasks)
    complete_methods = filter_complete_methods(method_scores, all_tasks)
    if not complete_methods:
        raise SystemExit("No complete methods found for the dataset.")

    methods, task_vectors, target_means = build_task_vectors(complete_methods, all_tasks)
    selection, score = improve_selection(
        dataset=dataset,
        quotas=quotas,
        task_vectors=task_vectors,
        target_means=target_means,
        seed=args.seed,
        restarts=args.restarts,
        steps_per_restart=args.steps_per_restart,
    )

    selected_set = set(selection)
    sampled_dataset = {
        domain: [task_id for task_id in task_ids if task_id in selected_set]
        for domain, task_ids in dataset.items()
    }
    report = selection_report(selection, dataset, methods, complete_methods, target_means)
    report["objective"] = score
    report["quotas"] = quotas

    args.output_json.write_text(json.dumps(sampled_dataset, indent=2) + "\n")
    args.output_report.write_text(json.dumps(report, indent=2) + "\n")

    print(f"Selected {len(selection)} / {len(all_tasks)} tasks")
    print(f"Methods used: {len(methods)}")
    print(f"Objective: {score:.8f}")
    for domain in dataset:
        print(f"{domain}: {len(sampled_dataset[domain])}")
    print("Per-method differences:")
    for method in methods:
        stats = report["per_method"][method]
        print(
            f"{method}: full={stats['full_mean']:.6f} sample={stats['sample_mean']:.6f} "
            f"abs_diff={stats['abs_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
