import json
from pathlib import Path

import pandas as pd

RESULTS_ROOT = Path("/data1/lwm/projects/ComputerRL/results")
RUNS = {
    "autoglm-os_baseline": RESULTS_ROOT / "autoglm-os_baseline",
    "autoglm-os_gta1_7b_restart": RESULTS_ROOT / "autoglm-os_gta1_7b_restart",
}


def extract_failure_reason(obj):
    logs = obj.get("action_logs") or []
    # Prefer explicit execution failures
    for entry in reversed(logs):
        if entry.get("execution_success") is False:
            exe_result = entry.get("exe_result")
            if exe_result:
                return str(exe_result)
            action = entry.get("action")
            if action:
                return f"execution_failed: {action}"
    # Check for done state with failure hints
    if logs:
        last = logs[-1]
        exe_result = last.get("exe_result")
        if exe_result and "success" not in str(exe_result).lower():
            return str(exe_result)
    # Fallbacks
    if logs and logs[-1].get("type") != "done":
        return "no_done_action (likely max_steps or interrupt)"
    return "no_explicit_error_in_log"


def load_run(run_path):
    data = {}
    for path in run_path.rglob("execution_log.json"):
        try:
            obj = json.loads(path.read_text())
        except Exception:
            continue
        task = obj.get("task_config", {})
        example_id = task.get("id") or path.parent.name
        domain = path.parent.parent.name
        instruction = task.get("instruction")
        score = obj.get("statistics", {}).get("score")
        failure_reason = ""
        if score == 0 or score == 0.0:
            failure_reason = f"score={score}; {extract_failure_reason(obj)}"
        data[example_id] = {
            "domain": domain,
            "instruction": instruction,
            "score": score,
            "failure_reason": failure_reason,
        }
    return data


def main():
    run_data = {name: load_run(path) for name, path in RUNS.items()}

    example_ids = set()
    for data in run_data.values():
        example_ids.update(data.keys())

    rows = []
    for example_id in sorted(example_ids):
        base = (
            run_data["autoglm-os_baseline"].get(example_id)
            or run_data["autoglm-os_gta1_7b_restart"].get(example_id)
            or {}
        )
        row = {
            "domain": base.get("domain"),
            "example_id": example_id,
            "instruction": base.get("instruction"),
        }
        for run_name in RUNS.keys():
            entry = run_data[run_name].get(example_id)
            if entry is None:
                row[f"{run_name}_score"] = ""
                row[f"{run_name}_failure_reason"] = "missing_execution_log"
            else:
                row[f"{run_name}_score"] = entry["score"]
                row[f"{run_name}_failure_reason"] = entry["failure_reason"]
        rows.append(row)

    df = pd.DataFrame(rows)
    cols = [
        "domain",
        "example_id",
        "instruction",
        "autoglm-os_baseline_score",
        "autoglm-os_baseline_failure_reason",
        "autoglm-os_gta1_7b_restart_score",
        "autoglm-os_gta1_7b_restart_failure_reason",
    ]
    df = df[cols].sort_values(["domain", "example_id"], ascending=[True, True])

    out_path = RESULTS_ROOT / "task_success_failures.xlsx"
    df.to_excel(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")


if __name__ == "__main__":
    main()
