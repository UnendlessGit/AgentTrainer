#!/usr/bin/env python3
"""Bounded live audit launcher. No game-state reads enter the learner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument("implementation", choices=["baseline", "improved", "canary", "screening", "smoke", "evaluate", "fixture"])
parser.add_argument("app", type=Path)
parser.add_argument("--seconds", type=float, default=90)
parser.add_argument("--seeds", default="7,42,101")
parser.add_argument("--tasks", default="keyboard,pointer,relative")
parser.add_argument("--evaluation-seconds", type=float, default=20)
parser.add_argument("--run-name", help="Unique label for a continuation or technical retry")
parser.add_argument("--source-workspace", default="improved", help="Existing audit workspace used by smoke tests")
parser.add_argument("--learning-rate", type=float, default=0.0003)
parser.add_argument("--exploration-scale", type=float, default=1)
parser.add_argument("--evaluation-seed", type=int, default=900001)
args = parser.parse_args()
run_name = args.run_name or args.implementation
if not run_name or any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in run_name):
    raise SystemExit("Run name must contain only ASCII letters, digits, '-' and '_'.")
if not args.source_workspace or any(not (c.isascii() and (c.isalnum() or c in "-_")) for c in args.source_workspace):
    raise SystemExit("Invalid isolated source workspace name.")
repo = Path(__file__).resolve().parents[1]
root = repo / "work/rl-validation/audit"
root.mkdir(parents=True, exist_ok=True)
if any((root / f"{run_name}{suffix}").exists() for suffix in [".json", "-manifest.json", ".log"]):
    raise SystemExit("This run name already has artifacts. Choose a new --run-name to preserve the evidence.")
if subprocess.run(["pgrep", "-x", "AgentTrainer"], stdout=subprocess.DEVNULL).returncode == 0:
    raise SystemExit("Quit the running AgentTrainer before starting an isolated audit process.")
ledger = root / "live-budget.json"
entries = json.loads(ledger.read_text()) if ledger.exists() else []
used = sum(row["seconds"] for row in entries)
remaining = 3600 - used
if remaining <= 0:
    raise SystemExit("The one-hour live budget is exhausted.")
base = dict(seconds=args.seconds, actionFPS=15, perceptionFPS=5, width=96,
            height=64, color=False, history=0, learningRate=args.learning_rate, explorationScale=args.exploration_scale)
cells = [dict(base, task=task, seed=int(seed)) for task in args.tasks.split(",")
         for seed in args.seeds.split(",")]
if args.implementation == "screening":
    changes = [{}, {"actionFPS": 5}, {"actionFPS": 60},
               {"actionFPS": 60, "perceptionFPS": 15},
               {"actionFPS": 60, "perceptionFPS": 60}, {"color": True},
               {"width": 320, "height": 180}, {"history": 4}, {"history": 8}]
    cells = [dict(base | change, task="keyboard", seed=7, seconds=20) for change in changes]
    args.evaluation_seconds = 0
manifest = dict(implementation=run_name, root=str(root), cells=cells,
                evaluationSeed=args.evaluation_seed,
                operation=args.implementation if args.implementation in ["smoke", "evaluate", "fixture"] else "train",
                evaluationSeconds=args.evaluation_seconds)
manifest_path = root / f"{run_name}-manifest.json"
manifest_path.write_text(json.dumps(manifest, indent=2))
binary = args.app.resolve() / "Contents/MacOS/AgentTrainer"
env = os.environ | {
    "AGENTTRAINER_RL_VALIDATION": "1",
    "AGENTTRAINER_WORKSPACE_ROOT": str(root / ((args.source_workspace if args.implementation in ["smoke", "evaluate"] else run_name) + "-workspace")),
    "AGENTTRAINER_RL_AUDIT_MANIFEST": str(manifest_path),
    "AGENTTRAINER_RL_VALIDATION_REPORT": str(root / f"{run_name}.json"),
}
metadata = {"implementation": run_name, "codeArm": args.implementation,
            "binarySHA256": hashlib.sha256(binary.read_bytes()).hexdigest(),
            "manifestSHA256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "fixtureSHA256": hashlib.sha256((repo / "Tests/Fixtures/RLAuditGame.swift").read_bytes()).hexdigest(),
            "commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
            "trackedDiffSHA256": hashlib.sha256(subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)).hexdigest()}
(root / f"{run_name}-provenance.json").write_text(json.dumps(metadata, indent=2))
began = time.monotonic()
process = None
resources = []
try:
    with (root / f"{run_name}.log").open("w") as output:
        process = subprocess.Popen([str(binary)], env=env, stdout=output, stderr=subprocess.STDOUT)
        deadline = began + max(1, remaining - 10)
        while process.poll() is None:
            if time.monotonic() >= deadline:
                # Normal application termination performs release/save cleanup,
                # including a currently running frozen evaluation.
                subprocess.run(["osascript", "-e", 'tell application id "local.agenttrainer.mac" to quit'], timeout=5)
                process.wait(timeout=8)
                raise SystemExit("Live budget reached; app asked to release/save and quit.")
            try:
                process.wait(timeout=min(1, max(0.01, deadline - time.monotonic())))
            except subprocess.TimeoutExpired:
                pass
            if process.poll() is None:
                try:
                    rss, cpu = subprocess.check_output(["ps", "-o", "rss=,%cpu=", "-p", str(process.pid)], text=True).split()
                    resources.append(dict(seconds=time.monotonic() - began, residentBytes=int(rss) * 1024, cpuPercent=float(cpu)))
                except (subprocess.CalledProcessError, ValueError):
                    pass
finally:
    elapsed = time.monotonic() - began
    entries.append(dict(implementation=run_name, seconds=elapsed))
    ledger.write_text(json.dumps(entries, indent=2))
    (root / f"{run_name}-resources.json").write_text(json.dumps(resources, indent=2))
    print(json.dumps(dict(implementation=run_name, seconds=elapsed,
                         totalSeconds=used + elapsed), indent=2), flush=True)
if process.returncode:
    raise SystemExit(f"Audit app exited {process.returncode}")
report = json.loads((root / f"{run_name}.json").read_text())
if report.get("error"):
    raise SystemExit(report["error"])
