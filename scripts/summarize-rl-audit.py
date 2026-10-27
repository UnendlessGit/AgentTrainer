#!/usr/bin/env python3
"""Summarize fixture-derived evidence; never infer learning from counters."""
import json
from pathlib import Path
import statistics
import hashlib
import struct
import math
import re

repo = Path(__file__).resolve().parents[1]
root = repo / "work/rl-validation/audit"
logs = [json.loads(line) for line in (root / "fixture.jsonl").read_text().splitlines()]

def tensor_hash(path):
    data = path.read_bytes()
    length = struct.unpack("<Q", data[:8])[0]
    header = json.loads(data[8:8 + length])
    digest = hashlib.sha256()
    for name in sorted(k for k in header if k != "__metadata__"):
        entry = header[name]
        digest.update(json.dumps([name, entry["dtype"], entry["shape"]], separators=(",", ":")).encode())
        begin, end = entry["data_offsets"]
        digest.update(data[8 + length + begin:8 + length + end])
    return digest.hexdigest()

def outcomes(label, training=False):
    # A label can have multiple episodes, but only independently received
    # terminal events establish complete evaluation outcomes.
    matching = [row for row in logs if row["label"] == label]
    if training:
        # Technical retries can reuse the first baseline label. Select its
        # latest complete launch, identified by the reset to episode one.
        starts = [i for i, row in enumerate(matching) if row["event"] == "reset" and row["episode"] == 1]
        matching = matching[starts[-1]:] if starts else []
    ends = [row for row in matching if row["event"] == "terminal"]
    if not training:
        ends = ends[-1:]
    counts = {}
    for row in ends:
        for key, value in row["counts"].items():
            counts[key] = counts.get(key, 0) + value
    result = {"episodes": len(ends), "counts": counts,
              "rewards": [row["reward"] for row in ends]}
    if training:
        result["interruptedEpisodes"] = [{"episode": row["episode"], "counts": row["counts"], "reward": row["reward"]}
                                         for row in matching if row["event"] == "interrupted"]
    for kind in range(4):
        n = counts.get(f"seconds{kind}", 0)
        if n:
            result[f"accuracy{kind}"] = counts.get(f"correctSeconds{kind}", 0) / n
        n = counts.get(f"trials{kind}", 0)
        if n:
            result[f"success{kind}"] = counts.get(f"successes{kind}", 0) / n
        n = counts.get(f"errorSamples{kind}", 0)
        if n:
            result[f"error{kind}"] = counts.get(f"errorSum{kind}", 0) / n
    if counts.get("keyDown49"):
        result["spaceEdgePrecision"] = counts.get("spaceEdges", 0) / counts["keyDown49"]
    return result

def arm(name):
    path = root / f"{name}.json"
    if not path.exists():
        return []
    report = json.loads(path.read_text())
    if report.get("error") and not report.get("results"):
        raise SystemExit(report["error"])
    result = []
    for cell in report["results"]:
        item = {k: cell[k] for k in ["label", "cell", "initialHash", "finalHash", "trainingSeconds", "optimizerSteps", "endedEarly", "status"]}
        initial = root / cell["label"] / "initial"
        item["initialTensorSHA256"] = tensor_hash(initial / "weights.safetensors")
        item["initialHeadsTensorSHA256"] = tensor_hash(initial / "heads.safetensors")
        for key in ["trainingInitialDelta", "policyDelta", "headsDelta"]:
            item[key] = {k: cell[key][k] for k in ["changed", "l2", "max"]}
        item["encoderDeltaL2"] = sum(t["l2"] ** 2 for t in cell["policyDelta"]["tensors"] if t["name"].startswith("convolutions.")) ** 0.5
        actor_prefixes = ("keyboardHead.", "buttonHead.", "modifierHead.", "absoluteMouseHead.", "relativeMouseHead.", "movementIntentHead.", "scrollHead.")
        item["actorOutputDeltaL2"] = sum(t["l2"] ** 2 for t in cell["policyDelta"]["tensors"] if t["name"].startswith(actor_prefixes)) ** 0.5
        item["criticDeltaL2"] = sum(t["l2"] ** 2 for t in cell["headsDelta"]["tensors"] if t["name"].startswith("critic.")) ** 0.5
        item["explorationDeltaL2"] = sum(t["l2"] ** 2 for t in cell["headsDelta"]["tensors"] if "StandardDeviation" in t["name"]) ** 0.5
        item["trainingOutcomes"] = outcomes(cell["label"] + "-train", training=True)
        item["episodeHistory"] = cell["episodes"]
        item["evaluations"] = {}
        for checkpoint in ["initial", "final"]:
            for mode in ["deterministic", "stochastic"]:
                item["evaluations"][checkpoint + "-" + mode] = outcomes(cell["label"] + "-" + checkpoint + "-" + mode)
            evaluation = cell.get(checkpoint + "Evaluation", {})
            item["evaluations"][checkpoint + "-integrity"] = {
                "performed": bool(evaluation),
                "unchanged": bool(evaluation) and evaluation.get("hashBefore") == evaluation.get("hashAfter"),
                "optimizerInvocations": evaluation.get("optimizerInvocations"),
            }
        samples = cell["samples"][5:] or cell["samples"]
        item["performance"] = {key: statistics.mean(s[key] for s in samples) for key in
            ["actionFPS", "perceptionFPS", "rewardFPS", "headMS", "visionMS", "observationAgeMS", "activeBytes", "cacheBytes"]}
        item["performance"]["discardedActions"] = samples[-1]["discarded"]
        item["performance"]["finalKL"] = samples[-1]["kl"]
        item["performance"]["peakActiveBytes"] = max(s["activeBytes"] for s in cell["samples"])
        item["performance"]["peakCacheBytes"] = max(s["cacheBytes"] for s in cell["samples"])
        item["performance"]["meanReportedHeadP95MS"] = statistics.mean(s["headP95MS"] for s in samples)
        item["performance"]["meanReportedVisionP95MS"] = statistics.mean(s["visionP95MS"] for s in samples)
        workspace = root / (name + "-workspace") / "Profiles"
        for state in workspace.glob("*/Versions/*/state.json"):
            policy = state.parent / "weights.safetensors"
            if policy.exists() and hashlib.sha256(policy.read_bytes()).hexdigest() == cell["finalHash"]:
                saved = json.loads(state.read_text())
                item["trainingState"] = {k: saved.get(k) for k in ["learningSemanticsVersion", "revision", "updates", "optimizerSteps", "attemptedUpdates", "rejectedUpdates", "episodeCount"]}
                config = dict(saved["configuration"])
                config["allowedKeys"] = sorted(config["allowedKeys"])
                config["allowedButtons"] = sorted(config["allowedButtons"])
                config["rewardRegion"].pop("sourceID", None)
                item["configuration"] = config
                item["effectiveChannels"] = saved.get("channels")
                item["admittedActionTotal"] = None  # Not instrumented in the original common harness.
                break
        result.append(item)
    return result

baseline, improved = arm("baseline"), arm("improved") + arm("improved-rest")
by_key = {(x["cell"]["task"], x["cell"]["seed"]): x for x in baseline}
pairs = []
for after in improved:
    key = (after["cell"]["task"], after["cell"]["seed"])
    before = by_key[key]
    pair = {"task": key[0], "seed": key[1],
            "identicalInitialPolicy": before["initialTensorSHA256"] == after["initialTensorSHA256"],
            "identicalInitialHeads": before["initialHeadsTensorSHA256"] == after["initialHeadsTensorSHA256"], "deltas": {}}
    for mode in ["deterministic", "stochastic"]:
        original = before["evaluations"]["final-" + mode]
        changed = after["evaluations"]["final-" + mode]
        pair["deltas"][mode] = {k: changed[k] - original[k] for k in original
            if k in changed and k.startswith(("accuracy", "success", "error", "spaceEdgePrecision"))}
    pairs.append(pair)
summary = {
    "baselineCommit": "5da8c0ec343f119ee0c1124bfd3032c9bd037740",
    "commonInstrumentationCommit": "164e589",
    "interpretation": "Paired descriptive engineering trials. Sparse controls and short sweeps do not establish mastery or statistical significance. Error reductions are favorable; accuracy/success increases are favorable.",
    "liveBudget": json.loads((root / "live-budget.json").read_text()),
    "baseline": baseline, "improved": improved, "pairs": pairs,
    "configurationFollowup": {"development": arm("mouse-development"), "confirmation": arm("mouse-confirmation")},
    "screening": arm("screening-fixed") or arm("screening"),
    "technicalRetests": [
        {"attempt": "failed-baseline", "reason": "Native integer-zero OCR failure; fixture changed to decimal length before both matched arms."},
        {"attempt": "improved", "reason": "Human-input safety stop during final frozen pointer-seed-7 evaluation; retained the three completed keyboard cells and retested the unfinished pointer cell in improved-rest."},
        {"attempt": "screening", "reason": "Low Action FPS was blocked by demonstration-only auxiliary horizon validation; fixed the RL startup/profile validation path and restarted the screen."},
    ],
    "provenance": [json.loads(path.read_text()) for name in ["baseline", "improved", "improved-rest"]
                   if (path := root / f"{name}-provenance.json").exists()],
}
summary["metricDefinitions"] = {
    "keyboardKinds": {"0": "exclusive A occupancy", "1": "exclusive D occupancy", "2": "fresh Space edges and exclusive occupancy", "3": "exclusive S hold occupancy"},
    "pointerKinds": {"0": "absolute pointing plus left click", "1": "absolute pointing plus right click", "2": "drag; original learning fixture has a known invalid-release/pickup-reward defect"},
    "relativeKinds": {"0": "relative motion", "1": "horizontal scroll", "2": "vertical scroll"},
    "accuracy": "correct cue seconds / observed cue seconds; a successful keyboard trial only requires some correct occupancy, not a complete sustained hold",
    "error": "mean Euclidean error in normalized canvas coordinates, separately for each cued phase",
    "success": "successful trials / exposures; absent exposure is missing, not zero success",
    "spaceEdgePrecision": "rewarded fresh Space edges / all received Space key-down edges",
    "performance": "means of one-second dashboard snapshots after the first five; reported p95 means are not whole-run latency percentiles",
    "trainingOutcomes": "independent fixture counters summed over complete training episodes; interrupted episode counters are separate",
}
summary["limitations"] = [
    "These short live trials do not demonstrate broad learning superiority, faster learning, or reliable conditional mastery of keyboard, click, drag, and scroll controls.",
    "Original keyboard prelude discarded key-down edges before the first cue. Corrected-fixture frozen re-evaluations are reported separately and do not replace the primary measurements.",
    "The original drag fixture allowed repeat pickup reward and retained picked state across invalid releases. Drag learning scores cannot establish valid held dragging. The corrected contract was checked using scripted HID, not a learned policy.",
    "The common primary harness did not record a cumulative admitted-action total or separate packing/transfer CPU timers. Optimizer state, independent received-event exposures, exclusions and inference/vision latency were recorded. No kernel or snapshot-transfer speedup is claimed.",
    "Actual wall-clock scheduling and OCR timing are not deterministic even with identical initial tensors and seeded fixture schedules.",
]
summary["manifests"] = []
for name in ["baseline", "improved", "improved-rest", "mouse-development", "screening-fixed", "held-key-baseline", "held-key-improved"]:
    path = root / f"{name}-manifest.json"
    if path.exists():
        manifest = json.loads(path.read_text())
        manifest.pop("root", None)
        summary["manifests"].append(manifest)
summary["correctedPreludeFrozenEvaluation"] = []
for name in ["held-key-baseline", "held-key-improved"]:
    for row in json.loads((root / f"{name}.json").read_text())["results"]:
        result = {"implementation": name, "task": row["task"], "seed": row["seed"], "sourceProfile": row["profile"]}
        for checkpoint in ["initial", "final"]:
            result[checkpoint] = dict(row[checkpoint])
            for mode in ["deterministic", "stochastic"]:
                result[checkpoint][mode] = outcomes(f"{name}-{row['task']}-{row['seed']}-{checkpoint}-{mode}")
        summary["correctedPreludeFrozenEvaluation"].append(result)
summary["deploymentChecks"] = {name: json.loads((root / f"{name}.json").read_text())
                                for name in ["smoke-keyboard", "smoke-mouse", "smoke-keyboard-final", "fixture-contract", "quit-evaluation"]}
summary["deploymentChecks"]["smoke-keyboard"]["measurementNote"] = "Immediate stepsAtResume=0 was a stale UI snapshot. The later check waits for resumed metrics and verifies retained/restored progress."
summary["resources"] = {}
for path in root.glob("*-resources.json"):
    rows = json.loads(path.read_text())
    if rows:
        summary["resources"][path.stem.removesuffix("-resources")] = {
            "samples": len(rows), "peakResidentBytes": max(row["residentBytes"] for row in rows),
            "meanCPUPercent": statistics.mean(row["cpuPercent"] for row in rows),
            "note": "External ps samples at approximately one-second intervals; includes evaluation/startup and all cells in the named process.",
        }
summary["lifecycleChecks"] = {}
for name in ["lifecycle-timeout", "lifecycle-missing"]:
    row = json.loads((root / f"{name}.json").read_text())
    fixture = [json.loads(line) for line in (root / f"{name}-fixture.jsonl").read_text().splitlines()]
    summary["lifecycleChecks"][name] = {k: row[k] for k in ["status", "error", "updates", "episodes"]}
    summary["lifecycleChecks"][name]["receivedEndKeys"] = [r for r in fixture if r.get("event") == "controllerEnd"]
summary["groupingAnalysis"] = {
    "note": "Analytical eight-unit-reward stream with discount 0.9 and lambda 1; an explanatory stress case, not live task results. Cumulative OCR does not reveal within-interval reward times.",
    "cells": [{"decisionsPerInterval": group,
               "firstReturn": group * (1 - 0.9 ** 8) / (1 - 0.9 ** group),
               "firstReturnWithSteadyIntervalRMS": (1 - 0.9 ** 8) / (1 - 0.9 ** group),
               "jointRatioForPerDecisionLogRatioPointOne": math.exp(group * 0.1),
               "jointSampleKL": math.exp(group * 0.1) - 1 - group * 0.1}
              for group in [1, 2, 4, 8]],
}
controlled = root / "controlled-final.json"
if controlled.exists():
    summary["controlledLearning"] = json.loads(controlled.read_text())
test_log = root / "tests-final.log"
if test_log.exists():
    matches = re.findall(r"Executed (\d+) tests, with (\d+) tests skipped and (\d+) failures .*? in ([\d.]+) \(([\d.]+)\) seconds", test_log.read_text())
    if matches:
        count, skipped, failures, _, elapsed = matches[-1]
        summary["regression"] = {"tests": int(count), "skipped": int(skipped), "failures": int(failures),
                                 "seconds": float(elapsed), "controlledLearningEnabled": True}
summary["liveSeconds"] = sum(row["seconds"] for row in summary["liveBudget"])
assert summary["liveSeconds"] <= 3600, "Live budget exceeded"
assert len(pairs) == 9 and all(p["identicalInitialPolicy"] and p["identicalInitialHeads"] for p in pairs)
assert all(x["actorOutputDeltaL2"] > 0 and math.isfinite(x["actorOutputDeltaL2"]) for x in baseline + improved)
destination = repo / "docs/validation-2.61.json"
destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("Live seconds:", round(summary["liveSeconds"], 1))
for pair in pairs:
    print(pair["task"], pair["seed"], "identical init", pair["identicalInitialPolicy"], pair["deltas"])
