#!/usr/bin/env python3
"""Summarize fixture-derived evidence; never infer learning from counters."""
import json
from pathlib import Path
import statistics
import hashlib
import struct

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

def outcomes(label):
    # A label can have multiple episodes, but only independently received
    # terminal events establish complete evaluation outcomes.
    ends = [row for row in logs if row["label"] == label and row["event"] == "terminal"][-1:]
    counts = {}
    for row in ends:
        for key, value in row["counts"].items():
            counts[key] = counts.get(key, 0) + value
    result = {"episodes": len(ends), "counts": counts}
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
        item["evaluations"] = {}
        for checkpoint in ["initial", "final"]:
            for mode in ["deterministic", "stochastic"]:
                item["evaluations"][checkpoint + "-" + mode] = outcomes(cell["label"] + "-" + checkpoint + "-" + mode)
            evaluation = cell.get(checkpoint + "Evaluation", {})
            item["evaluations"][checkpoint + "-integrity"] = {
                "unchanged": evaluation.get("hashBefore") == evaluation.get("hashAfter"),
                "optimizerInvocations": evaluation.get("optimizerInvocations"),
            }
        samples = cell["samples"][5:] or cell["samples"]
        item["performance"] = {key: statistics.mean(s[key] for s in samples) for key in
            ["actionFPS", "perceptionFPS", "rewardFPS", "headMS", "visionMS", "observationAgeMS", "activeBytes", "cacheBytes"]}
        item["performance"]["discardedActions"] = samples[-1]["discarded"]
        item["performance"]["finalKL"] = samples[-1]["kl"]
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
    "technicalRetests": [
        {"attempt": "failed-baseline", "reason": "Native integer-zero OCR failure; fixture changed to decimal length before both matched arms."},
        {"attempt": "improved", "reason": "Human-input safety stop during final frozen pointer-seed-7 evaluation; retained the three completed keyboard cells and retested the unfinished pointer cell in improved-rest."},
    ],
    "provenance": [json.loads(path.read_text()) for name in ["baseline", "improved", "improved-rest"]
                   if (path := root / f"{name}-provenance.json").exists()],
}
summary["liveSeconds"] = sum(row["seconds"] for row in summary["liveBudget"])
destination = repo / "docs/validation-2.61.json"
destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("Live seconds:", round(summary["liveSeconds"], 1))
for pair in pairs:
    print(pair["task"], pair["seed"], "identical init", pair["identicalInitialPolicy"], pair["deltas"])
