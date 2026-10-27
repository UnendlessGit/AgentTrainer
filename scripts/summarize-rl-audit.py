#!/usr/bin/env python3
"""Summarize fixture-derived evidence; never infer learning from counters."""
import json
from pathlib import Path
import statistics

repo = Path(__file__).resolve().parents[1]
root = repo / "work/rl-validation/audit"
logs = [json.loads(line) for line in (root / "fixture.jsonl").read_text().splitlines()]

def outcomes(label):
    # A label can have multiple episodes, but only independently received
    # terminal events establish complete evaluation outcomes.
    ends = [row for row in logs if row["label"] == label and row["event"] == "terminal"]
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
    if report.get("error"):
        raise SystemExit(report["error"])
    result = []
    for cell in report["results"]:
        item = {k: cell[k] for k in ["label", "cell", "initialHash", "finalHash", "trainingSeconds", "optimizerSteps", "endedEarly", "status"]}
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

baseline, improved = arm("baseline"), arm("improved")
by_key = {(x["cell"]["task"], x["cell"]["seed"]): x for x in baseline}
pairs = []
for after in improved:
    key = (after["cell"]["task"], after["cell"]["seed"])
    before = by_key[key]
    pair = {"task": key[0], "seed": key[1], "identicalInitialPolicy": before["initialHash"] == after["initialHash"], "deltas": {}}
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
}
summary["liveSeconds"] = sum(row["seconds"] for row in summary["liveBudget"])
destination = repo / "docs/validation-2.61.json"
destination.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
print("Live seconds:", round(summary["liveSeconds"], 1))
for pair in pairs:
    print(pair["task"], pair["seed"], "identical init", pair["identicalInitialPolicy"], pair["deltas"])
