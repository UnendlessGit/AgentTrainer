import Foundation

struct BinaryDecisionObservation: Sendable {
    var score: Float
    var target: Bool
}

struct BinaryDecisionCalibrationResult: Equatable, Sendable {
    var threshold: Float
    var matthewsCorrelation: Double
    var defaultMatthewsCorrelation: Double
    var bestObservedThreshold: Float
    var bestObservedMatthewsCorrelation: Double
    var positiveSupport: Int
    var negativeSupport: Int
    var usedCalibratedThreshold: Bool

    var disabledForSafety: Bool {
        threshold >= BinaryDecisionContract.disabledThreshold
    }
}

/// Selects an executable boundary without undoing probability calibration in
/// the loss. Matthews correlation rewards both true presses and true releases,
/// so an always-held or always-off result cannot win merely because a control
/// is common or rare. The sweep is O(n log n) and runs only at validation time.
enum BinaryDecisionCalibration {
    private struct Confusion {
        var truePositives = 0
        var falsePositives = 0
        var falseNegatives = 0
        var trueNegatives = 0
    }

    static var defaultThresholds: [Float] {
        [Float](
            repeating: BinaryDecisionContract.defaultThreshold,
            count: ActionLayout.count
        )
    }

    static func normalized(_ stored: [Float]?) -> [Float] {
        guard let stored, stored.count == ActionLayout.count else {
            return defaultThresholds
        }
        var result = defaultThresholds
        for index in ActionLayout.binary where stored[index].isFinite {
            result[index] = min(1, max(0, stored[index]))
        }
        return result
    }

    static func threshold(
        for index: Int,
        in thresholds: [Float]?
    ) -> Float {
        guard let thresholds,
              thresholds.indices.contains(index),
              thresholds[index].isFinite else {
            return BinaryDecisionContract.defaultThreshold
        }
        return min(1, max(0, thresholds[index]))
    }

    static func isActive(
        score: Float,
        for index: Int,
        in thresholds: [Float]?
    ) -> Bool {
        isActive(
            score: score,
            threshold: threshold(for: index, in: thresholds)
        )
    }

    static func isActive(score: Float, threshold: Float) -> Bool {
        guard score.isFinite, threshold.isFinite else { return false }
        let boundary = min(1, max(0, threshold))
        return boundary < BinaryDecisionContract.disabledThreshold
            && score >= boundary
    }

    static func calibrate(
        _ observations: [BinaryDecisionObservation]
    ) -> BinaryDecisionCalibrationResult {
        let finite = observations.compactMap { observation -> BinaryDecisionObservation? in
            guard observation.score.isFinite else { return nil }
            return BinaryDecisionObservation(
                score: min(1, max(0, observation.score)),
                target: observation.target
            )
        }
        let positives = finite.count(where: \.target)
        let negatives = finite.count - positives
        let fallback = confusion(
            finite,
            threshold: BinaryDecisionContract.defaultThreshold
        )
        let defaultCorrelation = matthews(fallback)
        let defaultRecall = recall(fallback)
        let defaultPrecision = precision(fallback)
        let defaultFalsePositiveRate = falsePositiveRate(fallback)
        let defaultIsSafe = defaultCorrelation
                >= BinaryDecisionContract.minimumUsefulMatthewsCorrelation
            && defaultRecall >= BinaryDecisionContract.minimumUsefulRecall
            && defaultPrecision >= BinaryDecisionContract.minimumUsefulPrecision
            && defaultFalsePositiveRate
                <= BinaryDecisionContract.maximumUsefulFalsePositiveRate
        let hasSupport = positives
                >= BinaryDecisionContract.minimumPositiveSupport
            && negatives >= BinaryDecisionContract.minimumNegativeSupport
        guard hasSupport else {
            let threshold = BinaryDecisionContract.disabledThreshold
            let selected = confusion(finite, threshold: threshold)
            return BinaryDecisionCalibrationResult(
                threshold: threshold,
                matthewsCorrelation: matthews(selected),
                defaultMatthewsCorrelation: defaultCorrelation,
                bestObservedThreshold: BinaryDecisionContract.defaultThreshold,
                bestObservedMatthewsCorrelation: defaultCorrelation,
                positiveSupport: positives,
                negativeSupport: negatives,
                usedCalibratedThreshold: threshold
                    != BinaryDecisionContract.defaultThreshold
            )
        }

        let sorted = finite.sorted {
            if $0.score != $1.score { return $0.score > $1.score }
            return $0.target && !$1.target
        }
        var running = Confusion(
            truePositives: 0,
            falsePositives: 0,
            falseNegatives: positives,
            trueNegatives: negatives
        )
        var bestThreshold: Float?
        var bestCorrelation = -Double.infinity
        var offset = 0
        while offset < sorted.count {
            let score = sorted[offset].score
            var end = offset
            while end < sorted.count, sorted[end].score == score {
                if sorted[end].target {
                    running.truePositives += 1
                    running.falseNegatives -= 1
                } else {
                    running.falsePositives += 1
                    running.trueNegatives -= 1
                }
                end += 1
            }
            let threshold: Float
            if end < sorted.count {
                threshold = score + (sorted[end].score - score) / 2
            } else {
                threshold = score
            }
            guard threshold < BinaryDecisionContract.disabledThreshold else {
                offset = end
                continue
            }
            let correlation = matthews(running)
            guard recall(running) >= BinaryDecisionContract.minimumUsefulRecall,
                  precision(running) >= BinaryDecisionContract.minimumUsefulPrecision,
                  falsePositiveRate(running)
                    <= BinaryDecisionContract.maximumUsefulFalsePositiveRate else {
                offset = end
                continue
            }
            let improves = correlation > bestCorrelation + 1e-12
            let ties = abs(correlation - bestCorrelation) <= 1e-12
            let isSaferTie = bestThreshold.map { threshold > $0 } ?? true
            if improves || (ties && isSaferTie) {
                bestThreshold = threshold
                bestCorrelation = correlation
            }
            offset = end
        }

        let useful = bestThreshold != nil
            && bestCorrelation
                >= BinaryDecisionContract.minimumUsefulMatthewsCorrelation
            && (
                !defaultIsSafe
                    || bestCorrelation - defaultCorrelation
                        >= BinaryDecisionContract.minimumMatthewsImprovement
            )
        let observedThreshold = bestThreshold
            ?? BinaryDecisionContract.defaultThreshold
        let observedCorrelation = bestThreshold == nil
            ? defaultCorrelation
            : bestCorrelation
        let selectedThreshold: Float
        if useful {
            selectedThreshold = observedThreshold
        } else if defaultIsSafe {
            selectedThreshold = BinaryDecisionContract.defaultThreshold
        } else {
            selectedThreshold = BinaryDecisionContract.disabledThreshold
        }
        let selectedCorrelation = selectedThreshold
            == BinaryDecisionContract.disabledThreshold
            ? matthews(confusion(finite, threshold: selectedThreshold))
            : useful ? bestCorrelation : defaultCorrelation
        return BinaryDecisionCalibrationResult(
            threshold: selectedThreshold,
            matthewsCorrelation: selectedCorrelation,
            defaultMatthewsCorrelation: defaultCorrelation,
            bestObservedThreshold: observedThreshold,
            bestObservedMatthewsCorrelation: observedCorrelation,
            positiveSupport: positives,
            negativeSupport: negatives,
            usedCalibratedThreshold: selectedThreshold
                != BinaryDecisionContract.defaultThreshold
        )
    }

    private static func confusion(
        _ observations: [BinaryDecisionObservation],
        threshold: Float
    ) -> Confusion {
        observations.reduce(into: Confusion()) { result, observation in
            switch (isActive(score: observation.score, threshold: threshold), observation.target) {
            case (true, true): result.truePositives += 1
            case (true, false): result.falsePositives += 1
            case (false, true): result.falseNegatives += 1
            case (false, false): result.trueNegatives += 1
            }
        }
    }

    private static func matthews(_ value: Confusion) -> Double {
        let truePositives = Double(value.truePositives)
        let falsePositives = Double(value.falsePositives)
        let falseNegatives = Double(value.falseNegatives)
        let trueNegatives = Double(value.trueNegatives)
        let denominator = (
            (truePositives + falsePositives)
                * (truePositives + falseNegatives)
                * (trueNegatives + falsePositives)
                * (trueNegatives + falseNegatives)
        ).squareRoot()
        guard denominator > 0 else { return 0 }
        return (truePositives * trueNegatives - falsePositives * falseNegatives)
            / denominator
    }

    private static func recall(_ value: Confusion) -> Double {
        Double(value.truePositives)
            / Double(max(1, value.truePositives + value.falseNegatives))
    }

    private static func precision(_ value: Confusion) -> Double {
        Double(value.truePositives)
            / Double(max(1, value.truePositives + value.falsePositives))
    }

    private static func falsePositiveRate(_ value: Confusion) -> Double {
        Double(value.falsePositives)
            / Double(max(1, value.falsePositives + value.trueNegatives))
    }
}
