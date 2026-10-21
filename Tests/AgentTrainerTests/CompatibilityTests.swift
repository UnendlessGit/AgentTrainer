import XCTest
@testable import AgentTrainer

final class CompatibilityTests: XCTestCase {
    private func adding(_ values: [String: Any], to value: some Encodable) throws -> Data {
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(value)) as? [String: Any])
        object.merge(values) { _, new in new }
        return try JSONSerialization.data(withJSONObject: object)
    }

    func testRetiredProfileSettingsAreIgnoredEvenWhenMalformed() throws {
        let profile = AIProfile.fresh()
        var object = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(profile)) as? [String: Any])
        var training = try XCTUnwrap(object["training"] as? [String: Any])
        training["dagger"] = ["correctionWeight": "invalid", "preRollSeconds": -100]
        object["training"] = training
        object["reinforcementLearning"] = ["algorithm": "unknown", "actionSpace": false]
        let decoded = try JSONDecoder().decode(AIProfile.self, from: JSONSerialization.data(withJSONObject: object))
        XCTAssertEqual(decoded, profile)
        let saved = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        XCTAssertNil(saved["reinforcementLearning"])
        XCTAssertNil((saved["training"] as? [String: Any])?["dagger"])
    }

    func testRetiredHotkeyAndPresetSettingsDoNotResetCurrentValues() throws {
        let settings = HotkeySettings(panic: .mouse(4), record: .mouse(5), run: .mouse(6))
        let data = try adding(["reinforcementLearning": "invalid"], to: settings)
        XCTAssertEqual(try JSONDecoder().decode(HotkeySettings.self, from: data), settings)
        let preset = RecordingPreset(id: UUID(), name: "Existing", createdAt: Date(), captureKind: .display,
            captureFPS: 60, showsCursor: true, region: CodableRect(.zero),
            trimStart: 0.5, trimEnd: 0.5, excludedKeyCodes: [49])
        let decoded = try JSONDecoder().decode(RecordingPreset.self,
            from: adding(["leadUpSeconds": ["invalid": true]], to: preset))
        XCTAssertEqual(try decoded.validated(), preset)
    }

    func testRetiredWorkflowTransportIsIgnoredWithoutLosingOtherSettings() throws {
        let data = Data(#"{"recordingExcludedKeyCodes":[13],"recordingTrimStart":0.25,"recordingTrimEnd":0.5,"trainingRunSettings":{"maximumSteps":42,"autosaveSteps":10},"runMouseMode":"Game Camera","recordingLeadUpSeconds":"invalid","rewardWebSocketSettings":{"url":false}}"#.utf8)
        let decoded = try JSONDecoder().decode(PersistentWorkflowSettings.self, from: data)
        XCTAssertEqual(decoded.recordingExcludedKeyCodes, [13])
        XCTAssertEqual(decoded.recordingTrimStart, 0.25)
        XCTAssertEqual(decoded.trainingRunSettings.maximumSteps, 42)
        let saved = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        XCTAssertNil(saved["recordingLeadUpSeconds"])
        XCTAssertNil(saved["rewardWebSocketSettings"])
    }

    func testLegacyRecordingMetadataRetainsOnlyItsSupervisionBoundary() throws {
        let manifest = RecordingManifest(id: UUID(), name: "Existing", createdAt: Date(), hostStartNanos: 1,
            duration: 4, capture: CaptureSpec(), globalRect: CodableRect(.zero), pixelWidth: 16,
            pixelHeight: 16, deliveredFPS: 30, eventCount: 0, supervisionStart: 2)
        let decoded = try JSONDecoder().decode(RecordingManifest.self, from: adding([
            "purpose": "daggerCorrection", "sourceProfileID": UUID().uuidString,
            "sourceVersionID": UUID().uuidString, "trainingWeight": "invalid"
        ], to: manifest))
        XCTAssertTrue(decoded.isStructurallyValid)
        XCTAssertEqual(decoded.effectiveSupervisionStart, 2)
        let saved = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(decoded)) as? [String: Any])
        XCTAssertNil(saved["purpose"])
        XCTAssertNil(saved["trainingWeight"])
        var invalid = decoded
        invalid.supervisionStart = 5
        XCTAssertFalse(invalid.isStructurallyValid)
    }

    func testLegacyActorLimitsSurviveRoundTripAndRemainDisableOnly() throws {
        let profile = AIProfile.fresh()
        let version = ModelVersionManifest(id: UUID(), name: "Existing", createdAt: Date(), globalStep: 3,
            trainingLoss: 0.25, preprocessing: profile.preprocessing, channels: profile.channels,
            training: profile.training, demonstratedKeyCodes: [13, 56], relativeMouseScale: GameCameraContract.deltaScale)
        let limits = ModelOutputCapabilities(allowedKeyCodes: [13, 56], allowedMouseButtons: [2],
            allowsAbsoluteMouseMovement: false, allowsRelativeMouseMovement: true, allowsScroll: false)
        let legacyLimits = try JSONSerialization.jsonObject(with: JSONEncoder().encode(limits))
        let decoded = try JSONDecoder().decode(ModelVersionManifest.self, from: adding([
            "trainingOrigin": "reinforcementLearning", "reinforcementActionSpace": legacyLimits,
            "reinforcementLearningSummary": "obsolete", "reinforcementActionBehavior": false
        ], to: version))
        XCTAssertEqual(decoded.outputCapabilities, limits)
        let saved = try JSONEncoder().encode(decoded)
        XCTAssertEqual(try JSONDecoder().decode(ModelVersionManifest.self, from: saved), decoded)
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: saved) as? [String: Any])
        XCTAssertNil(object["reinforcementActionSpace"])
        XCTAssertNotNil(object["outputCapabilities"])

        var current = ActionChannels.all
        current.keyboard = false
        let channels = limits.restricting(current, mouseMode: .relative)
        XCTAssertFalse(channels.keyboard)
        XCTAssertTrue(channels.mouseMovement)
        XCTAssertFalse(channels.scroll)
        XCTAssertFalse(limits.restricting(.all, mouseMode: .absolute).mouseMovement)
        let restrictions = limits.restricting(ActionRestrictions(blockedMouseButtons: [2]))
        XCTAssertTrue((0..<8).allSatisfy { !restrictions.allowsButton(UInt8($0)) })
        var prediction = [Float](repeating: 1, count: ActionLayout.count)
        prediction = RuntimeActionSemantics.temporalControlValues(prediction, channels: .all,
            restrictions: limits.restricting(ActionRestrictions()), allowedKeyCodes: limits.allowedKeyCodes)
        XCTAssertEqual(prediction[ActionLayout.keyboard.lowerBound + 13], 1)
        XCTAssertEqual(prediction[ActionLayout.keyboard.lowerBound + 12], 0)
        XCTAssertEqual(prediction[ActionLayout.shift.lowerBound], 1)
        XCTAssertEqual(prediction[ActionLayout.buttons.lowerBound + 2], 1)
        XCTAssertEqual(prediction[ActionLayout.buttons.lowerBound], 0)
    }

    func testLegacyActorWithoutLimitsFailsClosed() throws {
        let profile = AIProfile.fresh()
        let version = ModelVersionManifest(id: UUID(), name: "Existing", createdAt: Date(), globalStep: 0,
            trainingLoss: 0, preprocessing: profile.preprocessing, channels: profile.channels, training: profile.training)
        let decoded = try JSONDecoder().decode(ModelVersionManifest.self,
            from: adding(["trainingOrigin": "reinforcementLearning"], to: version))
        let limits = try XCTUnwrap(decoded.outputCapabilities)
        let channels = limits.restricting(.all, mouseMode: .absolute)
        XCTAssertFalse(channels.mouseMovement || channels.buttons || channels.keyboard || channels.modifiers || channels.scroll)
        let invalid = ModelOutputCapabilities(allowedKeyCodes: [128])
        let invalidObject = try JSONSerialization.jsonObject(with: JSONEncoder().encode(invalid))
        XCTAssertThrowsError(try JSONDecoder().decode(ModelVersionManifest.self,
            from: adding(["outputCapabilities": invalidObject], to: version)))
    }
}
