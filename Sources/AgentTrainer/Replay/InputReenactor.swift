import AppKit
import CoreGraphics
import Foundation

final class InputReenactor: @unchecked Sendable {
    var onFinish: (@Sendable (String?) -> Void)?
    private let lock = NSLock()
    private var task: Task<Void, Never>?
    private var heldKeys: Set<UInt16> = []
    private var heldButtons: Set<UInt8> = []
    private var cursor = CGEvent(source: nil)?.location ?? .zero
    private var generation: UUID?
    private var gameCameraReplay = false
    private let hidEventSource = CGEventSource(stateID: .hidSystemState)

    func start(recording: RecordingItem, speed: Double = 1) throws {
        guard AXIsProcessTrusted() else { throw AgentTrainerError.permission("Accessibility permission is required for guarded reenactment.") }
        guard speed.isFinite, speed > 0 else { throw AgentTrainerError.invalidConfiguration("Reenactment speed must be finite and positive.") }
        let events = try InputEventReader.read(url: recording.directory.appendingPathComponent(recording.manifest.eventFile))
        guard !events.isEmpty else { throw AgentTrainerError.noData }
        stop()
        let token = UUID()
        let base = recording.manifest.hostStartNanos
        let trimStart = recording.manifest.trimStart
        let trimEnd = recording.manifest.trimEnd ?? recording.manifest.duration
        guard base > 0, trimStart.isFinite, trimEnd.isFinite,
              trimStart >= 0, trimEnd >= trimStart,
              trimStart * 1_000_000_000 < Double(UInt64.max - base) else {
            throw AgentTrainerError.storage("This recording has an invalid reenactment timeline.")
        }
        let selected = events.filter { event in guard event.timestampNanos >= base else { return false }; let t = Double(event.timestampNanos - base) / 1e9; return t >= trimStart && t <= trimEnd }
        let isGameCamera = InputEventReader.mouseDiagnostics(events: selected, globalRect: recording.manifest.globalRect.cgRect).isGameCamera
        let startNanos = base + UInt64(trimStart * 1_000_000_000)
        var initialKeys: Set<UInt16> = [], initialButtons: Set<UInt8> = []
        var initialModifiers: UInt64 = 0
        var initialPoint = CGPoint(x: recording.manifest.globalRect.cgRect.midX, y: recording.manifest.globalRect.cgRect.midY)
        for event in events where event.timestampNanos >= base && event.timestampNanos < startNanos {
            initialModifiers = event.modifiers
            switch event.kind {
            case .key:
                if event.isDown { initialKeys.insert(event.keyCode) } else { initialKeys.remove(event.keyCode) }
            case .mouseButton:
                initialPoint = CGPoint(x: event.x, y: event.y)
                if event.isDown { initialButtons.insert(event.button) } else { initialButtons.remove(event.button) }
            case .mouseMove, .scroll: initialPoint = CGPoint(x: event.x, y: event.y)
            case .flags: break
            }
        }
        var timeline = [InputSample(timestampNanos: startNanos, kind: .mouseMove, x: initialPoint.x, y: initialPoint.y, modifiers: initialModifiers)]
        if initialModifiers != 0 { timeline.append(InputSample(timestampNanos: startNanos, kind: .flags, modifiers: initialModifiers)) }
        timeline += initialKeys.sorted().map { InputSample(timestampNanos: startNanos, kind: .key, keyCode: $0, modifiers: initialModifiers, isDown: true) }
        timeline += initialButtons.sorted().map { InputSample(timestampNanos: startNanos, kind: .mouseButton, x: initialPoint.x, y: initialPoint.y, button: $0, modifiers: initialModifiers, isDown: true) }
        timeline += selected
        lock.withLock { generation = token; gameCameraReplay = isGameCamera }
        let newTask = Task.detached(priority: .userInitiated) { [weak self] in
            guard let self else { return }
            let clock = ContinuousClock(); let started = clock.now
            let replaySpeed = min(100, speed)
            for event in timeline {
                if Task.isCancelled { break }
                guard event.timestampNanos >= base else { continue }
                let relative = max(0, (Double(event.timestampNanos - base) / 1e9 - trimStart) / replaySpeed)
                try? await clock.sleep(until: started.advanced(by: .seconds(relative)))
                if Task.isCancelled { break }
                self.post(event, controlRect: recording.manifest.globalRect.cgRect, gameCamera: isGameCamera, token: token)
            }
            self.finish(token: token, reason: Task.isCancelled ? "Reenactment stopped" : nil)
        }
        lock.withLock { if generation == token { task = newTask } else { newTask.cancel() } }
    }

    func stop() {
        let snapshot = lock.withLock { () -> (Task<Void, Never>?, Set<UInt16>, Set<UInt8>, CGPoint, Bool) in
            let value = task; task = nil; generation = nil
            let keys = heldKeys, buttons = heldButtons, point = cursor, gameCamera = gameCameraReplay; heldKeys.removeAll(); heldButtons.removeAll(); gameCameraReplay = false
            return (value, keys, buttons, point, gameCamera)
        }
        snapshot.0?.cancel(); release(keys: snapshot.1, buttons: snapshot.2, at: snapshot.3, gameCamera: snapshot.4)
    }

    private func post(_ sample: InputSample, controlRect: CGRect, gameCamera: Bool, token: UUID) {
        lock.lock(); defer { lock.unlock() }
        guard generation == token else { return }
        switch sample.kind {
        case .mouseMove:
            if gameCamera {
                cursor = CGPoint(x: controlRect.midX, y: controlRect.midY)
                let dx = boundedInt64(sample.deltaX)
                let dy = boundedInt64(sample.deltaY)
                guard dx != 0 || dy != 0 else { return }
                _ = CGWarpMouseCursorPosition(cursor)
                // Raw locked-camera movement stays mouseMoved even while a
                // replayed button is held; clicks are separate edge events.
                guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: .mouseMoved, mouseCursorPosition: cursor, mouseButton: .left) else { return }
                event.setIntegerValueField(.mouseEventDeltaX, value: dx)
                event.setIntegerValueField(.mouseEventDeltaY, value: dy)
                taggedPost(event)
                _ = CGWarpMouseCursorPosition(cursor)
            } else {
                cursor = CGPoint(x: sample.x, y: sample.y).clampedForReplay(to: controlRect)
                let movement = movementEvent()
                guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: movement.type, mouseCursorPosition: cursor, mouseButton: movement.button) else { return }
                event.setIntegerValueField(.mouseEventDeltaX, value: boundedInt64(sample.deltaX))
                event.setIntegerValueField(.mouseEventDeltaY, value: boundedInt64(sample.deltaY))
                taggedPost(event)
            }
        case .mouseButton:
            let button = CGMouseButton(rawValue: UInt32(sample.button)) ?? .left
            let type: CGEventType = sample.button == 0 ? (sample.isDown ? .leftMouseDown : .leftMouseUp) : sample.button == 1 ? (sample.isDown ? .rightMouseDown : .rightMouseUp) : (sample.isDown ? .otherMouseDown : .otherMouseUp)
            guard let event = CGEvent(mouseEventSource: hidEventSource, mouseType: type, mouseCursorPosition: cursor, mouseButton: button) else { return }
            event.setIntegerValueField(.mouseEventButtonNumber, value: Int64(sample.button)); taggedPost(event)
            if sample.isDown { heldButtons.insert(sample.button) } else { heldButtons.remove(sample.button) }
        case .scroll:
            if let event = CGEvent(scrollWheelEvent2Source: hidEventSource, units: .pixel, wheelCount: 2, wheel1: boundedInt32(sample.scrollY), wheel2: boundedInt32(sample.scrollX), wheel3: 0) { taggedPost(event) }
        case .key:
            if let event = CGEvent(keyboardEventSource: hidEventSource, virtualKey: sample.keyCode, keyDown: sample.isDown) { event.flags = CGEventFlags(rawValue: sample.modifiers); taggedPost(event) }
            if sample.isDown { heldKeys.insert(sample.keyCode) } else { heldKeys.remove(sample.keyCode) }
        case .flags:
            let mappings: [(mask: CGEventFlags, keys: [UInt16])] = [
                (.maskShift, [56, 60]), (.maskControl, [59, 62]),
                (.maskAlternate, [58, 61]), (.maskCommand, [55, 54])
            ]
            for mapping in mappings {
                let held = heldKeys.intersection(mapping.keys)
                let desired = sample.modifiers & mapping.mask.rawValue != 0
                if desired, held.isEmpty {
                    let code = mapping.keys.contains(sample.keyCode) ? sample.keyCode : mapping.keys[0]
                    if let event = CGEvent(keyboardEventSource: hidEventSource, virtualKey: code, keyDown: true) {
                        event.flags = CGEventFlags(rawValue: sample.modifiers); taggedPost(event); heldKeys.insert(code)
                    }
                } else if !desired, !held.isEmpty {
                    for code in held {
                        if let event = CGEvent(keyboardEventSource: hidEventSource, virtualKey: code, keyDown: false) {
                            event.flags = CGEventFlags(rawValue: sample.modifiers); taggedPost(event)
                        }
                        heldKeys.remove(code)
                    }
                }
            }
        }
    }

    private func finish(token: UUID, reason: String?) {
        let snapshot = lock.withLock { () -> (Set<UInt16>, Set<UInt8>, CGPoint, Bool)? in
            guard generation == token else { return nil }
            generation = nil; task = nil
            let value = (heldKeys, heldButtons, cursor, gameCameraReplay); heldKeys.removeAll(); heldButtons.removeAll(); gameCameraReplay = false; return value
        }
        guard let snapshot else { return }
        release(keys: snapshot.0, buttons: snapshot.1, at: snapshot.2, gameCamera: snapshot.3)
        onFinish?(reason)
    }

    private func release(keys: Set<UInt16>, buttons: Set<UInt8>, at point: CGPoint, gameCamera: Bool) {
        for key in keys { if let event = CGEvent(keyboardEventSource: hidEventSource, virtualKey: key, keyDown: false) { taggedPost(event) } }
        for raw in buttons { let button = CGMouseButton(rawValue: UInt32(raw)) ?? .left; let type: CGEventType = raw == 0 ? .leftMouseUp : raw == 1 ? .rightMouseUp : .otherMouseUp; if let event = CGEvent(mouseEventSource: hidEventSource, mouseType: type, mouseCursorPosition: point, mouseButton: button) { taggedPost(event) } }
        if gameCamera {
            CGAssociateMouseAndMouseCursorPosition(1)
            let neutralPoint = CGEvent(source: hidEventSource)?.location ?? point
            if let neutral = CGEvent(mouseEventSource: hidEventSource, mouseType: .mouseMoved, mouseCursorPosition: neutralPoint, mouseButton: .left) {
                neutral.setIntegerValueField(.mouseEventDeltaX, value: 0); neutral.setIntegerValueField(.mouseEventDeltaY, value: 0); taggedPost(neutral)
            }
        }
    }

    private func taggedPost(_ event: CGEvent) { event.setIntegerValueField(.eventSourceUserData, value: agentTrainerSyntheticTag); event.post(tap: .cghidEventTap) }

    private func boundedInt64(_ value: Double) -> Int64 {
        guard value.isFinite else { return 0 }
        return Int64(min(Double(GameCameraContract.maximumPostedDelta), max(-Double(GameCameraContract.maximumPostedDelta), value)).rounded())
    }

    private func boundedInt32(_ value: Double) -> Int32 {
        guard value.isFinite else { return 0 }
        return Int32(min(10_000, max(-10_000, value)).rounded())
    }

    private func movementEvent() -> (type: CGEventType, button: CGMouseButton) {
        if heldButtons.contains(0) { return (.leftMouseDragged, .left) }
        if heldButtons.contains(1) { return (.rightMouseDragged, .right) }
        if let other = heldButtons.sorted().first { return (.otherMouseDragged, CGMouseButton(rawValue: UInt32(other)) ?? .center) }
        return (.mouseMoved, .left)
    }
}

private extension CGPoint { func clampedForReplay(to rect: CGRect) -> CGPoint { CGPoint(x: min(rect.maxX, max(rect.minX, x)), y: min(rect.maxY, max(rect.minY, y))) } }
