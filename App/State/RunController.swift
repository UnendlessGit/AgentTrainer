import ATCore
import Foundation
import Observation

@MainActor @Observable
final class RunController {
    private let settings: AppSettings
    private let models: ModelStore
    init(settings: AppSettings, models: ModelStore) { self.settings = settings; self.models = models }
    var isRunning = false
    var footerLine = ""
    func toggle() async {}
    func togglePause() {}
    func emergencyStop() {}
}
