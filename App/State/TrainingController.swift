import ATCore
import Foundation
import Observation

@MainActor @Observable
final class TrainingController {
    private let settings: AppSettings
    private let models: ModelStore
    private let library: LibraryStore
    init(settings: AppSettings, models: ModelStore, library: LibraryStore) {
        self.settings = settings; self.models = models; self.library = library
    }
    var footerLine: String? { nil }
}
