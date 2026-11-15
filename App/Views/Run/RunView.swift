import ATCore
import SwiftUI
struct RunView: View {
    let app: AppModel
    var body: some View { ContentUnavailableView("Run", systemImage: "play.circle", description: Text("Coming in the next build.")) }
}
struct ArenaView: View {
    let app: AppModel
    var body: some View { ContentUnavailableView("Test Arena", systemImage: "gamecontroller", description: Text("Coming in the next build.")) }
}
struct AutoLabelButton: View {
    let app: AppModel
    let recording: LibraryRecording
    @Binding var annotations: RecordingAnnotations
    var body: some View { EmptyView() }
}
struct LanguageDownloadsRows: View {
    let app: AppModel
    var body: some View { EmptyView() }
}
