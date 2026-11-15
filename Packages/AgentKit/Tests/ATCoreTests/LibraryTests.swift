import Foundation
import Testing
@testable import ATCore

@Suite struct LibraryTests {
    func makeRecording(_ dir: URL, _ name: String) throws -> URL {
        let url = dir.appendingPathComponent(name + ".atrec")
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        let m = RecordingMeta(name: name, appVersion: "t", target: CaptureTargetInfo(kind: .display), captureWidth: 2, captureHeight: 2,
                              sourceRect: CodableRect(x: 0, y: 0, width: 1, height: 1), captureFPS: 30, showsCursor: false, input: InputCaptureOptions())
        try RecordingIO.writeMeta(m, to: url)
        return url
    }

    @Test func foldersAndMoves() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("lib-\(UUID())")
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let lib = LibraryFileSystem(root: root)
        let games = try lib.createFolder(named: "Games", in: root)
        let mc = try lib.createFolder(named: "Minecraft", in: games)
        let rec = try makeRecording(root, "Session 1")
        let moved = try lib.move(rec, into: mc)
        #expect(lib.recordings(in: mc, recursive: false).count == 1)
        #expect(lib.recordings(in: root, recursive: true).count == 1)
        let renamed = try lib.rename(moved, to: "Forest run")
        #expect(renamed.lastPathComponent == "Forest run.atrec")
        #expect(try RecordingIO.readMeta(renamed).name == "Forest run")
        #expect(throws: LibraryError.self) { try lib.move(games, into: mc) }
        #expect(throws: LibraryError.self) { try lib.createFolder(named: "a/b", in: root) }
        let tree = lib.folderTree()
        #expect(tree.children.first?.name == "Games" && tree.totalRecordingCount == 1)
        let renamedFolder = try lib.rename(mc, to: "MC")
        #expect(FileManager.default.fileExists(atPath: renamedFolder.appendingPathComponent("Forest run.atrec").path))
    }

    @Test func modelDefinitionRoundTrips() throws {
        let m = ModelDefinition.make(.appControl, name: "Desk")
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("m-\(UUID()).atmodel")
        defer { try? FileManager.default.removeItem(at: url) }
        try ModelPackage.write(m, to: url)
        var back = try ModelPackage.read(url)
        #expect(abs(back.createdAt.timeIntervalSince(m.createdAt)) < 1) // ISO-8601 keeps whole seconds
        back.createdAt = m.createdAt
        #expect(back == m)
        #expect(back.actions.mouseMode == .absolute && back.architecture.literalTyping)
    }
}
