import Foundation

enum ProcessExecutionError: LocalizedError {
    case commandFailed(String, Int32, String)

    var errorDescription: String? {
        switch self {
        case .commandFailed(let command, let status, let details):
            let suffix = details.isEmpty ? "" : " \(details)"
            return "\(command) failed with status \(status).\(suffix)"
        }
    }
}

enum ProcessRunner {
    static func run(
        _ executable: String,
        _ arguments: [String]
    ) async throws -> (stdout: Data, stderr: Data) {
        try await Task.detached(priority: .utility) {
            let process = Process()
            let stdout = Pipe()
            let stderr = Pipe()
            process.executableURL = URL(fileURLWithPath: executable)
            process.arguments = arguments
            process.standardOutput = stdout
            process.standardError = stderr
            try process.run()

            // Both pipes must drain while the child is running. Waiting first
            // deadlocks once either pipe fills its kernel buffer.
            async let outputRead = Task.detached(priority: .utility) {
                stdout.fileHandleForReading.readDataToEndOfFile()
            }.value
            async let errorRead = Task.detached(priority: .utility) {
                stderr.fileHandleForReading.readDataToEndOfFile()
            }.value
            process.waitUntilExit()
            let (output, error) = await (outputRead, errorRead)
            guard process.terminationStatus == 0 else {
                let details = String(data: error, encoding: .utf8)?
                    .trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                throw ProcessExecutionError.commandFailed(
                    URL(fileURLWithPath: executable).lastPathComponent,
                    process.terminationStatus,
                    details
                )
            }
            return (output, error)
        }.value
    }
}
