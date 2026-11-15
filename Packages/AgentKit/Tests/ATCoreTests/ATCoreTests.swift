import Testing
@testable import ATCore

@Test func bundleIdentifiersAreStable() {
    #expect(AgentTrainerInfo.bundleIdentifier == "com.unendless.AgentTrainer")
}
