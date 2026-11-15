import Testing
@testable import ATCore

@Test func bundleIdentifiersAreStable() {
    #expect(AgentTrainerInfo.bundleIdentifier == "com.unendless.AgentTrainer")
}

@Test func presetsHaveValidAttentionShapes() {
    for a in [PolicyArchitecture.p2p150M, .p2p300M, .compact60M] {
        #expect(a.width % a.heads == 0)
        #expect(a.headDim % 2 == 0)
        #expect(a.heads % a.resolvedKVHeads == 0)
        #expect(a.resolvedKVHeads == a.kvHeads)
        #expect(a.width % a.decoderHeads == 0)
    }
}

@Test func invalidKVHeadsFallBackToOnePerHead() {
    var a = PolicyArchitecture()
    a.heads = 12; a.kvHeads = 16
    #expect(a.resolvedKVHeads == 12)
    a.kvHeads = 4
    #expect(a.resolvedKVHeads == 4)
}
