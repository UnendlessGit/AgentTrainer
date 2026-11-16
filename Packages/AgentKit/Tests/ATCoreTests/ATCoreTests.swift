import Testing
import Foundation
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

@Test func singleClassHeadsDoNotDivideByZero() {
    #expect(ActionSpaceConfig.lossScale(classes: 1) == 0)
    #expect(abs(ActionSpaceConfig.lossScale(classes: 2) - 1 / log(2.0)) < 1e-12)
    var space = ActionSpaceConfig.p2p
    space.pointerEvents = []
    #expect(space.pointerClasses == 1)
    #expect(ActionSpaceConfig.lossScale(classes: space.pointerClasses).isFinite)
}

@Test func progressWithNaNStillReachesTheApp() {
    let p = TrainingProgress(step: 3, totalSteps: 10, epoch: 0.1, loss: .nan, headLosses: [.nan, .infinity, 0, 0], learningRate: 1e-4,
                             gradNorm: .nan, secondsPerStep: 1, framesPerSecond: 100, gpuMemoryGB: 1, peakMemoryGB: 2)
    let data = TrainingEvent.progress(p).encoded()
    #expect(!data.isEmpty)
    guard case .progress(let back)? = TrainingEvent.decode(data) else { Issue.record("did not decode"); return }
    #expect(back.step == 3)
    #expect(back.loss.isNaN)
    #expect(back.headLosses[1] == .infinity)
}
