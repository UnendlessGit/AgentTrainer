import Testing
@testable import ATCore

@Suite struct ActionSpaceTests {
    @Test func bucketizeMatchesTorchSemantics() {
        let e = ActionSpaceConfig.p2pXEdges
        #expect(bucketize(0, e) == 11) // P2P's zero bin for x
        #expect(bucketize(1, e) == 12)
        #expect(bucketize(-1, e) == 10)
        #expect(bucketize(2, e) == 13 && bucketize(3, e) == 13 && bucketize(4, e) == 14)
        #expect(bucketize(10_000, e) == 22 && bucketize(-10_000, e) == 0)
        #expect(bucketize(0, ActionSpaceConfig.p2pYEdges) == 8)
    }

    @Test func p2pSpaceHasPixel2PlaySizes() {
        let a = ActionSpaceConfig.p2p
        #expect(a.keyClasses == 20 && a.pointerClasses == 4 && a.xClasses == 23 && a.yClasses == 17)
        #expect(a.slotCount == 8)
        #expect(a.zeroXClass == 11 && a.zeroYClass == 8)
    }

    @Test func encodeSortsAndPadsLikeP2P() {
        let a = ActionSpaceConfig.p2p
        var s = TimestepAction()
        s.keys = ["w", "space"] // P2P classes 11 and 1
        s.pointer = [.left]
        let c = a.encode(s)
        #expect(Array(c[0 ..< 4]) == [1, 11, 0, 0])
        #expect(Array(c[4 ..< 6]) == [1, 0])
        #expect(c[6] == 11 && c[7] == 8)
    }

    @Test func extraKeysOnlyAddClassesAfterP2P() {
        let a = ActionSpaceConfig.gameDefault
        #expect(Array(a.keys.prefix(19)) == KeyCatalog.p2pKeys)
        #expect(a.keyIndex["r"]! > 19)
    }

    @Test func decodeRoundTripsBins() {
        var a = ActionSpaceConfig.p2p
        a.mouseSampling = .center
        for v in [-600.0, -100, -20, -5, -1, 0, 1, 2, 9, 40, 130, 700] {
            var s = TimestepAction(); s.dx = v
            let c = a.encode(s)
            let back = a.decode(c)
            #expect(bucketize(back.dx, a.mouseXEdges) == Int(c[6]), "x=\(v) decoded to \(back.dx)")
        }
    }

    @Test func truncatedNormalStaysInRange() {
        for _ in 0 ..< 200 {
            let v = ActionSpaceConfig.truncatedNormal(lo: 30, hi: 46, std: 96)
            #expect(v >= 29.5 && v <= 46.5)
        }
    }

    @Test func absoluteModeUsesGrid() {
        var a = ActionSpaceConfig.appDefault
        a.absoluteColumns = 10; a.absoluteRows = 5
        var s = TimestepAction(); s.absX = 0.95; s.absY = 0.0
        let c = a.encode(s)
        #expect(c[a.keySlots + a.pointerSlots] == 9 && c[a.keySlots + a.pointerSlots + 1] == 0)
    }

    @Test func scrollClassification() {
        #expect(PointerEvent.fromScroll(lines: 0, 1) == .scrollUp)
        #expect(PointerEvent.fromScroll(lines: 0, -5) == .scrollDownFast)
        #expect(PointerEvent.fromScroll(lines: 2, 0) == .scrollRight)
        #expect(PointerEvent.fromScroll(lines: 0, 0.2) == nil)
    }

    @Test func keyCatalogCoversP2PAndTyping() {
        for k in KeyCatalog.p2pKeys { #expect(KeyCatalog.info(k) != nil, "\(k)") }
        #expect(KeyCatalog.typingMap["A"]! == ("a", true))
        #expect(KeyCatalog.typingMap["?"]! == ("slash", true))
        #expect(Set(KeyCatalog.all.map(\.keyCode)).count == KeyCatalog.all.count)
    }
}
