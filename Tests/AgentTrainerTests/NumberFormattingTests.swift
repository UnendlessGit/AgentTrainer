import Foundation
import XCTest
@testable import AgentTrainer

final class NumberFormattingTests: XCTestCase {
    func testTechnicalNumbersUseDecimalPoints() {
        let style = FloatingPointFormatStyle<Double>.number.locale(AppNumberFormat.locale)
        XCTAssertEqual(style.precision(.fractionLength(4)).format(0.0015), "0.0015")
        XCTAssertEqual(style.format(1234.5), "1234.5")
        XCTAssertEqual(1234.formatted(.number.locale(AppNumberFormat.locale)), "1234")
        XCTAssertEqual(0.125.formatted(.percent.locale(AppNumberFormat.locale)), "12.5%")
        XCTAssertEqual(AppNumberFormat.bytes(1_250_000, style: .file), "1.2 MB")
        XCTAssertEqual(AppNumberFormat.bytes(1_310_720, style: .memory), "1.2 MB")
    }

    func testDecimalEditorRetainsSmallRatesAndPrecision() throws {
        let format = AppNumberFormat.decimalInput
        let values = [0.0000001, 0.0003, 0.0015, 0.1234567890123456, 1.0 / 30, -125.5, 1_000_000]
        XCTAssertEqual(format.format(0.0000001), "0.0000001")
        XCTAssertEqual(format.format(1_000_000), "1000000")
        for value in values {
            let text = format.format(value)
            XCTAssertFalse(text.contains(","))
            XCTAssertEqual(try format.parseStrategy.parse(text), value)
        }
        XCTAssertEqual(try format.parseStrategy.parse(" 1.5e-4\n"), 0.00015)
    }

    func testDecimalEditorRejectsAmbiguousOrPartialValues() {
        for text in ["0,0015", "1,234.5", "1.2 trailing", "", ".", "NaN", "inf", "1e999"] {
            XCTAssertThrowsError(try AppNumberFormat.decimalInput.parseStrategy.parse(text), text)
        }
    }

    func testFutureActionListRoundTripsWithoutChangingValues() throws {
        let values = [1.0 / 30, 0.1, 0.0000001, 1_000.25]
        let text = AppNumberFormat.decimalList(values)
        XCTAssertEqual(text, "0.03333333333333333, 0.1, 0.0000001, 1000.25")
        XCTAssertEqual(try AppNumberFormat.parseDecimalList(text), values)
        for text in ["0.1, bad, 0.3", "0.1,,0.3", "0.1,", "", "nan, 0.3"] {
            XCTAssertThrowsError(try AppNumberFormat.parseDecimalList(text), text)
        }
    }

    func testRegionalRegressionLaunchUsesRequestedLocale() {
        // The release check launches this suite with -AppleLocale for each Mac
        // region, so it exercises the actual process default as well as styles.
        guard let expected = ProcessInfo.processInfo.environment["AGENTTRAINER_EXPECTED_TEST_LOCALE"] else { return }
        XCTAssertEqual(Locale.current.identifier, expected)
        if ["de_DE", "fr_FR", "tr_TR"].contains(expected) {
            XCTAssertTrue(1.5.formatted().contains(","), "The regression must run in a decimal-comma locale")
            XCTAssertEqual(AppNumberFormat.decimalInput.format(1.5), "1.5")
        }
    }
}
