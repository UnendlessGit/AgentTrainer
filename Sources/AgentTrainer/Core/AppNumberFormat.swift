import Foundation

/// Technical numbers use the same decimal point on every Mac. Setting a SwiftUI
/// environment locale alone does not affect strings formatted by Foundation.
enum AppNumberFormat {
    static let locale = Locale(identifier: "en_US_POSIX")
    static let decimalInput = DecimalInputFormat()

    static func bytes(_ count: Int64, style: ByteCountFormatStyle.Style) -> String {
        count.formatted(.byteCount(style: style).locale(locale))
    }

    static func decimalList(_ values: [Double]) -> String {
        values.map(decimalInput.format).joined(separator: ", ")
    }

    static func parseDecimalList(_ text: String) throws -> [Double] {
        // Commas separate values, never decimal digits. Reject the whole edit
        // when a token is invalid instead of silently dropping list elements.
        try text.split(separator: ",", omittingEmptySubsequences: false).map {
            try decimalInput.parseStrategy.parse(String($0))
        }
    }
}

/// A strict, locale-independent editor keeps small learning rates visible and
/// prevents a decimal comma from being silently read as grouping or a prefix.
struct DecimalInputFormat: ParseableFormatStyle {
    var parseStrategy: DecimalInputParseStrategy { DecimalInputParseStrategy() }

    func format(_ value: Double) -> String {
        value.formatted(.number.locale(AppNumberFormat.locale).grouping(.never).precision(.significantDigits(1...17)))
    }
}

struct DecimalInputParseStrategy: ParseStrategy {
    enum InvalidNumber: Error { case invalid }

    func parse(_ value: String) throws -> Double {
        guard let number = Double(value.trimmingCharacters(in: .whitespacesAndNewlines)),
              number.isFinite else { throw InvalidNumber.invalid }
        return number
    }
}
