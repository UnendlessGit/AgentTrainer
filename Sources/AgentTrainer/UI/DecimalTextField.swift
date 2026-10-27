import SwiftUI

/// Keep an in-progress edit separate from the saved number. A value-bound
/// TextField writes valid prefixes (for example the `0` in `0,0003`) before it
/// can reject the complete entry, which can accidentally autosave that prefix.
struct DecimalTextField: View {
    let title: String
    @Binding private var value: Double
    @State private var text: String
    @FocusState private var isFocused: Bool

    init(_ title: String, value: Binding<Double>) {
        self.title = title
        _value = value
        _text = State(initialValue: AppNumberFormat.decimalInput.format(value.wrappedValue))
    }

    var body: some View {
        TextField(title, text: $text)
            .focused($isFocused)
            .onSubmit { commit() }
            .onChange(of: isFocused) { _, focused in
                if !focused { commit() }
            }
            .onChange(of: value) { _, updated in
                if !isFocused { text = AppNumberFormat.decimalInput.format(updated) }
            }
            .onDisappear { commit() }
    }

    private func commit() {
        if let parsed = try? AppNumberFormat.decimalInput.parseStrategy.parse(text) {
            value = parsed
        }
        text = AppNumberFormat.decimalInput.format(value)
    }
}
