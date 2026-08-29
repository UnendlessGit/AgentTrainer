import SwiftUI

/// Reusable categorical picker grid. The caller owns selections by stable IDs,
/// so display sorting and later renames never alter model tensor positions.
struct ContextSelectionEditor: View {
    let fields: [ContextFieldDefinition]
    @Binding var selection: [UUID: UUID]
    var blankLabel = "Not set"
    var accent = ATColor.cyan

    private var displayFields: [ContextFieldDefinition] {
        fields.sorted {
            $0.name.localizedStandardCompare($1.name) == .orderedAscending
        }
    }

    var body: some View {
        if fields.isEmpty {
            Text("No context fields yet.")
                .font(.caption)
                .foregroundStyle(.secondary)
        } else {
            LazyVGrid(
                columns: [GridItem(.adaptive(minimum: 190), spacing: 12)],
                alignment: .leading,
                spacing: 10
            ) {
                ForEach(displayFields) { field in
                    VStack(alignment: .leading, spacing: 5) {
                        Text(field.name)
                            .font(.caption.bold())
                            .foregroundStyle(accent)
                        Picker(field.name, selection: selectionBinding(for: field)) {
                            Text(blankLabel).tag(UUID?.none)
                            ForEach(field.values.sorted {
                                $0.name.localizedStandardCompare($1.name) == .orderedAscending
                            }) { value in
                                Text(value.name).tag(Optional(value.id))
                            }
                        }
                        .labelsHidden()
                        .frame(maxWidth: .infinity)
                    }
                }
            }
        }
    }

    private func selectionBinding(for field: ContextFieldDefinition) -> Binding<UUID?> {
        Binding(
            get: { selection[field.id] },
            set: { valueID in
                var updated = selection
                if let valueID { updated[field.id] = valueID }
                else { updated.removeValue(forKey: field.id) }
                selection = updated
            }
        )
    }
}

/// Uses the same editor for one or many Library recordings. Mixed values remain
/// untouched until the user deliberately chooses a value or Blank.
struct RecordingContextEditor: View {
    @ObservedObject var model: AppModel
    let recordings: [RecordingItem]
    var accent = ATColor.amber

    private enum Choice: Hashable {
        case mixed
        case blank
        case value(UUID)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            if model.contextFields.isEmpty {
                Text("Create a context field to label recordings.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(model.contextFields) { field in
                    VStack(alignment: .leading, spacing: 5) {
                        Text(field.name)
                            .font(.caption.bold())
                            .foregroundStyle(accent)
                        Picker(field.name, selection: choiceBinding(for: field)) {
                            if commonChoice(for: field) == .mixed {
                                Text("Mixed — leave unchanged").tag(Choice.mixed)
                            }
                            Text(recordings.count == 1 ? "Blank" : "Blank — remove from all")
                                .tag(Choice.blank)
                            ForEach(field.values.sorted {
                                $0.name.localizedStandardCompare($1.name) == .orderedAscending
                            }) { value in
                                Text(value.name).tag(Choice.value(value.id))
                            }
                        }
                        .labelsHidden()
                    }
                }
            }
        }
    }

    private func commonChoice(for field: ContextFieldDefinition) -> Choice {
        guard let first = recordings.first else { return .blank }
        let firstValue = first.manifest.effectiveContextAssignments.first {
            $0.fieldID == field.id
        }?.valueID
        guard recordings.dropFirst().allSatisfy({ recording in
            recording.manifest.effectiveContextAssignments.first {
                $0.fieldID == field.id
            }?.valueID == firstValue
        }) else { return .mixed }
        return firstValue.map(Choice.value) ?? .blank
    }

    private func choiceBinding(for field: ContextFieldDefinition) -> Binding<Choice> {
        Binding(
            get: { commonChoice(for: field) },
            set: { choice in
                guard choice != .mixed else { return }
                let valueID: UUID? = switch choice {
                case .mixed, .blank: nil
                case .value(let id): id
                }
                Task {
                    await model.setContextValue(
                        valueID,
                        fieldID: field.id,
                        recordings: recordings
                    )
                }
            }
        )
    }
}

/// Shared catalog manager used from Record and Library. Deleting a definition
/// explicitly removes its assignments from affected recordings, while saved
/// model versions remain immutable and keep their trained vocabulary snapshot.
struct ContextCatalogEditor: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @State private var newFieldName = ""
    @State private var isAddingField = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Context fields and values").font(.title2.bold())
                    Text("Create reusable categorical labels. Names and values are entirely yours; AgentTrainer adds no game-specific meaning.")
                        .font(.caption).foregroundStyle(.secondary)
                }
                Spacer()
                Button("Done") { dismiss() }.primaryButton(color: ATColor.cyan)
            }

            HStack {
                TextField("New field name", text: $newFieldName)
                    .textFieldStyle(.roundedBorder)
                Button("Add Field") {
                    let name = newFieldName
                    isAddingField = true
                    Task {
                        if await model.createContextField(name: name) {
                            newFieldName = ""
                        }
                        isAddingField = false
                    }
                }
                .primaryButton(color: ATColor.violet)
                .disabled(
                    isAddingField
                        || newFieldName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                )
            }

            Divider()
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(model.contextFields) { field in
                        ContextFieldManagementRow(field: field, model: model)
                    }
                    if model.contextFields.isEmpty {
                        ContentUnavailableView(
                            "No context fields",
                            systemImage: "tag",
                            description: Text("Add your first field above, then create any values it needs.")
                        )
                        .padding(.vertical, 40)
                    }
                }
                .padding(.vertical, 2)
            }
        }
        .padding(22)
        .frame(minWidth: 620, minHeight: 480)
        .background(ATColor.canvas)
    }
}

private struct ContextFieldManagementRow: View {
    let field: ContextFieldDefinition
    @ObservedObject var model: AppModel
    @State private var fieldName: String
    @State private var newValueName = ""
    @State private var isAddingValue = false
    @State private var confirmsFieldDeletion = false

    init(field: ContextFieldDefinition, model: AppModel) {
        self.field = field
        self.model = model
        _fieldName = State(initialValue: field.name)
    }

    var body: some View {
        OLEDCard(padding: 12) {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Image(systemName: "tag.fill").foregroundStyle(ATColor.violet)
                    TextField("Field name", text: $fieldName)
                        .textFieldStyle(.roundedBorder)
                    Button("Rename") {
                        Task { await model.renameContextField(field, name: fieldName) }
                    }
                    .disabled(
                        fieldName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                            || fieldName == field.name
                    )
                    Button(role: .destructive) {
                        confirmsFieldDeletion = true
                    } label: {
                        Image(systemName: "trash")
                    }
                    .help("Delete this field and remove it from recordings")
                }

                if field.values.isEmpty {
                    Text("No values yet")
                        .font(.caption).foregroundStyle(.secondary)
                } else {
                    VStack(spacing: 7) {
                        ForEach(field.values) { value in
                            ContextValueManagementRow(
                                field: field,
                                value: value,
                                model: model
                            )
                        }
                    }
                }

                HStack {
                    TextField("New value", text: $newValueName)
                        .textFieldStyle(.roundedBorder)
                    Button("Add Value") {
                        let name = newValueName
                        isAddingValue = true
                        Task {
                            if await model.createContextValue(fieldID: field.id, name: name) {
                                newValueName = ""
                            }
                            isAddingValue = false
                        }
                    }
                    .primaryButton(color: ATColor.green)
                    .disabled(
                        isAddingValue
                            || newValueName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    )
                }
            }
        }
        .alert("Delete context field?", isPresented: $confirmsFieldDeletion) {
            Button("Cancel", role: .cancel) {}
            Button("Delete Field", role: .destructive) {
                Task { await model.deleteContextField(field) }
            }
        } message: {
            Text("“\(field.name)” and its values will be removed from all recordings. Existing trained brains keep their immutable context snapshot.")
        }
        .onChange(of: field.name) { _, updated in fieldName = updated }
    }
}

private struct ContextValueManagementRow: View {
    let field: ContextFieldDefinition
    let value: ContextValueDefinition
    @ObservedObject var model: AppModel
    @State private var valueName: String
    @State private var confirmsDeletion = false

    init(
        field: ContextFieldDefinition,
        value: ContextValueDefinition,
        model: AppModel
    ) {
        self.field = field
        self.value = value
        self.model = model
        _valueName = State(initialValue: value.name)
    }

    var body: some View {
        HStack {
            Circle().fill(ATColor.cyan).frame(width: 5, height: 5)
            TextField("Value", text: $valueName).textFieldStyle(.roundedBorder)
            Button("Rename") {
                Task {
                    await model.renameContextValue(value, in: field, name: valueName)
                }
            }
            .disabled(
                valueName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || valueName == value.name
            )
            Button(role: .destructive) { confirmsDeletion = true } label: {
                Image(systemName: "minus.circle")
            }
            .buttonStyle(.plain)
            .foregroundStyle(ATColor.coral)
        }
        .alert("Delete context value?", isPresented: $confirmsDeletion) {
            Button("Cancel", role: .cancel) {}
            Button("Delete Value", role: .destructive) {
                Task { await model.deleteContextValue(value, in: field) }
            }
        } message: {
            Text("“\(value.name)” will be removed from recordings labeled with it. Existing trained brains remain unchanged.")
        }
        .onChange(of: value.name) { _, updated in valueName = updated }
    }
}
