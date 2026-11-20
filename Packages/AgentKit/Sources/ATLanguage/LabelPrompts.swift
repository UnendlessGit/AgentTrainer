import ATCore
import Foundation

/// The labeller's answer for one clip.
public struct LabelAnswer: Equatable, Sendable {
    public var observation: String
    public var instruction: String
    public var variations: [String]
    public var sameAsPrevious: Bool
}

/// The double-check's verdict on an instruction.
public struct LabelReview: Equatable, Sendable {
    public var accurate: Bool
    public var instruction: String?
    public var variations: [String]
}

/// Prompts for the labelling model and strict parsing of its answers.
public enum LabelPrompts {
    // MARK: Labelling

    public static let system = """
    You label gameplay screen recordings for AgentTrainer. Each label is an instruction an AI agent will \
    receive while it plays; it must learn to do what the person in the clip did when it gets that \
    instruction. Labels that don't match the footage teach it the wrong thing, so accuracy matters \
    more than detail.

    How to label:
    - Watch the whole clip. Movement matters: notice where the view goes and what the person heads for.
    - Use the input log to know what they pressed: W/S usually move forward/back, A/D sideways or steer, \
    Space jumps, mouse movement turns the camera. The log is exact; the picture tells you why.
    - Write one goal-level instruction in the imperative, 3 to 12 words, about what they were trying to \
    do (e.g. "Drive to the gas station and park", "Jump across the gap to the ledge"). Not key names.
    - Mention only things clearly visible. If you can't tell what something is, describe how it looks \
    ("the red building"). Never invent places, objects, enemies or goals.
    - Say left or right only when the input log or the picture clearly shows that direction.
    - Follow what the person told you about the footage; it overrides your own guesses.
    - Continuity: you get the instructions written for the clips just before. If this clip carries on \
    the same goal, say so and repeat that instruction exactly. If something new happens, write a new \
    instruction about what is new; don't fall back on a generic one like "Keep driving" or "Move forward" \
    when the clip shows something more specific.
    Reply with JSON only.
    """

    public static func labelPrompt(clip: LabelWindow, recordingName: String, app: String?, task: String,
                                   inputLog: String, previous: [TextSegment], guidance: String, variations: Int) -> String {
        var p = "Clip: \(f(clip.start))–\(f(clip.end)) s of the recording “\(recordingName)”"
        if let app, !app.isEmpty { p += ", captured from \(app)" }
        p += ". The video above is this clip.\n"
        if !task.trimmingCharacters(in: .whitespaces).isEmpty { p += "The person described the session as: “\(task)”.\n" }
        if !guidance.isEmpty { p += "\nWhat the person told you about this footage (follow it):\n\(guidance)\n" }
        p += "\nInput log (seconds from the start of this clip):\n\(inputLog)\n"
        if previous.isEmpty {
            p += "\nThis is the first clip of the recording.\n"
        } else {
            p += "\nInstructions for the clips just before (oldest first):\n"
            for s in previous {
                p += "- \(f(s.start))–\(f(s.end)) s: “\(s.text)”"
                if let o = s.observation, !o.isEmpty { p += " (seen: \(o))" }
                p += "\n"
            }
        }
        p += """

        Reply with this JSON:
        {
          "observation": "1–2 sentences: what is on screen and what changes during the clip",
          "same_goal_as_previous": true or false,
          "instruction": "the instruction",
          "variations": [\(variations) other ways to say the same instruction]
        }
        """
        p += variations > 0 ? """

        The \(variations) variations must mean exactly the same thing and be just as accurate: change the verb, \
        the sentence shape or how specific it is (one shorter, one more descriptive), all in the imperative. \
        Don't add anything that isn't in the clip.
        """ : "\nLeave \"variations\" empty."
        return p
    }

    // MARK: Double-check

    /// The follow-up asking the model to check its own instruction (it still has the clip in memory).
    public static func reviewPrompt(answer: LabelAnswer, variations: Int) -> String {
        """
        Now check that instruction, “\(answer.instruction)”, against the clip and the input log once more. \
        Is everything it mentions actually visible, and is it what the person was doing? Small wording \
        issues don't matter; wrong objects, places, directions or goals do.
        Reply with this JSON:
        {
          "accurate": true or false,
          "problem": "what is wrong, or empty",
          "instruction": "a corrected instruction (only when not accurate)",
          "variations": [\(variations) variations of the corrected instruction (only when not accurate)]
        }
        """
    }

    // MARK: Chat

    public static let chatSystem = """
    You are the labelling assistant in AgentTrainer, a Mac app that trains AI agents to play games from \
    screen recordings. Soon you will write instructions (labels) for the person's recordings. Right \
    now you're talking with them about the footage: you have seen an overview of it (frames spread \
    across the recordings). Describe what you see when asked, ask about anything unclear (which game, \
    what the goals are, names of places or characters), and agree on how the instructions should be \
    written. Keep replies short: 2 to 5 sentences. Don't write the labels yet.
    """

    /// The conversation, condensed for the labelling prompt.
    public static func guidance(from conversation: [LabelChatMessage], limit: Int = 3000) -> String {
        let lines = conversation.compactMap { m -> String? in
            let t = m.text.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !t.isEmpty else { return nil }
            return (m.role == .user ? "Person: " : "You: ") + t
        }
        var out: [String] = [], total = 0
        // Keep the latest turns when it's long; the person's own words matter most.
        for l in lines.reversed() {
            if total + l.count > limit, !out.isEmpty { break }
            out.insert(String(l.prefix(limit)), at: 0)
            total += l.count
        }
        return out.joined(separator: "\n")
    }

    // MARK: Parsing

    /// The first JSON object in a reply (models sometimes wrap it in prose or code fences).
    static func jsonObject(_ text: String) -> [String: Any]? {
        var t = text
        if let r = t.range(of: "</think>") { t = String(t[r.upperBound...]) }
        guard let open = t.firstIndex(of: "{") else { return nil }
        var depth = 0, inString = false, escaped = false
        var i = open
        while i < t.endIndex {
            let c = t[i]
            if inString {
                if escaped { escaped = false } else if c == "\\" { escaped = true } else if c == "\"" { inString = false }
            } else if c == "\"" {
                inString = true
            } else if c == "{" {
                depth += 1
            } else if c == "}" {
                depth -= 1
                if depth == 0 {
                    let s = String(t[open ... i])
                    return (try? JSONSerialization.jsonObject(with: Data(s.utf8))) as? [String: Any]
                }
            }
            i = t.index(after: i)
        }
        return nil
    }

    public static func parseAnswer(_ text: String, variations: Int) -> LabelAnswer? {
        guard let o = jsonObject(text), let raw = o["instruction"] as? String, let instruction = clean(raw) else { return nil }
        let obs = (o["observation"] as? String ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let same = (o["same_goal_as_previous"] as? Bool) ?? ((o["same_goal_as_previous"] as? String)?.lowercased() == "true")
        return LabelAnswer(observation: obs, instruction: instruction,
                           variations: cleanVariations(o["variations"] as? [Any] ?? [], main: instruction, limit: variations),
                           sameAsPrevious: same)
    }

    public static func parseReview(_ text: String, variations: Int) -> LabelReview? {
        guard let o = jsonObject(text) else { return nil }
        let accurate = (o["accurate"] as? Bool) ?? ((o["accurate"] as? String)?.lowercased() != "false")
        let fixed = (o["instruction"] as? String).flatMap(clean)
        return LabelReview(accurate: accurate || fixed == nil, instruction: accurate ? nil : fixed,
                           variations: fixed.map { cleanVariations(o["variations"] as? [Any] ?? [], main: $0, limit: variations) } ?? [])
    }

    /// One instruction: a single line, no quotes or label prefixes, capitalised, 2–16 words.
    public static func clean(_ s: String) -> String? {
        var t = s.trimmingCharacters(in: .whitespacesAndNewlines)
        if let line = t.split(separator: "\n").first { t = String(line) }
        for prefix in ["instruction:", "label:", "goal:", "- ", "* "] where t.lowercased().hasPrefix(prefix) { t = String(t.dropFirst(prefix.count)) }
        t = t.trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’`. ")).trimmingCharacters(in: .whitespaces)
        let words = t.split(separator: " ").count
        guard words >= 2, words <= 16 else { return nil }
        return t.prefix(1).uppercased() + t.dropFirst()
    }

    static func key(_ s: String) -> String {
        s.lowercased().filter { $0.isLetter || $0.isNumber || $0 == " " }.split(separator: " ").joined(separator: " ")
    }

    static func cleanVariations(_ raw: [Any], main: String, limit: Int) -> [String] {
        var seen: Set<String> = [key(main)], out: [String] = []
        for v in raw {
            guard out.count < limit, let s = v as? String, let c = clean(s), seen.insert(key(c)).inserted else { continue }
            out.append(c)
        }
        return out
    }

    static func f(_ x: Double) -> String { String(format: "%.1f", x) }
}
