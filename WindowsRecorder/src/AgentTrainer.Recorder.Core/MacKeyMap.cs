namespace AgentTrainer.Recorder.Core;

/// <summary>
/// Translates Windows set-1 physical scan codes to the Apple virtual-key space
/// already used by AgentTrainer's fixed 128-key policy tensor. Text/layout
/// translation is intentionally not used.
/// </summary>
public static class MacKeyMap
{
    public const ushort VkPause = 0x13;
    public const ushort VkSnapshot = 0x2C;

    public static ushort? Translate(ushort makeCode, bool extended, ushort virtualKey = 0)
    {
        if (virtualKey == VkPause) return 113;     // F15 / Pause
        if (virtualKey == VkSnapshot) return 105; // F13 / Print Screen
        if (extended)
        {
            return makeCode switch
            {
                0x1C => 76,  // keypad enter
                0x1D => 62,  // right control
                0x35 => 75,  // keypad divide
                0x38 => 61,  // right option / AltGr
                0x47 => 115, // home
                0x48 => 126, // up
                0x49 => 116, // page up
                0x4B => 123, // left
                0x4D => 124, // right
                0x4F => 119, // end
                0x50 => 125, // down
                0x51 => 121, // page down
                0x52 => 114, // help / insert
                0x53 => 117, // forward delete
                0x5B => 55,  // left command / Windows
                0x5C => 54,  // right command / Windows
                0x5D => 110, // context menu (stable otherwise-unused slot)
                _ => null
            };
        }

        return makeCode switch
        {
            0x01 => 53, 0x02 => 18, 0x03 => 19, 0x04 => 20, 0x05 => 21,
            0x06 => 23, 0x07 => 22, 0x08 => 26, 0x09 => 28, 0x0A => 25,
            0x0B => 29, 0x0C => 27, 0x0D => 24, 0x0E => 51, 0x0F => 48,
            0x10 => 12, 0x11 => 13, 0x12 => 14, 0x13 => 15, 0x14 => 17,
            0x15 => 16, 0x16 => 32, 0x17 => 34, 0x18 => 31, 0x19 => 35,
            0x1A => 33, 0x1B => 30, 0x1C => 36, 0x1D => 59, 0x1E => 0,
            0x1F => 1, 0x20 => 2, 0x21 => 3, 0x22 => 5, 0x23 => 4,
            0x24 => 38, 0x25 => 40, 0x26 => 37, 0x27 => 41, 0x28 => 39,
            0x29 => 50, 0x2A => 56, 0x2B => 42, 0x2C => 6, 0x2D => 7,
            0x2E => 8, 0x2F => 9, 0x30 => 11, 0x31 => 45, 0x32 => 46,
            0x33 => 43, 0x34 => 47, 0x35 => 44, 0x36 => 60, 0x37 => 67,
            0x38 => 58, 0x39 => 49, 0x3A => 57, 0x3B => 122, 0x3C => 120,
            0x3D => 99, 0x3E => 118, 0x3F => 96, 0x40 => 97, 0x41 => 98,
            0x42 => 100, 0x43 => 101, 0x44 => 109, 0x45 => 71, 0x46 => 107,
            0x47 => 89, 0x48 => 91, 0x49 => 92, 0x4A => 78, 0x4B => 86,
            0x4C => 87, 0x4D => 88, 0x4E => 69, 0x4F => 83, 0x50 => 84,
            0x51 => 85, 0x52 => 82, 0x53 => 65, 0x56 => 10, 0x57 => 103,
            0x58 => 111,
            _ => null
        };
    }

    public static bool IsModifierKey(ushort appleCode) => appleCode is 54 or 55 or 56 or 58 or 59 or 60 or 61 or 62;

    public static ulong ModifierMask(ushort appleCode) => appleCode switch
    {
        56 or 60 => QuartzModifierFlags.Shift,
        59 or 62 => QuartzModifierFlags.Control,
        58 or 61 => QuartzModifierFlags.Option,
        54 or 55 => QuartzModifierFlags.Command,
        _ => 0
    };

    public static string Name(ushort code) => code switch
    {
        0 => "A", 1 => "S", 2 => "D", 3 => "F", 4 => "H", 5 => "G", 6 => "Z", 7 => "X", 8 => "C", 9 => "V",
        11 => "B", 12 => "Q", 13 => "W", 14 => "E", 15 => "R", 16 => "Y", 17 => "T", 18 => "1", 19 => "2", 20 => "3",
        21 => "4", 22 => "6", 23 => "5", 24 => "=", 25 => "9", 26 => "7", 27 => "-", 28 => "8", 29 => "0", 30 => "]",
        31 => "O", 32 => "U", 33 => "[", 34 => "I", 35 => "P", 36 => "Enter", 37 => "L", 38 => "J", 39 => "'", 40 => "K",
        41 => ";", 42 => "\\", 43 => ",", 44 => "/", 45 => "N", 46 => "M", 47 => ".", 48 => "Tab", 49 => "Space", 50 => "`",
        51 => "Backspace", 53 => "Esc", 54 or 55 => "Win", 56 or 60 => "Shift", 57 => "Caps", 58 or 61 => "Alt", 59 or 62 => "Ctrl",
        123 => "Left", 124 => "Right", 125 => "Down", 126 => "Up",
        _ => $"K{code}"
    };
}
