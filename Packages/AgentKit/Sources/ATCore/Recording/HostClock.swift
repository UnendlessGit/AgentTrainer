import Darwin
import Foundation

/// One monotonic nanosecond timeline (mach absolute time) shared by screen frames and input
/// events, so they can be aligned exactly.
public enum HostClock {
    private static let timebase: mach_timebase_info_data_t = {
        var t = mach_timebase_info_data_t()
        mach_timebase_info(&t)
        return t
    }()

    /// Current time in nanoseconds on the host clock.
    public static func now() -> UInt64 { ticksToNanoseconds(mach_absolute_time()) }

    public static func ticksToNanoseconds(_ ticks: UInt64) -> UInt64 {
        timebase.numer == timebase.denom ? ticks : ticks * UInt64(timebase.numer) / UInt64(timebase.denom)
    }

    public static func nanosecondsToTicks(_ ns: UInt64) -> UInt64 {
        timebase.numer == timebase.denom ? ns : ns * UInt64(timebase.denom) / UInt64(timebase.numer)
    }

    /// Sleeps until an absolute host time (used by fixed-rate loops).
    public static func sleep(untilNanoseconds ns: UInt64) {
        mach_wait_until(nanosecondsToTicks(ns))
    }
}
