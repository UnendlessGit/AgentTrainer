# AgentTrainer 2.62 — consistent decimal points

Version 2.62 (build 40) fixes numeric settings and displays that changed decimal separators between Macs.

## Cause

SwiftUI numeric fields and Foundation number/percent/byte-count formatting inherited the Mac's regional preferences. Period-decimal regions displayed `0.0015`; comma-decimal regions displayed `0,0015`. The future-action horizon list also used localized decimal formatting but split its contents on commas, so a displayed list could be read back as different values.

## Changes

- Numeric fields and displays use an explicit `en_US_POSIX` numeric locale: decimal points without locale-dependent digit grouping.
- Decimal editors commit complete entries on Return or focus loss, reject invalid edits without saving partial numeric prefixes, use strict parsing, accept decimal points and scientific notation, and preserve up to 17 significant digits. Small learning rates such as `0.0000001` remain visible instead of rounding to zero.
- Future-action lists use decimal points and commas between values. Displaying and committing a list preserves its values and precision; malformed tokens reject the entire edit instead of silently dropping entries.
- The same policy covers demonstration and RL settings, recording/run settings, training and runtime metrics, reward charts, HUD values, storage sizes, and diagnostic messages. Date/time formatting still follows the Mac's preferences.

Saved profiles and model formats are unchanged. Existing values require no migration. The installer retains the existing signing certificate and designated requirement.

## Validation

- `./test.sh`: 244 tests, 7 skipped, no failures; signing regression checks passed.
- The five numeric regressions passed in separate processes launched with `-AppleLocale en_US`, `de_DE`, `fr_FR`, and `tr_TR`. Each process asserted its actual default locale; comma-region runs also verified that the system formatter still produced a decimal comma while app formatting produced a decimal point.
- Regression coverage includes small learning rates, precise decimal round trips, strict rejection of partial/ambiguous input, future-action list preservation, percentages, and storage sizes.
- Shell syntax checks passed for the build, test, and signing scripts.
- Final signed-app UI checks under `de_DE` used an isolated workspace: demonstration and advanced RL fields displayed decimal points, a learning rate of `0.0000001` persisted, entering `0,0003` left that saved rate unchanged, and a future-action list saved exactly `[0.125, 0.375, 0.625]`.
- The installed app and the app mounted from the DMG both report 2.62 (40), pass strict signature validation, and match the pre-update designated requirement. The signed DMG is 9,581,856 bytes. The source archive excludes private keys and signing pins; SHA-256 checksums were verified.
