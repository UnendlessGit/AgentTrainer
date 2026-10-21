# AgentTrainer 2.47

AgentTrainer 2.47 keeps a stable signing identity across local rebuilds and signed DMG updates, so macOS can recognize the updated app and retain existing privacy grants.

## Stable signing

- Builds select an exact certificate fingerprint and exclude revoked certificates, including duplicate certificates with the same display name.
- The build saves a machine-local certificate and designated-requirement pin outside the repository. Clean builds and source updates reuse it.
- Missing signing keys, ambiguous certificates, and permission-identity changes stop the build instead of silently switching identities or using ad-hoc signing.
- Staged, installed, and packaged apps must pass strict signature checks and preserve the installed app's designated requirement.
- The signed DMG retains its signature when downloaded from GitHub. Install it at `/Applications/AgentTrainer.app` for subsequent updates.

A new Mac still needs permissions granted once. Building from source requires access to an appropriate signing key; private keys and machine-local signing pins are never included in the repository or source archive. This release uses Apple Development signing and is not notarized.

## Included app changes

This release also includes the behavioral-cloning-only workflow from 2.46: DAgger and reinforcement learning were removed, while native recordings and compatible Policy-v6 brains remain usable. See [the 2.46 notes](https://github.com/UnendlessGit/AgentTrainer/blob/v2.47/RELEASE_NOTES_2.46.md) for compatibility details.

## Validation

The functional suite passed 159 tests with the opt-in benchmark skipped. Signing regressions passed, covering revoked/ambiguous certificates, persistence across clean builds, missing keys, and identity-change rejection. Compact, default, and temporal-stress release benchmark profiles passed on the repeat run; the first run exceeded the default-profile augmentation timing threshold while background activity was present. No benchmark thresholds were changed.

Version 2.47, build 34. Requires Apple silicon and macOS 15 or later.
