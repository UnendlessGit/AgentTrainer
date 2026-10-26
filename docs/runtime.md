# Runtime

Back to [architecture overview](../ARCHITECTURE.md).

Live inference, input execution, UI behavior, and updates in AgentTrainer 2.52. Timing features and policy decoding are described in [Model and training](model-and-training.md#explicit-timing).

## Runtime execution

`AgentRuntime` captures live frames, preprocesses them, and caches evaluated visual embeddings at Perception FPS. A separate Action FPS timer admits lightweight timing-conditioned policy work onto the same serialized MLX inference queue. New image work coalesces, and at most one head/execution is pending. This bounds queued work when inference is slower than capture.

Temporal history uses a fixed-capacity ring of reduced packed images, embeddings, controls, and timing. Missing startup slots use the current reduced image with zero controls and invalid timing. Idle capture keeps the original source-frame timestamp. The action timer checks staleness independently of pending inference. At 60 Action FPS, nominal execution spacing is approximately 16.7 ms; event timestamp precision does not provide sub-tick output scheduling.

`InputInjector` filters each action using saved output capabilities, current channel disables, profile key/button/modifier restrictions, demonstrated key codes, live cursor/keyboard permissions, and the control region unless full-Mac control is selected. It ignores predictions shorter than 146 values or containing non-finite values in the first 146 positions. Mode selection chooses absolute cursor or relative game-camera execution. Relative outputs are sensitivity-scaled, rounded, and posted as HID movement.

Executed events, including forced releases, feed separate action/perception `CausalControlTimeline` reducers under the injector lock. Commands blocked by permissions, restrictions, rounding, or clipping do not become fictitious control history. Synthetic events carry `agentTrainerSyntheticTag`; physical-input safety monitoring ignores that tag.

The prediction latch retains held-state actions between fresh predictions while consuming relative movement and scroll once. Permission/context/session revisions reject stale inference before posting. A context change clears history/predictions and releases old-context held controls before admitting new-context output. Panic, human takeover, and normal stop drain actions and release controls; teardown then drains capture/inference and late action callbacks before restart.

## UI and updates

The SwiftUI interface uses shared appearance settings, a custom top bar/sidebar, and reusable context controls. Decorative motion responds to Reduce Motion, user settings, and app activation. Charts have bounded histories; sizing displays use `NeuralInputSizing` and `ModelSizing`. Context catalog renames update recording snapshots; deletion removes matching assignments after UI confirmation, independently of saved-brain vocabularies.

`GitHubReleaseUpdater` requests releases from the configured GitHub repository, parses stable semantic versions, and offers newer versions. Downloads use HTTPS, approved hosts, and declared/observed size limits. Installation verifies the named DMG's SHA-256 entry, disk image, app bundle identifier/version, strict code signature, and designated requirement equality with the installed app. A helper acknowledges readiness before the app quits, then replaces the app using incoming/rollback copies on the target volume. Failed acknowledgement terminates the helper.
