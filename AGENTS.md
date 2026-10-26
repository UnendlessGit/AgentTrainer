# App signing

- Every app build and installation must use a valid signing certificate. Never use ad-hoc signing.
- Preserve the existing signing identity and designated requirement, including the machine-local pin managed by `scripts/signing.sh`.
- Never delete or bypass the signing pin, switch signing identities, or reset macOS privacy grants to work around a signing failure.
- If the required certificate or private key is unavailable, report the problem instead of installing a differently signed app.
- Keep private keys and machine-local signing pins out of the repository and distribution archives.
- For signing/build changes, run `zsh scripts/test-signing.sh`, check shell syntax, and verify that installed and packaged apps retain the same designated requirement.
