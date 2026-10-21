# AgentTrainer builds

- Read `DEVELOPMENT_GUIDE.md` before changing the app.
- Use `./build.sh` to rebuild, sign, package, and update the installed app. Keep the default `/Applications/AgentTrainer.app` path unless the user requests another location.
- Preserve the machine-local signing pin managed by `scripts/signing.sh`. Never delete it, reset macOS privacy grants, switch signing identities, or use ad-hoc signing to work around a build failure. Report an unavailable certificate/key instead.
- Quit the app normally before replacing it so recordings save and training pauses. Never force-kill it to rebuild.
- For signing/build changes, run `zsh scripts/test-signing.sh`, check shell syntax, and verify the installed and packaged apps retain the same designated requirement. Do not include machine-local signing pins or private keys in source archives.
