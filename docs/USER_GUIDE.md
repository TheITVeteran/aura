# Aura — User Guide

*Last reviewed against the tree: 2026-08-01.*

## Install
1. Download `Aura.dmg` from the releases page.
2. Drag `Aura.app` to your Applications folder.
3. Open it. The first-run wizard walks you through model selection,
   memory location, permissions, voice, and a fallback choice.

If you'd rather run from source (advanced):
```bash
git clone https://github.com/youngbryan97/aura
cd aura
make setup      # or: make setup-prod for a fail-closed install
make run        # foreground desktop launch
```

Full install detail, boot modes, and environment variables are in
[INSTALL.md](../INSTALL.md).

## Talk to Aura
- Open Aura. The launch screen names every organ that's still warming
  up: Core / Memory / Cortex / Voice / Autonomy. Once everything you
  need is ready, the chat input becomes active.
- Type a message and press Enter. If the answer is taking longer than
  usual, you'll see a thinking indicator with an estimated wait —
  Aura's local 32B can take 15–40 seconds for the first turn.

## Manage Memory
- The Memory tab lists scars, narrative arcs, episodic journal, and
  the Eternal Record. You can pin a memory (it survives reaping), drop
  a topic (Aura will stop bringing it up), or export the whole record
  as a tarball under Settings → Backup.

## Use Voice
- Voice input requires explicit permission per session. Click the mic
  icon. The first time you do this, macOS asks for microphone access.
- Voice output is on by default. Turn it off in Settings → Voice.

## Common Issues
| Symptom | Likely cause | Fix |
|---|---|---|
| Banner: "My local Cortex is offline" | 32B failed to load | Settings → Models → Reset cortex; check disk space. |
| "I'm under load right now" replies | RAM pressure > 90% | Close memory-heavy apps; Settings → Memory → Compact. |
| Voice button greyed out | Permission revoked | Settings → Permissions → grant microphone. |
| Chat input stays disabled | Boot still warming | Check the boot screen at the top — wait for Cortex: Ready. |
| Aura answers, but flatly | An organ was missing from the turn | The turn surface reports which cognitive organs engaged; a missing organ is treated as a defect, not a note. |
| "I can't do that right now" | A capability exists but is unavailable | She distinguishes not having a capability from not being able to use it right now, and will say which. |

If something is wrong at the runtime level rather than the UI level, run
`aura doctor`, and `aura doctor --bundle` to produce a redacted diagnostics
tarball. Every incident class in [runbooks/](runbooks/) is written against
fields that bundle emits.

## Update Aura

Updates run through the release train (`tools/release_train.py`), not through
a channel picker:

```bash
make update        # autostash → fast-forward-only pull → compile sanity check
make update-live   # the same, plus a smoke run and a relaunch of the live instance
make rollback      # return to the last recorded good point
make release-status
```

Every update records a rollback point before it touches anything, and a
failed compile or smoke check stops the train rather than leaving a
half-updated tree. `make update` is deliberately boring: it refuses to
merge, so a diverged local tree fails loudly instead of resolving itself.

## Uninstall
- Drag `Aura.app` to the trash. Your data stays at `~/.aura/`.
- To remove all data too: `rm -rf ~/.aura`.

For deeper docs see `docs/OPERATOR_GUIDE.md` and `docs/RESEARCH_GUIDE.md`.
