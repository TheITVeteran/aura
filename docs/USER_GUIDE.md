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

Open her. The launch screen names every organ still warming up — Core,
Memory, Cortex, Voice, Autonomy — so you can see what's not ready yet
instead of guessing at a spinner. When the parts you need are up, the chat
input goes live.

Type and press Enter. If a reply is taking a while you'll get a thinking
indicator with an estimate. The first turn on a local 32B runs 15–40
seconds. That's the model loading and thinking, not something being wrong.

## Manage Memory

The Memory tab shows scars, narrative arcs, the episodic journal, and the
Eternal Record.

Three things you can do there:

- **Pin a memory** and it survives reaping. Nothing sweeps it later.
- **Drop a topic** and she stops bringing it up.
- **Export the whole record** as a tarball under Settings → Backup.

It's her memory, but it's your data. All of it comes out in one file.

## Use Voice

Voice input needs explicit permission each session — click the mic icon.
The first time, macOS will ask for microphone access.

Voice output is on by default. Settings → Voice turns it off.

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

Drag `Aura.app` to the trash. Your data stays at `~/.aura/` — deleting the
app does not delete what she remembers.

To remove that too:

```bash
rm -rf ~/.aura
```

That one is not reversible. Export from Settings → Backup first if there's
any chance you want it later.

For deeper docs see `docs/OPERATOR_GUIDE.md` and `docs/RESEARCH_GUIDE.md`.
