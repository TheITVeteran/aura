# Ambient voice, streamed speech, and media in the chat

What changed, why, and — at the end — what is not measured.

The goal was ordinary: launch Aura, start talking, and have her answer in the
same chat thread as everything else; ask her to play something and have it
play there; and when something fails, hear about it in her own words rather
than in a sentence a developer wrote months earlier.

Four things had to be true for that, and none of them were.

---

## 1. She could not speak until she had finished thinking

`core/voice/duplex/governed_stream.py` existed, was fully implemented, had a
complete test suite — and had **zero production callers**. The live path
blocked on the finished governed reply and then chunked the finished string.
So time-to-first-audio was still proportional to *total* reply length, and the
45-word spoken cap that compensated for it was still in place. That cap is
why spoken answers came out shallower than the same question typed: latency
management wearing the costume of a style choice.

The fix is not a new cognition lane. The governed pipeline already produces
its reply incrementally — `core/cognitive/state_machine.py` emits
`chat_stream_chunk` as tokens land. What was missing was a way for one
surface to receive **its own turn's** chunks: that telemetry topic is global,
and a voice lane that spoke the desktop's reply would be a far worse defect
than a slow one.

`core/conversation/reply_stream.py` binds a channel to the *async context* of
a turn. The publish site walks no registry and takes no lock — it asks what
channel this turn is running under. Context propagates into every await and
is copied into every task spawned underneath, so the binding follows the turn
and cannot leak sideways. A turn with nothing bound (every text turn today)
costs one ContextVar read.

Three properties are load-bearing:

- **Publishing never blocks cognition.** The queue is bounded and writes are
  non-blocking. A stalled consumer loses chunks and is told it lost them. The
  stream is an accelerator; the finished reply still arrives the ordinary way.
- **Each clause is governed before it is synthesised.** Most governance is
  clause-local — scaffold leakage, claims the clock contradicts, instruments
  that do not exist. Only whole-reply obligations need the end, and those
  bind the last clause.
- **The finished reply remains the authority.** Streamed text is
  pre-stabilisation. `reconcile()` compares what was actually *delivered*
  against what the turn stands behind, and a divergence is spoken rather than
  swallowed — the listener is holding a sentence Aura no longer stands
  behind, and silence there is how a hallucination gets established.

## 2. An open microphone hears the whole room

Removing the wake word is easy and makes the product unusable in an
afternoon: an always-on microphone hears the television, the other half of a
phone call, and the person you are actually talking to. Answering any of
those is worse than missing a turn — a missed turn costs a repeat, an
unwanted answer talks over your call.

`core/voice/duplex/addressivity.py` demotes the wake word rather than
deleting it. It becomes the strongest of several signals, which is what the
published work on device-directed speech detection converges on: acoustics,
the recognised text, the recogniser's uncertainty, and — the most useful term
for follow-ups — whether a conversation was already open.

The decision is **a ladder, not a score**. A score needs weights, and weights
nobody measured are opinions with decimal points; worse, a score is
unfalsifiable in the field, where the only report you get is "it answered when
it shouldn't have". Each rung is a rule someone can read and argue with, and
every verdict carries the reasons that produced it.

```
0  explicit    the user opened the floor (focused mode, push-to-talk)
1  named       her name, in a vocative position
2  open floor  she spoke seconds ago and this continues it
3  cold open   phrased as a request, long enough, near enough, room is quiet
   otherwise   silence
```

It fails closed, and it is **not a transcript filter**: a rejected utterance
is still transcribed and still shown, so the user can always see what she
heard and decide otherwise.

## 3. It interrupted people who paused to think

This is the single most common complaint about every voice assistant that
ships, and it is structural rather than a tuning miss. Deciding turn-end from
the transcript plus a silence timer discards the signal humans actually use,
which is intonation. Whisper punctuates from a language-model prior, so it
writes a full stop onto someone drawing breath — and a full stop is what makes
an endpointer pounce.

`core/voice/duplex/acoustic_endpoint.py` fits the pitch trend over the final
voiced stretch, in semitones so one threshold serves every speaker. The
safety argument is the asymmetry: it can only ever **extend** the wait. A
wrong reading costs a beat of latency and can never cost an interruption.

## 4. Media went somewhere else, and failures had a script

Ask any shipped assistant to play something and the best case is a hand-off:
a card that opens another app, a link, a new tab. Aura runs on the machine
that holds the file, so `core/media/` indexes what is here and
`interface/routes/media.py` serves it with real Range support — an endpoint
that only answers 200 with the whole file *plays*, which is why that defect
ships, but the scrubber does not work and a large file buffers entirely
before starting.

And when it is not here and there is no network, nothing composes a sentence.
`core/conversation/failure_context.py` records what was tried, what stopped
it, the probe's actual reading, and what is still possible; the turn reads
those facts and says it in her own words. `what still works` is part of the
record on purpose — a failure report listing only the failure invites
over-generalising "one host is unreachable" into "I'm offline".

---

## What is not measured

- **No end-to-end voice latency number.** What the tests establish is that a
  structural dependency is gone: TTFA no longer scales with total reply
  length. The figure on the live 32B under load has not been taken, and the
  numbers in `config.py` describe components rather than the whole path.
- **No addressivity accuracy.** The rungs are tested against transcripts
  chosen to be plausible, not sampled from real use. False-accept and
  false-reject rates in a room with a television on are unknown.
- **The acoustic thresholds are literature-shaped priors**, not readings
  taken on this host. What is established is the asymmetry that makes them
  safe to ship, not that they are correctly placed.
- **The addressivity gate has no null.** It has never been run against a
  control condition — an equivalent gate with its evidence shuffled — so the
  ladder's structure has not been shown to beat a simpler rule.

## Where to look

| Concern | File |
| --- | --- |
| Turn-scoped reply channel, reconciliation | `core/conversation/reply_stream.py` |
| Failures as facts | `core/conversation/failure_context.py` |
| Clause governance while streaming | `core/voice/duplex/governed_stream.py` |
| Was that meant for her | `core/voice/duplex/addressivity.py` |
| Pitch contour at turn end | `core/voice/duplex/acoustic_endpoint.py` |
| Session wiring | `core/voice/duplex/session.py` |
| Local media index, playback resolution | `core/media/` |
| Byte serving with Range | `interface/routes/media.py` |
| Ambient client, chat binding | `interface/static/voice_mode.js` |

Settings: `voice.auto_listen` turns ambient listening on; `voice.input_enabled`
and `voice.output_enabled` gate the lane entirely. `AURA_VOICE_AMBIENT`,
`AURA_VOICE_NAMES`, `AURA_VOICE_OPEN_FLOOR_S` and `AURA_MEDIA_ROOTS` tune the
rest.
