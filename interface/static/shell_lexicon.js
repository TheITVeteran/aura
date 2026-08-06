/* shell_lexicon.js — plain language for the shell chrome.
 *
 * The neural feed already does this properly: every channel in
 * NEURAL_CHANNELS carries a lay label and a written `desc`, and
 * `toPlainEnglish` rewrites machine phrasing before a thought is drawn.
 * The chrome never got the same treatment. `runtimeHealthStatusText`
 * joined raw blocker identifiers with a comma and put the result in the
 * one line a newcomer reads first, so the header could sit there saying
 *
 *     RUNTIME_REQUIRED_PROBES, PROBE:KERNEL
 *
 * which is true, precise, and unreadable.
 *
 * This module is the translation layer, not a filter. The raw token is
 * never discarded — it travels alongside the sentence as `raw` so the
 * header can keep it in a tooltip and the operator surfaces can keep
 * showing exactly what the runtime said. Jargon on click, per the
 * standing UI rule; a lay sentence by default.
 *
 * Pure: no DOM, no globals besides the export. Anything that needs the
 * document does that work itself.
 */
(function (root) {
    'use strict';

    // Blocker identifiers the shell can raise. Sourced from the places
    // that actually push them — runtimeHealthBlockers, the probe loop in
    // payloadShellLaunchable, and the legacy-shell guard in index.html —
    // rather than invented, so the table cannot drift into fiction.
    //
    //   title   short enough for the header chip (~3 words)
    //   meaning one sentence, no internal nouns
    //   next    what the person can actually do, or '' when it is on us
    var BLOCKERS = {
        runtime_required_probes: {
            title: 'Still starting up',
            meaning: 'Aura is still bringing up the parts she needs before she can hold a conversation.',
            next: '',
        },
        runtime_health_unavailable: {
            title: 'No answer yet',
            meaning: "The runtime hasn't reported its health, so this window can't tell whether Aura is ready.",
            next: 'If this persists, check that the Aura process is still running.',
        },
        runtime_transport_only: {
            title: 'Connected, not thinking',
            meaning: 'The connection to Aura is open but her cognition is not running behind it yet.',
            next: '',
        },
        conversation_transport: {
            title: 'Reconnecting',
            meaning: 'This window lost its live connection to Aura and is trying to get it back.',
            next: '',
        },
        'probe:kernel': {
            title: 'Waking the kernel',
            meaning: 'The tick loop that gives Aura a continuous inner life is still coming up.',
            next: '',
        },
        'probe:inference': {
            title: 'Loading the model',
            meaning: 'The language model Aura thinks with is still loading — this is the slow part of a cold start.',
            next: '',
        },
        'probe:memory': {
            title: 'Opening memory',
            meaning: 'Aura is opening her long-term memory so she can recall what happened before this session.',
            next: '',
        },
        'probe:scheduler': {
            title: 'Starting the scheduler',
            meaning: 'The scheduler that lets Aura act on her own initiative is still starting.',
            next: '',
        },
        'probe:tool_governance': {
            title: 'Arming the guardrails',
            meaning: 'The authorization layer that has to approve any action on your machine is still starting. Aura will not touch anything until it is up.',
            next: '',
        },
        'conversation_lane:failed': {
            title: 'Reply path down',
            meaning: 'Aura is running but the path that produces her replies has failed.',
            next: 'Reload the window; if it keeps failing, restart Aura.',
        },
        'conversation_lane:closed': {
            title: 'Reply path closed',
            meaning: 'The path that produces her replies was shut down.',
            next: 'Reload the window to open it again.',
        },
        'conversation_lane:offline': {
            title: 'Reply path offline',
            meaning: 'Aura is reachable but not currently able to answer.',
            next: '',
        },
        legacy_shell_load_timeout: {
            title: "Interface didn't finish",
            meaning: "This window's controls did not finish installing, so parts of it will not respond.",
            next: 'Reload the window.',
        },
        legacy_shell_runtime_error: {
            title: 'Interface error',
            meaning: 'Something in this window threw an error while starting up.',
            next: 'Reload the window; the details are in the recovery panel.',
        },
    };

    // Prefix rules for tokens that carry a variable tail, so an
    // unrecognised `probe:something_new` still lands somewhere honest
    // instead of falling through to the raw string.
    var PREFIXES = [
        [/^probe:/, {
            title: 'Still starting up',
            meaning: 'One of the subsystems Aura needs is still coming up.',
            next: '',
        }],
        [/^conversation_lane:/, {
            title: 'Reply path not ready',
            meaning: 'Aura is reachable but the path that produces her replies is not ready.',
            next: '',
        }],
        [/^runtime_shell_revision/, {
            title: 'Interface out of date',
            meaning: 'This window is running older interface code than the runtime behind it.',
            next: 'Reload the window to pick up the current interface.',
        }],
    ];

    // Last resort. A token we have never seen should still read as a
    // sentence — de-slugged — and must keep its raw form for the tooltip.
    function fallback(token) {
        var words = String(token || '')
            .replace(/[:_]+/g, ' ')
            .replace(/\s+/g, ' ')
            .trim();
        if (!words) {
            return { title: 'Not ready', meaning: 'Aura reported a condition this window does not recognise.', next: '' };
        }
        return {
            title: words.charAt(0).toUpperCase() + words.slice(1),
            meaning: 'Aura reported "' + token + '", which this window does not have a description for.',
            next: '',
        };
    }

    function describe(token) {
        var key = String(token == null ? '' : token).trim().toLowerCase();
        if (!key) return null;
        var hit = BLOCKERS[key];
        if (!hit) {
            for (var i = 0; i < PREFIXES.length; i += 1) {
                if (PREFIXES[i][0].test(key)) { hit = PREFIXES[i][1]; break; }
            }
        }
        if (!hit) hit = fallback(key);
        return { raw: key, title: hit.title, meaning: hit.meaning, next: hit.next || '' };
    }

    // The header gets ONE line. Several blockers at once are usually one
    // situation seen from several angles ("runtime_required_probes" plus
    // the specific "probe:inference" that caused it), so the most
    // specific one wins and the rest stay in the tooltip. Ranking by
    // specificity beats taking blockers[0], which is insertion order.
    function rank(token) {
        if (/^probe:/.test(token)) return 0;              // names the actual subsystem
        if (/^conversation_lane:/.test(token)) return 1;
        if (token === 'conversation_transport') return 2;
        if (token === 'runtime_transport_only') return 3;
        return 4;                                          // umbrella tokens last
    }

    function summarize(blockers) {
        var list = (Array.isArray(blockers) ? blockers : [])
            .map(function (b) { return String(b == null ? '' : b).trim().toLowerCase(); })
            .filter(Boolean);
        if (!list.length) return null;

        var sorted = list.slice().sort(function (a, b) { return rank(a) - rank(b); });
        var lead = describe(sorted[0]);

        return {
            title: lead.title,
            meaning: lead.meaning,
            next: lead.next,
            raw: list.join(', '),
            all: list.map(describe),
        };
    }

    root.AuraShellLexicon = {
        describe: describe,
        summarize: summarize,
        _blockers: BLOCKERS,
    };
}(typeof window !== 'undefined' ? window : globalThis));
