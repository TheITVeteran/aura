"""Generated successor solver for Aura-G3."""
from __future__ import annotations

import math

HANDLERS = ['compose', 'gcd', 'mod']

def solve(task):
    meta = task.metadata
    if task.kind == 'gcd' and 'gcd' in HANDLERS:
        return math.gcd(int(meta['a']), int(meta['b']))
    if task.kind == 'mod' and 'mod' in HANDLERS:
        return pow(int(meta['a']), int(meta['b']), int(meta['m']))
    if task.kind == 'compose' and 'compose' in HANDLERS:
        x = int(meta['x'])
        return int(meta['c']) * (int(meta['a']) * x + int(meta['b'])) + int(meta['d'])
    if task.kind == 'sort' and 'sort' in HANDLERS:
        return sorted(list(meta['arr']))
    if task.kind == 'palindrome' and 'palindrome' in HANDLERS:
        s = str(meta['s'])
        return s == s[::-1]
    return None
