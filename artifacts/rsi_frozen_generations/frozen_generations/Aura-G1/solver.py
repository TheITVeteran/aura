# Generated successor solver for Aura-G1.
import math

def solve(task):
    if task.kind == 'gcd':
        a = task.metadata.get('a')
        b = task.metadata.get('b')
        if isinstance(a, int) and isinstance(b, int):
            return math.gcd(a, b)
    return None