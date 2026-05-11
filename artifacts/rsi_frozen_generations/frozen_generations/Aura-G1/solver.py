# Generated successor solver for Aura-G1.
import math

def solve(task):
    if task.kind == 'gcd':
        a = task.metadata['a']
        b = task.metadata['b']
        return math.gcd(a, b)
    return None