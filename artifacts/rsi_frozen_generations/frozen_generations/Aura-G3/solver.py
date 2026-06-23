# Generated successor solver for Aura-G3.
import math

def solve(task):
    kind = task.kind
    metadata = task.metadata
    
    if kind == 'compose':
        a = metadata.get('a', 0)
        b = metadata.get('b', 0)
        c = metadata.get('c', 0)
        d = metadata.get('d', 0)
        x = metadata.get('x', 0)
        return c * (a * x + b) + d
    elif kind == 'gcd':
        a = metadata.get('a', 0)
        b = metadata.get('b', 0)
        return math.gcd(a, b)
    elif kind == 'mod':
        a = metadata.get('a', 1)
        b = metadata.get('b', 1)
        m = metadata.get('m', 1)
        return pow(a, b, m)
    else:
        return None