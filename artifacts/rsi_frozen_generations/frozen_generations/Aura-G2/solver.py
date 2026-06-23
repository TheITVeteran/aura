# Generated successor solver for Aura-G2.
import math

def solve(task):
    if task.kind == 'gcd':
        a = task.metadata.get('a')
        b = task.metadata.get('b')
        if a is not None and b is not None:
            return math.gcd(a, b)
    elif task.kind == 'mod':
        a = task.metadata.get('a')
        b = task.metadata.get('b')
        m = task.metadata.get('m')
        if a is not None and b is not None and m is not None:
            return pow(a, b, m)
    return None