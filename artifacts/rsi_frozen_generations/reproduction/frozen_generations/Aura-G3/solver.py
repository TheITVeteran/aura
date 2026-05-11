# Generated successor solver for Aura-G3.
import math

def solve(task):
    if task.kind == 'compose':
        a = task.metadata['a']
        b = task.metadata['b']
        c = task.metadata['c']
        d = task.metadata['d']
        x = task.metadata['x']
        return c * (a * x + b) + d
    elif task.kind == 'gcd':
        a = task.metadata['a']
        b = task.metadata['b']
        return math.gcd(a, b)
    elif task.kind == 'mod':
        a = task.metadata['a']
        b = task.metadata['b']
        m = task.metadata['m']
        return pow(a, b, m)
    else:
        return None