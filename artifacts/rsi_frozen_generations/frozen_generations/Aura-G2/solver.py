# Generated successor solver for Aura-G2.
import math

class Task:
    def __init__(self, kind, metadata):
        self.kind = kind
        self.metadata = metadata

def solve(task):
    if task.kind == 'gcd':
        a = task.metadata.get('a', 0)
        b = task.metadata.get('b', 0)
        return math.gcd(a, b)
    elif task.kind == 'mod':
        a = task.metadata.get('a', 0)
        b = task.metadata.get('b', 0)
        m = task.metadata.get('m', 0)
        return pow(a, b, m)
    else:
        return None