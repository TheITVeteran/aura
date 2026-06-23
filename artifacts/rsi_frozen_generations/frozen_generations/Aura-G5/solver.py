# Generated successor solver for Aura-G5.
def solve(task):
    kind = task.kind
    metadata = task.metadata

    if kind == 'compose':
        a = metadata['a']
        b = metadata['b']
        c = metadata['c']
        d = metadata['d']
        x = metadata['x']
        return c * (a * x + b) + d
    elif kind == 'gcd':
        a = metadata['a']
        b = metadata['b']
        return gcd(a, b)
    elif kind == 'mod':
        a = metadata['a']
        b = metadata['b']
        m = metadata['m']
        return pow(a, b, m)
    elif kind == 'palindrome':
        s = metadata['s']
        return s == s[::-1]
    elif kind == 'sort':
        arr = metadata['arr']
        return sorted(arr)
    else:
        return None

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a