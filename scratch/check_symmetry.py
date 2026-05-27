# Symmetry validator for our ASCII brain

brain_art = """
                         .  .:::..      ..:::.  .
                      .::'  .::'::..  ..::'::.  `::.
                    .::'   .::'   `::::'   `::.   `::.
                   .::'    :: [Unified Will] ::    `::.
                  ::'      ::      [::]      ::      `::
                 ::'       `::.   .::::.   .::'       `::
                ::'          `::..::'  `::..::'          `::
               ::  [Memory]     `::'    `::'     [Cortex]  ::
               ::   .::..        ::      ::        ..::.   ::
               ::  ::'  `::.     ::  [Φ] ::     .::'  `::  ::
               ::  :: [Affect]   ::      ::   [Phi Core] ::::
               `:: `::[Engine]   ::.    .::   [Cortex32B]::'
                `::. `::..  ..::'  `::::'  `::..  ..::' .::'
                  `::.  `::::'   [Substrate]   `::::'  .::'
                    `::..  `::.    [64-LTC]    .::'  ..::'
                       `::.. `::.            .::' ..::'
                          `::..`::.  [Stem]  .::'..::'
                             `:::[Brainstem]:::'
                               ::[Reflex Lane]::
                               `::.        .::'
                                 `::..  ..::'
                                    `::::'
"""

lines = [line for line in brain_art.splitlines() if line]
width = 61

# Pad all lines to width
padded_lines = [line.ljust(width) for line in lines]

def get_mirror_char(c):
    mirrors = {
        '(': ')', ')': '(',
        '[': ']', ']': '[',
        '{': '}', '}': '{',
        '<': '>', '>': '<',
        '/': '\\', '\\': '/',
        "'": '`', '`': "'"
    }
    return mirrors.get(c, c)

print("Line | Left Side (original) | Right Side (mirrored) | Difference")
print("-" * 75)
for idx, line in enumerate(padded_lines, 1):
    left = line[:width//2]
    right = line[width//2 + 1:]
    
    # Mirror the right side
    mirrored_right = "".join(get_mirror_char(c) for c in reversed(right))
    
    # Calculate differences
    diffs = []
    for i, (l_char, r_char) in enumerate(zip(left, mirrored_right)):
        if l_char != r_char:
            diffs.append(f"col {i}: '{l_char}' vs '{r_char}'")
            
    diff_str = "; ".join(diffs[:3])
    if len(diffs) > 3:
        diff_str += f" (+{len(diffs)-3} more)"
    
    status = "OK" if not diffs else "MISMATCH"
    print(f"{idx:02d}   | {left} | {mirrored_right} | {status} - {diff_str}")
