# Script to center the ASCII brain by shifting leading spaces

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
shift = 7

centered_lines = []
for line in lines:
    # count leading spaces
    leading_spaces = len(line) - len(line.lstrip(' '))
    new_leading = max(0, leading_spaces - shift)
    centered_line = " " * new_leading + line.lstrip(' ')
    centered_lines.append(centered_line)

for idx, line in enumerate(centered_lines, 1):
    print(f"{idx:02d} |{line}|")

print("\n--- HTML Code block to copy ---")
print("\n".join(centered_lines))
