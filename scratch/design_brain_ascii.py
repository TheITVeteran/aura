# Quick scratch script to test ASCII brain symmetry and layout

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
max_len = max(len(line) for line in lines)
print(f"Max line length: {max_len}")
for i, line in enumerate(lines, 1):
    print(f"{i:02d} | {line.ljust(max_len)} | length: {len(line)}")
