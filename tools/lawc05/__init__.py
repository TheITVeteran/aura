"""LAWC-05: what code can and cannot do to physical law, computed.

Three modules, in order of how much they constrain each other:

- :mod:`nogo`     the energy bound on shifting an ambient constant, with the
                  reachability check the pseudo-inverse would otherwise hide
- :mod:`frontier` the sourcing bound, which is what actually binds in the
                  regime a real instrument could measure
- :mod:`lawfield` a copy universe where law IS a writable field, simulated
                  honestly enough that its dispersion can be recovered blind
"""

from . import frontier, nogo

__all__ = ["frontier", "nogo"]
