"""An undeliberated action must be distinguishable from an unnecessary one.

`deliberation_worthy_action = False` is the value the executor uses BOTH for
"policy says this action needs no deliberation" and, before this change, for
"the pre-action cortex could not be imported" — logged at debug. So a cortex
that failed to load let every consequential action run undeliberated with no
signal that the check had not run.
"""
from __future__ import annotations

import inspect

from core.runtime import action_executor as module


def test_an_unimportable_cortex_is_recorded_as_lost_capability():
    source = inspect.getsource(module)
    block = source.split("except ImportError as exc:", 1)[1][:1400]
    assert "record_degradation" in block
    assert "SILENT_LOSS_OF_CAPABILITY" in block


def test_it_is_no_longer_only_a_debug_line():
    source = inspect.getsource(module)
    block = source.split("except ImportError as exc:", 1)[1][:1400]
    assert 'logger.debug("Pre-action cortex unavailable' not in block


def test_the_degradation_names_what_was_skipped():
    source = inspect.getsource(module)
    block = source.split("except ImportError as exc:", 1)[1][:1400]
    assert "WITHOUT pre-action" in block
    assert "severity=\"degraded\"" in block


def test_the_classification_is_importable():
    from core.runtime.errors import FallbackClassification

    assert FallbackClassification.SILENT_LOSS_OF_CAPABILITY
    assert module.FallbackClassification is FallbackClassification
