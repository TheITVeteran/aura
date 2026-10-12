"""The actual logits processor forces structural progress without altering quoted data."""
import json

import pytest

from core.brain.llm.a_shape_the_decoder_enforces import JsonState, enforce_json, feed

pytestmark = pytest.mark.unit


class Tokenizer:
    texts = ['prompt', '\n\n', '{', '"answer"', ':', '1', '}', '',
             '{"answer":' + '\n' * 64 + '1}', '{"answer":"' + ' ' * 80 + '"}']
    eos_token_id = 7
    all_special_ids = [7]

    def __len__(self):
        return len(self.texts)

    def decode(self, ids):
        return ''.join(self.texts[i] for i in ids)


@pytest.fixture
def cpu_mlx():
    mx = pytest.importorskip('mlx.core')
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        yield mx
    finally:
        mx.set_default_device(previous)


def test_a_newline_loving_sampler_reaches_a_complete_object_and_eos(cpu_mlx):
    mx, tokenizer = cpu_mlx, Tokenizer()
    held = enforce_json(tokenizer, require='object')
    assert held is not None
    tokens = [0]
    answer = [2, 3, 4, 5, 6, 7]       # {, "answer", :, 1, }, end: what the model means to say, in order
    for _ in range(60):
        logits = mx.full((len(tokenizer),), -100.)
        logits[1] = 100.  # the model always prefers another pair of newlines
        meant = next((ident for ident in answer if ident not in tokens), 7)
        logits[meant] = 20.  # and after them, the next token of what it means to say
        masked = held(mx.array(tokens), logits)
        ident = int(mx.argmax(masked).item())
        assert bool(mx.isfinite(masked[ident]).item())
        tokens.append(ident)
        if ident == tokenizer.eos_token_id:
            break
    assert tokens[-1] == tokenizer.eos_token_id, "formatting consumed the whole decode budget"
    assert json.loads(tokenizer.decode(tokens[1:-1])) == {"answer": 1}
    assert held.state['refused'] == 0


def test_spacing_masks_and_speculative_rewinds_preserve_their_own_prefix(cpu_mlx):
    mx, tokenizer = cpu_mlx, Tokenizer()
    held = enforce_json(tokenizer, require='object')
    logits = mx.zeros((len(tokenizer),))
    assert bool(mx.isfinite(held(mx.array([0]), logits)[1]).item())
    assert not bool(mx.isfinite(held(mx.array([0, 1, 1, 1, 1]), logits)[1]).item())
    # Same grammar state, different spacing history; the old mask cannot travel.
    assert bool(mx.isfinite(held(mx.array([0, 1]), logits)[1]).item())
    assert held.state['spacing'] == 2
    assert held.state['refused'] == 0


def test_whitespace_inside_a_value_stays_data_and_mixed_tokens_cannot_bypass_progress(cpu_mlx):
    mx, tokenizer = cpu_mlx, Tokenizer()
    held = enforce_json(tokenizer, require='object')
    masked = held(mx.array([0]), mx.zeros((len(tokenizer),)))
    assert not bool(mx.isfinite(masked[8]).item())
    assert bool(mx.isfinite(masked[9]).item())
    assert feed(JsonState(), tokenizer.texts[8]) is not None  # valid grammar, refused generation idling
    held(mx.array([0, 9]), mx.zeros((len(tokenizer),)))
    assert json.loads(tokenizer.texts[9]) == {"answer": ' ' * 80}
    assert held.state['refused'] == 0
