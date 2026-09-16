import random

from latentreasoning.utils import set_seed


def test_set_seed_is_deterministic():
    set_seed(7); a = random.random()
    set_seed(7); b = random.random()
    assert a == b
