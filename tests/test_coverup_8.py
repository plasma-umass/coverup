# file src/coverup/coverup.py:375-381
# lines [377, 378, 380, 381]
# branches ['377->378', '377->380', '380->exit', '380->381']

import pytest
from coverup.coverup import State

class MockBar:
    def __init__(self):
        self.updated_usage = None

    def update_usage(self, usage):
        self.updated_usage = usage

@pytest.fixture
def state():
    initial_coverage = {'token1': 0, 'token2': 0}
    return State(initial_coverage)

@pytest.fixture
def state_with_bar():
    initial_coverage = {'token1': 0, 'token2': 0}
    state = State(initial_coverage)
    state.bar = MockBar()
    return state

def test_add_usage_without_bar(state):
    state.usage = {'token1': 1, 'token2': 2}
    additional_usage = {'token1': 3, 'token2': 4}
    state.add_usage(additional_usage)
    assert state.usage['token1'] == 4
    assert state.usage['token2'] == 6

def test_add_usage_with_bar(state_with_bar):
    state_with_bar.usage = {'token1': 1, 'token2': 2}
    additional_usage = {'token1': 3, 'token2': 4}
    state_with_bar.add_usage(additional_usage)
    assert state_with_bar.usage['token1'] == 4
    assert state_with_bar.usage['token2'] == 6
    assert state_with_bar.bar.updated_usage == state_with_bar.usage
