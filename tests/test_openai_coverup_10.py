# file src/coverup/coverup.py:341-347
# lines [343, 344, 346, 347]
# branches ['343->344', '343->346', '346->exit', '346->347']

import pytest
from unittest.mock import Mock
from src.coverup.coverup import State

@pytest.fixture
def state_with_bar():
    initial_coverage = {'token1': 0, 'token2': 0}
    state = State(initial_coverage)
    state.usage = {'token1': 0, 'token2': 0}
    state.bar = Mock()
    return state

def test_add_usage_with_bar(state_with_bar):
    additional_usage = {'token1': 1, 'token2': 2}
    state_with_bar.add_usage(additional_usage)
    
    assert state_with_bar.usage['token1'] == 1
    assert state_with_bar.usage['token2'] == 2
    state_with_bar.bar.update_usage.assert_called_once_with({'token1': 1, 'token2': 2})
