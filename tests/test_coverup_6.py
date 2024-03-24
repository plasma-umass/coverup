# file src/coverup/coverup.py:350-355
# lines [352, 354, 355]
# branches ['354->exit', '354->355']

import pytest
from src.coverup.coverup import State
from unittest.mock import Mock

@pytest.fixture
def state_with_bar():
    initial_coverage = {}  # Assuming initial_coverage is a dictionary, adjust if it's a different type
    state = State(initial_coverage)
    state.counters = {'test': 0}
    state.bar = Mock()
    return state

def test_inc_counter_with_bar(state_with_bar):
    state_with_bar.inc_counter('test')
    assert state_with_bar.counters['test'] == 1
    state_with_bar.bar.update_counters.assert_called_once_with(state_with_bar.counters)
