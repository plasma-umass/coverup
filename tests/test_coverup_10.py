# file src/coverup/coverup.py:384-389
# lines [386, 388, 389]
# branches ['388->exit', '388->389']

import pytest
from coverup.coverup import State

# Assuming that the State class requires an 'initial_coverage' argument for the constructor
# and has attributes self.counters and self.bar that can be set after initialization

@pytest.fixture
def state_with_bar(mocker):
    initial_coverage = {}  # Replace with appropriate initial coverage if needed
    state = State(initial_coverage)
    state.counters = {'test': 0}
    state.bar = mocker.Mock()
    return state

@pytest.fixture
def state_without_bar():
    initial_coverage = {}  # Replace with appropriate initial coverage if needed
    state = State(initial_coverage)
    state.counters = {'test': 0}
    state.bar = None
    return state

def test_inc_counter_with_bar(state_with_bar):
    state_with_bar.inc_counter('test')
    assert state_with_bar.counters['test'] == 1
    state_with_bar.bar.update_counters.assert_called_once_with(state_with_bar.counters)

def test_inc_counter_without_bar(state_without_bar):
    state_without_bar.inc_counter('test')
    assert state_without_bar.counters['test'] == 1
    assert state_without_bar.bar is None
