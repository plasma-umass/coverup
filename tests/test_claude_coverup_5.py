# file src/coverup/coverup.py:333-338
# lines [335, 336, 337, 338]
# branches ['336->exit', '336->337']

import pytest
from unittest.mock import Mock
from coverup.coverup import State, Progress

@pytest.fixture
def state():
    return State(initial_coverage=0.0)

def test_set_progress_bar_with_bar(state):
    mock_bar = Mock(spec=Progress)
    state.usage = 'some_usage'
    state.counters = {'key': 'value'}

    state.set_progress_bar(mock_bar)

    assert state.bar == mock_bar
    mock_bar.update_usage.assert_called_once_with('some_usage')
    mock_bar.update_counters.assert_called_once_with({'key': 'value'})

def test_set_progress_bar_without_bar(state):
    state.set_progress_bar(None)

    assert state.bar is None
