# file src/coverup/coverup.py:367-372
# lines [369, 370, 371, 372]
# branches ['370->exit', '370->371']

import pytest
from coverup.coverup import State

class MockProgress:
    def __init__(self):
        self.usage_updated = False
        self.counters_updated = False

    def update_usage(self, usage):
        self.usage_updated = True

    def update_counters(self, counters):
        self.counters_updated = True

@pytest.fixture
def state():
    initial_coverage = {}  # Assuming initial_coverage is a dictionary, adjust if necessary
    return State(initial_coverage)

@pytest.fixture
def progress_bar():
    return MockProgress()

def test_set_progress_bar_with_none(state):
    state.bar = 'existing_bar'  # Set an initial value to ensure it's removed
    state.set_progress_bar(None)
    assert state.bar is None, "State 'bar' attribute should be None when set to None"

def test_set_progress_bar_with_mock(state, progress_bar):
    state.usage = 'usage'
    state.counters = 'counters'
    state.set_progress_bar(progress_bar)
    assert state.bar is progress_bar, "State 'bar' attribute should be set to the progress bar instance"
    assert progress_bar.usage_updated, "Progress bar's update_usage method should be called with state's usage"
    assert progress_bar.counters_updated, "Progress bar's update_counters method should be called with state's counters"
