# file src/coverup/coverup.py:323-325
# lines [325]
# branches []

import pytest
from src.coverup.coverup import State

@pytest.fixture
def state_with_coverage():
    state = State(initial_coverage={'initial': 100})
    yield state
    # Cleanup code if necessary

def test_get_initial_coverage(state_with_coverage):
    expected_coverage = {'initial': 100}
    assert state_with_coverage.get_initial_coverage() == expected_coverage
