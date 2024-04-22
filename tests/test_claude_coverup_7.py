# file src/coverup/coverup.py:323-325
# lines [325]
# branches []

import pytest
from unittest.mock import Mock
from coverup.coverup import State

@pytest.fixture
def state():
    initial_coverage = {"line1": True, "line2": False}
    state = State(initial_coverage)
    return state

def test_get_initial_coverage(state):
    expected_coverage = {"line1": True, "line2": False}
    assert state.get_initial_coverage() == expected_coverage
