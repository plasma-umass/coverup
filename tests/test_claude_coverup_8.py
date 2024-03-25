# file src/coverup/coverup.py:328-330
# lines [330]
# branches []

import pytest
from unittest.mock import Mock
from coverup.coverup import State

@pytest.fixture
def state():
    initial_coverage = Mock()
    return State(initial_coverage)

def test_set_final_coverage(state):
    expected_coverage = {'file1.py': 80, 'file2.py': 90}
    state.set_final_coverage(expected_coverage)
    assert state.final_coverage == expected_coverage
