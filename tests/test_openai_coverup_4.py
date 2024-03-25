# file src/coverup/coverup.py:328-330
# lines [330]
# branches []

import pytest
from src.coverup.coverup import State

class MockState(State):
    def __init__(self):
        pass  # Mock the __init__ to not require initial_coverage

def test_set_final_coverage():
    state = MockState()
    coverage_data = {'lines': {1: True, 2: False, 3: True}}

    # Precondition: final_coverage should not be set
    assert not hasattr(state, 'final_coverage')

    # Action: Set the final coverage
    state.set_final_coverage(coverage_data)

    # Postcondition: final_coverage should be set to coverage_data
    assert hasattr(state, 'final_coverage')
    assert state.final_coverage == coverage_data
