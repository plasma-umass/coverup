# file src/coverup/coverup.py:362-364
# lines [364]
# branches []

import pytest
from coverup.coverup import State

# Assuming the State class __init__ method requires an initial_coverage argument
# If not, the State class should be modified accordingly

@pytest.fixture
def clean_state():
    # Setup: create a new state instance with a dummy initial coverage
    initial_coverage = {'lines': set(), 'branches': set()}
    state = State(initial_coverage)
    yield state
    # Teardown: clean up any changes to avoid state pollution
    state.final_coverage = None

def test_set_final_coverage(clean_state):
    test_coverage = {'lines': {1, 2, 3}, 'branches': {4, 5, 6}}
    
    # Set the final coverage
    clean_state.set_final_coverage(test_coverage)
    
    # Assert that the final coverage is set correctly
    assert clean_state.final_coverage == test_coverage
