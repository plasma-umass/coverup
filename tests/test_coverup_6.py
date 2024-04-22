# file src/coverup/coverup.py:357-359
# lines [359]
# branches []

import pytest
from coverup.coverup import State

# Assuming the State class has an __init__ method that requires an initial_coverage argument.
# Since the code for the State class is not provided, we will mock the coverage attribute.

def test_get_initial_coverage():
    expected_coverage = {'file1.py': {'lines': [1, 2, 3]}, 'file2.py': {'lines': [4, 5, 6]}}
    state = State(initial_coverage=expected_coverage)
    
    # Call the method we want to test
    coverage = state.get_initial_coverage()
    
    # Assert that the method returns the correct coverage
    assert coverage == expected_coverage, "The get_initial_coverage method did not return the expected coverage data"
