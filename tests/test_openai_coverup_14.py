# file src/coverup/coverup.py:254-256
# lines [256]
# branches []

import pytest
from unittest.mock import patch
from coverup.coverup import get_required_modules

# Assuming module_available is a global variable or accessible within the scope
# If it's not, you would need to mock or patch the appropriate object to provide it.

def test_get_required_modules():
    # Mock the module_available dictionary to include modules with different statuses
    mocked_module_available = {
        'module1': 0,  # Not available
        'module2': 1,  # Available
        'module3': 0,  # Not available
    }

    with patch('coverup.coverup.module_available', mocked_module_available):
        # Call the function under test
        result = get_required_modules()

    # Verify that the result only includes the modules that were not available
    assert 'module1' in result
    assert 'module2' not in result
    assert 'module3' in result

    # Verify that the length of the result is correct
    assert len(result) == 2

    # No cleanup is necessary as the patch context manager will automatically undo the patch after the block
