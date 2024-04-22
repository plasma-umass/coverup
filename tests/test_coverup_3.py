# file src/coverup/coverup.py:298-300
# lines [300]
# branches []

import pytest
from coverup.coverup import get_required_modules

# Assuming module_available is a global variable in coverup.py that needs to be patched
module_available = {
    'module1': 0,
    'module2': 1,
    'module3': 0,
    'module4': 1
}

def test_get_required_modules(monkeypatch):
    # Patch the module_available dictionary to control the test environment
    monkeypatch.setattr('coverup.coverup.module_available', module_available)
    
    # Call the function under test
    missing_modules = get_required_modules()
    
    # Assert that the function returns the correct list of missing modules
    assert missing_modules == ['module1', 'module3'], "The function should return a list of modules with availability not equal to 1"
    
    # Clean up by undoing the monkeypatch after the test
    monkeypatch.undo()
