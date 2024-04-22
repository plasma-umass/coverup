# file src/coverup/coverup.py:166-182
# lines []
# branches ['170->177', '177->182']

import pytest
import re

# Assuming the original function is in coverup/coverup.py
from coverup.coverup import clean_error

# Test to cover branch 170->177
def test_clean_error_with_failures_section():
    error_message = "===== FAILURES ====\n" \
                    "_________________________ test_failure _________________________\n" \
                    "\n" \
                    "self = <test_module.TestClass object at 0x7f4b2d2c1f10>\n" \
                    "\n" \
                    "    def test_failure(self):\n" \
                    ">       assert 0\n" \
                    "E       assert 0\n" \
                    "\n" \
                    "test_module.py:10: AssertionError\n"
    expected_cleaned_error = "self = <test_module.TestClass object at 0x7f4b2d2c1f10>\n" \
                             "\n" \
                             "    def test_failure(self):\n" \
                             ">       assert 0\n" \
                             "E       assert 0\n" \
                             "\n" \
                             "test_module.py:10: AssertionError\n"
    assert clean_error(error_message) == expected_cleaned_error

# Test to cover branch 177->182
def test_clean_error_with_summary_info():
    error_message = "self = <test_module.TestClass object at 0x7f4b2d2c1f10>\n" \
                    "\n" \
                    "    def test_failure(self):\n" \
                    ">       assert 0\n" \
                    "E       assert 0\n" \
                    "\n" \
                    "test_module.py:10: AssertionError\n" \
                    "=== short test summary info ====\n" \
                    "FAILED test_module.py::TestClass::test_failure - assert 0\n"
    expected_cleaned_error = "self = <test_module.TestClass object at 0x7f4b2d2c1f10>\n" \
                             "\n" \
                             "    def test_failure(self):\n" \
                             ">       assert 0\n" \
                             "E       assert 0\n" \
                             "\n" \
                             "test_module.py:10: AssertionError\n"
    assert clean_error(error_message) == expected_cleaned_error
