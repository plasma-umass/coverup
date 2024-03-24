# file src/coverup/utils.py:26-46
# lines [30, 31, 33, 34, 35, 36, 37, 39, 40, 42, 44, 46]
# branches ['34->exit', '34->35', '36->37', '36->39', '39->40', '39->42']

import pytest
from coverup.utils import format_ranges

def test_format_ranges_with_negative_ranges():
    # Define the sets of lines and negative lines to pass to the function
    lines = {1, 2, 4, 5}
    negative = {3}

    # Call the function with the test data
    result = format_ranges(lines, negative)

    # Verify the result is as expected
    assert result == "1-2, 4-5", "The format_ranges function did not return the expected string"
