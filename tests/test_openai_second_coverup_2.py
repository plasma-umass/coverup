# file src/coverup/utils.py:26-46
# lines [40]
# branches ['39->40']

import pytest
from coverup.utils import format_ranges

def test_format_ranges_single_line():
    lines = {1, 2, 4}
    negative = {3}
    expected = "1-2, 4"
    assert format_ranges(lines, negative) == expected

def test_format_ranges_with_negative_gap():
    lines = {1, 2, 4, 5}
    negative = {3}
    expected = "1-2, 4-5"
    assert format_ranges(lines, negative) == expected

def test_format_ranges_with_negative_gap_yielding_single_line():
    lines = {1, 3, 5}
    negative = {2, 4}
    expected = "1, 3, 5"
    assert format_ranges(lines, negative) == expected
