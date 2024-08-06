# file src/coverup/segment.py:36-56
# lines [46, 47]
# branches ['45->46', '46->47', '46->56']

import pytest
from coverup.segment import CodeSegment
from unittest.mock import mock_open, patch

@pytest.fixture
def code_segment():
    cs = CodeSegment(
        filename="fake_file.py",
        name="test_segment",
        begin=1,
        end=3,
        lines_of_interest=set(),
        missing_lines=set(),
        executed_lines=set(),
        missing_branches=set(),
        context=[], imports=[]
    )
    return cs

def test_get_excerpt_without_executed_lines(code_segment, monkeypatch):
    mock_file_content = "line1\nline2\nline3\n"
    monkeypatch.setattr("builtins.open", mock_open(read_data=mock_file_content))
    excerpt = code_segment.get_excerpt()
    expected_excerpt = "            line1\n            line2\n"
    assert excerpt == expected_excerpt
