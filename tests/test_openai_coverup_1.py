# file src/coverup/coverup.py:453-457
# lines []
# branches ['456->456']

import pytest
from coverup.coverup import extract_python

def test_extract_python_without_code_block():
    with pytest.raises(RuntimeError) as exc_info:
        extract_python("This is a response without a Python code block.")
    assert "Unable to extract Python code from response" in str(exc_info.value)
