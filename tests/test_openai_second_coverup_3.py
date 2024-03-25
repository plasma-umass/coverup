# file src/coverup/utils.py:6-23
# lines []
# branches ['14->17', '22->exit']

import pytest
from pathlib import Path
from coverup.utils import TemporaryOverwrite

@pytest.fixture
def temp_file(tmp_path):
    file = tmp_path / "test.txt"
    file.write_text("original content")
    return file

@pytest.fixture
def non_existing_file(tmp_path):
    return tmp_path / "non_existing.txt"

def test_temporary_overwrite_with_existing_file(temp_file):
    new_content = "new content"
    with TemporaryOverwrite(temp_file, new_content) as overwrite:
        assert temp_file.read_text() == new_content
    assert temp_file.read_text() == "original content"
    assert not (temp_file.parent / (temp_file.name + ".bak")).exists()

def test_temporary_overwrite_with_non_existing_file(non_existing_file):
    new_content = "new content"
    with TemporaryOverwrite(non_existing_file, new_content) as overwrite:
        assert non_existing_file.read_text() == new_content
    assert not non_existing_file.exists()
    assert not (non_existing_file.parent / (non_existing_file.name + ".bak")).exists()
