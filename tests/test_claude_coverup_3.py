# file src/coverup/utils.py:6-23
# lines [9, 10, 11, 14, 15, 17, 18, 21, 22, 23]
# branches ['14->15', '14->17', '22->exit', '22->23']

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from coverup.utils import TemporaryOverwrite

@pytest.fixture
def temp_dir():
    with TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)

def test_temporary_overwrite(temp_dir, monkeypatch):
    # Create a temporary file
    file_path = temp_dir / "test_file.txt"
    file_path.write_text("Initial content")

    # Mock the Path.exists method
    monkeypatch.setattr(Path, "exists", lambda self: True)

    # Test the TemporaryOverwrite context manager
    new_content = "New content"
    with TemporaryOverwrite(file_path, new_content):
        # Verify that the file content was overwritten
        assert file_path.read_text() == new_content

    # Verify that the file was restored to its original content
    assert file_path.read_text() == "Initial content"

    # Clean up the