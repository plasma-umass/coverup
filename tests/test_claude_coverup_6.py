# file src/coverup/coverup.py:590-594
# lines [591, 592, 593, 594]
# branches []

import sys
from pathlib import Path
from unittest.mock import patch
import os
from coverup import coverup

def test_add_to_pythonpath(tmp_path):
    source_dir = tmp_path / "source"
    source_dir.mkdir()

    with patch.dict('os.environ', {'PYTHONPATH': '/existing/path'}):
        coverup.add_to_pythonpath(source_dir)
        assert str(source_dir.parent) in sys.path[0]
        assert os.environ['PYTHONPATH'] == f"{str(source_dir.parent)}:/existing/path"

    with patch.dict('os.environ', clear=True):
        coverup.add_to_pythonpath(source_dir)
        assert str(source_dir.parent) in sys.path[0]
        assert os.environ['PYTHONPATH'] == str(source_dir.parent)
