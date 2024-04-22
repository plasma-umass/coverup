# file src/coverup/coverup.py:281-295
# lines [282, 283, 284, 286, 287, 288, 289, 290, 291, 292, 293, 295]
# branches ['283->284', '283->295']

import pytest
import subprocess
import sys
from unittest.mock import patch, MagicMock
from coverup.coverup import install_missing_imports, CodeSegment

@pytest.fixture
def code_segment():
    return CodeSegment(
        'example_code', 'example_path', 
        begin=0, end=0, 
        lines_of_interest=set(), 
        missing_lines=set(), 
        executed_lines=set(), 
        missing_branches=set(), 
        context={}
    )

@pytest.fixture
def mock_subprocess_run_success():
    with patch('subprocess.run') as mock_run:
        mock_run.return_value = subprocess.CompletedProcess(args=[], returncode=0)
        yield mock_run

@pytest.fixture
def mock_subprocess_run_failure():
    with patch('subprocess.run') as mock_run:
        mock_run.side_effect = subprocess.CalledProcessError(
            returncode=1, cmd=[], output=b'An error occurred'
        )
        yield mock_run

@pytest.fixture
def mock_log_write():
    with patch('coverup.coverup.log_write') as mock_log:
        yield mock_log

def test_install_missing_imports_success(code_segment, mock_subprocess_run_success, mock_log_write):
    modules = ['nonexistent_module']
    result = install_missing_imports(code_segment, modules)
    assert result == True
    mock_subprocess_run_success.assert_called_once_with(
        [f"{sys.executable}", "-m", "pip", "install", "nonexistent_module"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
    )
    mock_log_write.assert_called_with(code_segment, 'Installed module nonexistent_module')

def test_install_missing_imports_failure(code_segment, mock_subprocess_run_failure, mock_log_write):
    modules = ['nonexistent_module']
    result = install_missing_imports(code_segment, modules)
    assert result == False
    mock_subprocess_run_failure.assert_called_once_with(
        [f"{sys.executable}", "-m", "pip", "install", "nonexistent_module"],
        check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60
    )
    mock_log_write.assert_called_with(code_segment, 'Unable to install module nonexistent_module:\nAn error occurred')
