# file src/coverup/pytest_plugin.py:21-29
# lines [21, 22, 23, 24, 26, 28, 29]
# branches ['26->26', '26->28']

import pytest
from pathlib import Path
from coverup.pytest_plugin import CoverUpPlugin

# Mock configuration object
class MockConfig:
    def __init__(self, rootpath, stop_after=None, outcomes_file=None):
        self.rootpath = rootpath
        self._options = {
            "--coverup-stop-after": stop_after,
            "--coverup-outcomes": outcomes_file,
        }

    def getoption(self, name):
        return self._options.get(name)

@pytest.fixture
def mock_config(tmp_path):
    return MockConfig(rootpath=tmp_path)

@pytest.fixture
def mock_config_with_options(tmp_path):
    stop_after = tmp_path / "stop_after"
    outcomes_file = tmp_path / "outcomes"
    return MockConfig(rootpath=tmp_path, stop_after=stop_after, outcomes_file=outcomes_file)

def test_coverup_plugin_init_without_options(mock_config):
    plugin = CoverUpPlugin(mock_config)
    assert plugin._rootpath == mock_config.rootpath
    assert plugin._stop_after is None
    assert plugin._outcomes_file is None
    assert not plugin._stop_now
    assert plugin._outcomes == {}

def test_coverup_plugin_init_with_options(mock_config_with_options):
    plugin = CoverUpPlugin(mock_config_with_options)
    assert plugin._rootpath == mock_config_with_options.rootpath
    assert plugin._stop_after == mock_config_with_options.getoption("--coverup-stop-after").resolve()
    assert plugin._outcomes_file == mock_config_with_options.getoption("--coverup-outcomes")
    assert not plugin._stop_now
    assert plugin._outcomes == {}
