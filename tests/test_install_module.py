import coverup.coverup as coverup
from test_prompt import MockSegment, pkg_fixture
import subprocess
import sys
from pathlib import Path
import pytest


DUMMY_MODULE='mymodule'


@pytest.fixture
def uninstall():
    yield
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", DUMMY_MODULE])


def test_install(pkg_fixture, monkeypatch, uninstall):
    log = ""
    def mock_log_write(seg, m: str) -> None:
        nonlocal log
        log += m

    monkeypatch.setattr("coverup.coverup.log_write", mock_log_write)
    monkeypatch.setattr("coverup.coverup.args",
        coverup.parse_args(["--package", "lib/ansible", "--tests", "tests", "--write-requirements-to", "reqs.txt"]),
        raising=False
    )

    missing = coverup.missing_imports(['mymodule'])
    assert missing

    ok = coverup.install_missing_imports(MockSegment(), missing)
    assert ok

    import importlib.metadata as md
    version = md.version(DUMMY_MODULE)

    assert DUMMY_MODULE in log
    assert version in log

    assert Path("reqs.txt").read_text() == f"{DUMMY_MODULE}=={version}\n"
