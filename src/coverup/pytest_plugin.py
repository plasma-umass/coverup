import json
import typing as T
import pytest
import json
from _pytest.pathlib import Path


def pytest_addoption(parser):
    parser.addoption('--coverup-outcomes', action='store',
                     type=Path, default=None,
                     help='Path where to store execution outcomes')
    parser.addoption('--coverup-run-only', action='store',
                     type=Path, default=None,
                     help='Path to only module to execute')
    parser.addoption('--coverup-stop-after', action='store',
                     type=Path, default=None,
                     help='Stop execution after reaching this module')


class CoverUpPlugin:
    def __init__(self, config):
        self._rootpath = config.rootpath
        self._stop_after = config.getoption("--coverup-stop-after")
        self._outcomes_file = config.getoption("--coverup-outcomes")

        if self._stop_after: self._stop_after = self._stop_after.resolve()

        self._stop_now = False
        self._outcomes = {}

    def pytest_collectreport(self, report):
        if report.failed:
            path = self._rootpath / report.fspath
            if path not in self._outcomes:
                self._outcomes[path] = report.outcome

    def pytest_collection_modifyitems(self, config, items):
        if (run_only := config.getoption("--coverup-run-only")):
            run_only = run_only.resolve()

            selected = []
            deselected = []

            for item in items:
                if run_only == item.path:
                    selected.append(item)
                else:
                    deselected.append(item)

            items[:] = selected
            if deselected:
                config.hook.pytest_deselected(items=deselected)

    def pytest_runtest_protocol(self, item, nextitem):
        if self._stop_after == item.path:
            if not nextitem or self._stop_after != nextitem.path:
                self._stop_now = True

    def pytest_runtest_logreport(self, report):
        path = self._rootpath / report.fspath
        if path not in self._outcomes or report.outcome != 'passed':
            self._outcomes[path] = report.outcome

        if self._stop_now and report.when == 'teardown':
            pytest.exit(f"Stopped after {self._stop_after}")

    def write_outcomes(self):
        if self._outcomes_file:
            with self._outcomes_file.open("w") as f:
                json.dump({str(k): v for k, v in self._outcomes.items()}, f)


def pytest_configure(config):
    config._coverup_plugin = CoverUpPlugin(config)
    config.pluginmanager.register(config._coverup_plugin, 'coverup_plugin')


def pytest_unconfigure(config):
    if (plugin := getattr(config, "_coverup_plugin", None)):
        plugin.write_outcomes()
