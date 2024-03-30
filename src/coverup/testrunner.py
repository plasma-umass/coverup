from pathlib import Path
import tempfile
import subprocess
import pytest
import typing as T
import sys
import json
import re
import os
from .delta import DeltaDebugger, _compact
from .utils import subprocess_run


async def measure_test_coverage(*, test: str, tests_dir: Path, pytest_args='', log_write=None):
    """Runs a given test and returns the coverage obtained."""
    with tempfile.NamedTemporaryFile(prefix="tmp_test_", suffix='.py', dir=str(tests_dir), mode="w") as t:
        t.write(test)
        t.flush()

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as j:
            try:
                # -qq to cut down on tokens
                p = await subprocess_run([sys.executable, '-m', 'slipcover', '--branch', '--json', '--out', j.name,
                                          '-m', 'pytest'] + pytest_args.split() + ['-qq', '-x', '--disable-warnings', t.name],
                                         check=True, timeout=60)
                if log_write:
                    log_write(str(p.stdout, 'UTF-8', errors='ignore'))

                cov = json.load(j)
            finally:
                j.close()
                try:
                    os.unlink(j.name)
                except FileNotFoundError:
                    pass

    return cov["files"]


class TestRunnerError(Exception):
    def __init__(self, message: str, outcomes, returncode, stdout):
        super(TestRunnerError, self).__init__(message + f"; rc={returncode}")
        self.outcomes = outcomes
        self.returncode = returncode
        self.stdout = stdout


def measure_suite_coverage(*, tests_dir: Path, source_dir: Path, pytest_args='', trace=None):
    """Runs an entire test suite and returns the coverage obtained."""
    with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as j:
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as o:
            try:
                p = subprocess.run([sys.executable,
                                    '-m', 'slipcover', '--source', source_dir, '--branch', '--json', '--out', j.name,
                                    '-m', 'pytest'] + pytest_args.split() +\
                                    ['-p', 'coverup.pytest_plugin', '--coverup-outcomes', str(o.name),
                                     '-qq', '--disable-warnings', tests_dir],
                                   check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

                if trace:
                    trace(f"tests rc={p.returncode}")
                    trace(str(p.stdout, 'UTF-8', errors='ignore'))

                outcomes = None
                try:
                    outcomes = json.load(o)
                    if not tests_dir.is_absolute():
                        parent = tests_dir.resolve().parent
                        outcomes = {Path(p).relative_to(parent): o for p, o in outcomes.items()}
                    else:
                        outcomes = {Path(p): o for p, o in outcomes.items()}

                except:
                    pass

                if p.returncode not in (pytest.ExitCode.OK, pytest.ExitCode.NO_TESTS_COLLECTED):
                    raise TestRunnerError("Test suite execution failed", outcomes,
                                          p.returncode, str(p.stdout, 'UTF-8', errors='ignore'))

                return json.load(j)
            finally:
                j.close()
                o.close()

                try:
                    os.unlink(j.name)
                except FileNotFoundError:
                    pass

                try:
                    os.unlink(o.name)
                except FileNotFoundError:
                    pass


class BadTestFinderError(Exception):
    pass


class BadTestsFinder(DeltaDebugger):
    """Finds tests that cause other tests to fail."""
    def __init__(self, *, tests_dir: Path, pytest_args: str = '', trace = None, progress = None):
        super(BadTestsFinder, self).__init__(trace=trace)
        self.tests_dir = tests_dir

        def find_tests(p):
            for f in p.iterdir():
                if f.is_dir():
                    yield from find_tests(f)
                else:
                    # TODO filter according to pytest customization rules
                    if f.is_file() and (f.stem.startswith('test_') or f.stem.endswith('_test')) and f.suffix == '.py':
                        yield f

        self.all_tests = set(find_tests(self.tests_dir))
        self.pytest_args = pytest_args.split()
        self.progress = progress


    def run_tests(self, tests_to_run: set = None, stop_after: Path = None, run_only: Path = None) -> Path:
        """Runs the tests, by default all, returning the first one that fails, or None.
           Throws RuntimeError if unable to parse pytest's output.
        """

        test_set = tests_to_run if tests_to_run else self.all_tests

        def link_tree(src, dst):
            dst.mkdir(parents=True, exist_ok=True)

            for src_file in src.iterdir():
                dst_file = dst / src_file.name

                if src_file.is_dir():
                    link_tree(src_file, dst_file)
                else:
                    if (src_file not in self.all_tests) or (src_file in test_set):
                        dst_file.hardlink_to(src_file)

        assert self.tests_dir.parent != self.tests_dir # we need a parent directory

        if self.progress: self.progress(f"running {'1/' if run_only else ('up to ' if stop_after else '')}{len(test_set)} test(s)")
        with tempfile.TemporaryDirectory(dir=self.tests_dir.parent) as tmpdir:
            tmpdir = Path(tmpdir)
            link_tree(self.tests_dir, tmpdir)

            def to_tmpdir(p: Path):
                return tmpdir / (p.relative_to(self.tests_dir))

            with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as outcomes_f:
                try:
                    command = [sys.executable, "-m", "pytest"] + self.pytest_args + \
                              ['-qq', '--disable-warnings',
                               '-p', 'coverup.pytest_plugin', '--coverup-outcomes', str(outcomes_f.name)] \
                            + (['--coverup-stop-after', str(to_tmpdir(stop_after))] if stop_after else []) \
                            + (['--coverup-run-only', str(to_tmpdir(run_only))] if run_only else []) \
                            + [str(tmpdir)]
#                    if self.trace: self.trace(' '.join(command))
                    p = subprocess.run(command, check=False, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=2*60*60)

                    if p.returncode not in (pytest.ExitCode.OK, pytest.ExitCode.TESTS_FAILED,
                                            pytest.ExitCode.INTERRUPTED, pytest.ExitCode.NO_TESTS_COLLECTED):
                        self.trace(f"tests rc={p.returncode}")
                        p.check_returncode()

                    outcomes = json.load(outcomes_f)
                finally:
                    outcomes_f.close()
                    try:
                        os.unlink(outcomes_f.name)
                    except FileNotFoundError:
                        pass

            tmpdir = tmpdir.resolve()
            return {self.tests_dir / Path(p).relative_to(tmpdir): o for p, o in outcomes.items()}


    def test(self, test_set: set, **kwargs) -> bool:
        target_test = kwargs.get('target_test')
        outcomes = self.run_tests(test_set, stop_after=target_test, run_only=kwargs.get('run_only'))
        if self.trace: self.trace(f"failing={_compact(set(p for p, o in outcomes.items() if o == 'failed'))}")

        while target_test not in outcomes:
            if self.trace: self.trace(f"{target_test} not executed, trying without failing tests.")
            test_set -= set(p for p, o in outcomes.items() if o == 'failed')

            if not test_set:
                raise BadTestFinderError(f"Unable to narrow down cause of {target_test} failure.")

            outcomes = self.run_tests(test_set, stop_after=target_test, run_only=kwargs.get('run_only'))
            if self.trace: self.trace(f"failing={_compact(set(p for p, o in outcomes.items() if o == 'failed'))}")

        return outcomes[kwargs.get('target_test')] == 'failed'


    def find_culprit(self, failing_test: Path, *, test_set = None) -> T.Set[Path]:
        """Returns the set of tests causing 'failing_test' to fail."""
        assert failing_test in self.all_tests

        if self.trace: self.trace(f"checking if {failing_test} passes by itself...")
        if self.run_tests({failing_test}) == {failing_test}:
            if self.trace: self.trace("it doesn't!")
            return {failing_test}

        if self.trace: self.trace("checking if failure is caused by test collection code...")
        tests_to_run = set(test_set if test_set is not None else self.all_tests) - {failing_test}
        outcomes = self.run_tests(tests_to_run.union({failing_test}), run_only=failing_test)

        if outcomes[failing_test] == 'failed':
            if self.trace: print("Issue is in test collection code; looking for culprit...")
            return self.debug(changes=tests_to_run, rest={failing_test},
                              target_test=failing_test, run_only=failing_test)

        if self.trace: print("Issue is in test run code; looking for culprit...")
        changes = set(test_set if test_set is not None else self.all_tests) - {failing_test}
        return self.debug(changes=changes, rest={failing_test}, target_test=failing_test)
