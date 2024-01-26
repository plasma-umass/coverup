import abc
import typing as T
from pathlib import Path
from .utils import TemporaryOverwrite

def compact(test_set):
    """Generates a (more) compact test name representation for debugging messages."""
    import re

    numbers = set()
    names = set()
    for t in test_set:
        if isinstance(t, Path):
            if (m := re.search("_(\\d+).py$", t.name)):
                numbers.add(int(m.group(1)))
            else:
                names.add(t.name)
        else:
            names.add(str(t))

    def get_ranges():
        it = iter(sorted(numbers))

        a = next(it, None)
        while a is not None:
            b = a
            while (n := next(it, None)) == b+1:
                b = n

            yield str(a) if a == b else f"{a}-{b}"
            a = n

    return ", ".join(list(get_ranges()) + sorted(names))


class DeltaDebugger(abc.ABC):
    """Implements delta debugging ("dd" algorithm), as per 'Yesterday, my program worked. Today, it does not. Why?' "
       https://dl.acm.org/doi/10.1145/318774.318946
    """
    def __init__(self, *, trace=None):
        self.trace = trace


    @abc.abstractmethod
    def test(self, test_set: set, **kwargs) -> bool:
        """Invoked to test whether the case is 'interesting'"""


    def debug(self, changes: set, rest: set = set(), **kwargs) -> set:
        if self.trace: self.trace(f"debug(changes={compact(changes)}; rest={compact(rest)})")

        len_changes = len(changes)
        if len_changes == 1: return changes # got it

        change_list = list(changes)

        c1 = set(change_list[:len_changes//2])
        if self.test(c1.union(rest), **kwargs):
            return self.debug(c1, rest)    # in 1st half

        c2 = set(change_list[len_changes//2:])
        if self.test(c2.union(rest), **kwargs):
            return self.debug(c2, rest)    # in 2nd half

        # "interference"
        return self.debug(c1, c2.union(rest), **kwargs).union(
               self.debug(c2, c1.union(rest), **kwargs))


class BadTestsFinder(DeltaDebugger):
    def __init__(self, test_dir: Path, *, pytest_args: str = '', trace = None):
        self.test_dir = test_dir
        self.all_tests = {p for p in test_dir.iterdir() if p.is_file() and
                          (p.stem.startswith('test_') or p.stem.endswith('_test')) and p.suffix == '.py'}
        self.pytest_args = pytest_args
        self.trace = trace


    def make_conftest(self, test_set: set) -> str:
        return "collect_ignore = [\n" +\
                ',\n'.join(f"  '{p.name}'" for p in self.all_tests - test_set) + "\n" +\
                "]\n"


    def run_tests(self, tests_to_run: set = None) -> Path:
        """Runs the tests, by default all, returning the first one that fails, or None.
           Throws RuntimeError if unable to parse pytest's output.
        """
        import tempfile
        import subprocess
        import sys
        import pytest

        test_set = tests_to_run if tests_to_run else self.all_tests

        # pytest loads 'conftest.py' like a module, and thus caches it...  If we modify it multiple
        # times in the same second, it may not notice it and use the cached version instead
        for p in (self.test_dir / "__pycache__").glob("conftest.*"):
            p.unlink()

        with TemporaryOverwrite(self.test_dir / "conftest.py", self.make_conftest(test_set)):
            p = subprocess.run((f"{sys.executable} -m pytest {self.pytest_args} -x -qq --disable-warnings --rootdir . {self.test_dir}").split(),
                               check=False, capture_output=True, timeout=60*60)

            if p.returncode == pytest.ExitCode.OK:
                if self.trace: self.trace(f"tests passed")
                return None

            if p.returncode != pytest.ExitCode.TESTS_FAILED:
                raise RuntimeError(f"Unexpected pytest return code ({p.returncode}). Output:\n" + str(p.stdout, 'UTF-8'))

            if not (first_failing := self.find_failed_test(str(p.stdout, 'UTF-8'))):
                raise RuntimeError("Unable to parse failing test out of pytest output. Output:\n" + str(p.stdout, 'UTF-8'))

            if self.trace: self.trace(f"tests rc={p.returncode} first_failing={first_failing}")

            return self.test_dir / first_failing


    def test(self, test_set: set, **kwargs) -> bool:
        if self.trace: self.trace(f"trying with {len(test_set)} test(s).")

        if first_failing := self.run_tests(test_set):
            # If given, check that it's target_test that failed: a different test may have failed.
            # If a test that comes after target_test fails, then this is really a success; but if it's a test
            # that comes before target_test, this may be "inconsistent" (in delta debugging terms).
            if not (target_test := kwargs.get('target_test')) or first_failing == target_test:
                return True # "interesting"/"reproduced"

        return False


    def find_culprit(self, failing_test: Path) -> T.Set[Path]:
        """Returns the set of tests causing 'failing_test' to fail."""
        assert failing_test in self.all_tests

# we unfortunately can't do this... the code that causes test(s) to fail may execute during pytest collection.
#        sorted_tests = sorted(self.all_tests)
#        test_set = set(sorted_tests[:sorted_tests.index(failing_test)])
#        assert self.test(test_set), "Test set doesn't fail!"

        return self.debug(changes=self.all_tests - {failing_test}, rest={failing_test}, target_test=failing_test)


    @staticmethod
    def find_failed_test(output: str) -> Path:
        import re
        if (m := re.search("^===+ short test summary info ===+\n" +\
                           "^(?:ERROR|FAILED) ([^\\s:]+)", output, re.MULTILINE)):
            return Path(m.group(1))

        return None


