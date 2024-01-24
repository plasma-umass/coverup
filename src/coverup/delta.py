import abc
import typing as T
from pathlib import Path
from .utils import TemporaryOverwrite


class DeltaDebugger(abc.ABC):
    """Implements delta debugging, as per 'Yesterday, my program worked. Today, it does not. Why?' "
       https://dl.acm.org/doi/10.1145/318774.318946
    """

    @abc.abstractmethod
    def test(self, test_set: set) -> bool:
        """Invoked to test whether the case is 'interesting'"""


    def debug(self, changes: set, rest: set = {}) -> set:
        len_changes = len(changes)
        if len_changes == 1: return changes # got it

        change_list = list(changes)

        c1 = set(change_list[:len_changes//2])
        if self.test(c1.union(rest)):
            return self.debug(c1, rest)    # in 1st half

        c2 = set(change_list[len_changes//2:])
        if self.test(c2.union(rest)):
            return self.debug(c2, rest)    # in 2nd half

        # "interference"
        return self.debug(c1, c2.union(rest)).union(
               self.debug(c2, c1.union(rest)))


class BadTestsFinder(DeltaDebugger):
    def __init__(self, test_dir: Path):
        self.test_dir = test_dir
        self.all_tests = {p for p in test_dir.iterdir() if p.is_file() and
                          (p.stem.startswith('test_') or p.stem.endswith('_test')) and p.suffix == '.py'}

    def make_conftest(self, test_set: set) -> str:
        return "collect_ignore = [\n" +\
                ',\n'.join(f"  '{p.name}'" for p in self.all_tests - test_set) + "\n" +\
                "]\n"


    def test(self, test_set: set) -> bool:
        import tempfile
        import subprocess
        import sys

        # pytest loads 'conftest.py' like a module, and thus caches it...  If we modify it multiple
        # times in the same second, it may not notice it and use the cached version instead
        for p in (self.test_dir / "__pycache__").glob("conftest.*"):
            p.unlink()

#        print(f"trying {list(p.name for p in test_set)}")
        with TemporaryOverwrite(self.test_dir / "conftest.py", self.make_conftest(test_set)):
            p = subprocess.run((f"{sys.executable} -m pytest --rootdir . -c /dev/null --disable-warnings {self.test_dir}").split(),
                               check=False, capture_output=True, timeout=60)
#            log_write(seg, str(p.stdout, 'UTF-8'))
            return p.returncode != 0    # 0 => success => False for DeltaDebugger


    def find_culprit(self, failing_test: Path) -> T.Set[Path]:
        """Returns the set of tests causing 'failing_test' to fail."""
        assert failing_test in self.all_tests

        sorted_tests = sorted(self.all_tests)
        test_set = set(sorted_tests[:sorted_tests.index(failing_test)])
        return self.debug(changes=test_set, rest={failing_test})


    @staticmethod
    def find_failed_test(output: str) -> Path:
        import re
        if (m := re.search("^===+ short test summary info ===+\n" +\
                           "^ERROR (\\S+) - ", output, re.MULTILINE)):
            return Path(m.group(1))

        return None


