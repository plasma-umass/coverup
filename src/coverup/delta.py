import abc
from pathlib import Path


def _compact(test_set):
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
        if self.trace: self.trace(f"debug(changes={_compact(changes)}; rest={_compact(rest)})")

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
