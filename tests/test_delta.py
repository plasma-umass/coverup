import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from coverup.delta import DeltaDebugger


class NumbersFinder(DeltaDebugger):
    def __init__(self, numbers: set):
        super().__init__(trace=print)
        self._numbers = numbers

    def test(self, testset: set, **kwargs) -> bool:
        return self._numbers.issubset(testset)


@given(st.integers(0, 50))
def test_single(culprit):
    nf = NumbersFinder({culprit})
    assert {culprit} == nf.debug(set(range(51)))


@given(st.lists(st.integers(0, 50), min_size=2).filter(lambda n: len(set(n)) == len(n)))
def test_multiple(nums):
    nf = NumbersFinder({*nums})
    assert {*nums} == nf.debug(set(range(51)))


@given(st.lists(st.integers(0, 50), min_size=2).filter(lambda n: len(set(n)) == len(n)))
def test_kwargs_passed_through(nums):
    class DD(NumbersFinder):
        def __init__(self, numbers: set):
            super().__init__({*nums})

        def test(self, testset: set, **kwargs) -> bool:
            assert 'foo' in kwargs
            return super().test(testset)

    dd = DD({*nums})
    assert {*nums} == dd.debug(set(range(51)), foo=True)
