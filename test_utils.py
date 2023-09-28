import coverup
from hypothesis import given
import hypothesis.strategies as st


def test_format_ranges():
    assert "" == coverup.format_ranges({})
    assert "1-3" == coverup.format_ranges({1,2,3})
    assert "1-2, 4-6, 8" == coverup.format_ranges({1,2,4,5,6,8})


@given(st.integers(0, 10000), st.integers(1, 10000))
def test_format_ranges_is_sorted(a, b):
    b = a+b
    assert f"{a}-{b}" == coverup.format_ranges({i for i in range(a,b+1)})
