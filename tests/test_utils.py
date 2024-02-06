import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from coverup import utils
import subprocess

def test_format_ranges():
    assert "" == utils.format_ranges(set(), set())
    assert "1-5" == utils.format_ranges({1,2,3,5}, set())
    assert "1-3, 5" == utils.format_ranges({1,2,3,5}, {4})
    assert "1-6, 10-11, 30" == utils.format_ranges({1,2,4,5,6,10,11,30}, {8,14,15,16,17})


@given(st.integers(0, 10000), st.integers(1, 10000))
def test_format_ranges_is_sorted(a, b):
    b = a+b
    assert f"{a}-{b}" == utils.format_ranges({i for i in range(a,b+1)}, set())


def test_lines_branches_do():
    assert "line 123 does" == utils.lines_branches_do({123}, set(), set())
    assert "lines 123-125, 199 do" == utils.lines_branches_do({123,124,125,199}, {128}, set())
    assert "branch 1->5 does" == utils.lines_branches_do(set(), set(), {(1,5)})
    assert "branches 1->2, 1->5 do" == utils.lines_branches_do(set(), set(), {(1,5),(1,2)})
    assert "line 123 and branches 1->exit, 1->2 do" == utils.lines_branches_do({123}, set(), {(1,2),(1,0)})
    assert "lines 123-124 and branch 1->exit do" == utils.lines_branches_do({123, 124}, set(), {(1,0)})
    assert "lines 123-125 and branches 1->exit, 1->2 do" == utils.lines_branches_do({123,124,125}, set(), {(1,2),(1,0)})

    # if a line doesn't execute, neither do the branches that touch it...
    assert "lines 123-125 do" == utils.lines_branches_do({123,124,125}, set(), {(123,124), (10,125)})


@pytest.mark.parametrize('check', [False, True])
@pytest.mark.asyncio
async def test_subprocess_run(check):
    p = await utils.subprocess_run(['/bin/echo', 'hi!'], check=check)
    assert p.stdout == b"hi!\n"


@pytest.mark.asyncio
async def test_subprocess_run_fails_checked():
    with pytest.raises(subprocess.CalledProcessError) as e:
        await utils.subprocess_run(['/usr/bin/false'], check=True)


@pytest.mark.asyncio
async def test_subprocess_run_fails_not_checked():
    p = await utils.subprocess_run(['/usr/bin/false'])
    assert p.returncode != 0


@pytest.mark.asyncio
async def test_subprocess_run_timeout():
    with pytest.raises(subprocess.TimeoutExpired) as e:
        await utils.subprocess_run(['/bin/sleep', '2'], timeout=1)
