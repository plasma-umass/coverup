import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from coverup.delta import DeltaDebugger, BadTestsFinder


class NumbersFinder(DeltaDebugger):
    def __init__(self, numbers: set):
        self._numbers = numbers

    def test(self, testset: set) -> bool:
        return self._numbers.issubset(testset)


@given(st.integers(0, 50))
def test_single(culprit):
    nf = NumbersFinder({culprit})
    assert {culprit} == nf.debug(set(range(51)))


@given(st.lists(st.integers(0, 50), min_size=2).filter(lambda n: len(set(n)) == len(n)))
def test_multiple(nums):
    nf = NumbersFinder({*nums})
    assert {*nums} == nf.debug(set(range(51)))


def test_make_conftest(tmpdir):
    test_dir = Path(tmpdir)

    def seq2p(seq):
        return test_dir / f"test_coverup_{seq}.py"

    all_tests = {seq2p(seq) for seq in range(10)}.union({test_dir / "not_our_test.py"})
    for t in all_tests:
        t.touch()

    (test_dir / "something_else.py").touch()
    (test_dir / "not_a_test.txt").touch()
    (test_dir / "test_me_not.log").touch()
    (test_dir / "README.md").touch()

    test_set = {seq2p(seq) for seq in {3, 5, 7}}

    btf = BadTestsFinder(test_dir)
    conftest = btf.make_conftest(test_set)

    vars = dict()
    exec(compile(conftest,'','exec'), vars)
    assert set(vars['collect_ignore']) == {str(p.name) for p in (all_tests - test_set)}


@pytest.mark.parametrize("existing_testconf", [True, False])
def test_find_culprit(tmpdir, existing_testconf):
    test_dir = Path(tmpdir)

    def seq2p(seq):
        return test_dir / f"test_coverup_{seq}.py"

    if existing_testconf:
        (test_dir / "testconf.py").write_text("# my precious")

    all_tests = {seq2p(seq) for seq in range(10)}
    for t in all_tests:
        t.write_text('def test_foo():\n    pass')

    culprit = seq2p(3)
    culprit.write_text('bad!!')

    seq2p(7).write_text('also bad!!') # but doesn't matter, as 6 is where we cut off below

    btf = BadTestsFinder(test_dir)
    assert {seq2p(3)} == btf.find_culprit(seq2p(6))

    if existing_testconf:
        assert (test_dir / "testconf.py").read_text() == "# my precious"
    else:
        assert not (test_dir / "testconf.py").exists()


def test_find_failed_test_collecting():
    # from python3 -m pytest --rootdir . --disable-warnings -c /dev/null -qq -x coverup-tests
    output = """\
========================================================================================== ERRORS ==========================================================================================
____________________________________________________________________ ERROR collecting coverup-tests/test_coverup_906.py ____________________________________________________________________
coverup-tests/test_coverup_906.py:42: in <module>
    test_get_all_host_vars()
coverup-tests/test_coverup_906.py:22: in test_get_all_host_vars
    inventory_module = InventoryModule()
lib/ansible/plugins/inventory/constructed.py:102: in __init__
    self._cache = FactCache()
lib/ansible/vars/fact_cache.py:26: in __init__
    raise AnsibleError('Unable to load the facts cache plugin (%s).' % (C.CACHE_PLUGIN))
E   ansible.errors.AnsibleError: Unable to load the facts cache plugin (non_existent_cache_plugin).
================================================================================= short test summary info ==================================================================================
ERROR coverup-tests/test_coverup_906.py - ansible.errors.AnsibleError: Unable to load the facts cache plugin (non_existent_cache_plugin).
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
    assert Path("coverup-tests/test_coverup_906.py") == BadTestsFinder.find_failed_test(output)
