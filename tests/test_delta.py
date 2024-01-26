import pytest
from hypothesis import given, strategies as st
from pathlib import Path
from coverup.delta import DeltaDebugger, BadTestsFinder


class NumbersFinder(DeltaDebugger):
    def __init__(self, numbers: set):
        super(NumbersFinder, self).__init__(trace=print)
        self._numbers = numbers

    def test(self, testset: set, *kwargs) -> bool:
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
@pytest.mark.parametrize("fail_load", [True, False])
def test_find_culprit(tmpdir, existing_testconf, fail_load):
    test_dir = Path(tmpdir)

    def seq2p(seq):
        return test_dir / f"test_coverup_{seq}.py"

    if existing_testconf:
        (test_dir / "testconf.py").write_text("# my precious")

    all_tests = {seq2p(seq) for seq in range(10)}
    for t in all_tests:
        t.write_text('def test_foo(): pass')

    culprit = seq2p(3)
    culprit.write_text("import sys\n" + "sys.hexversion=0")

    failing = seq2p(6)
    if fail_load:
        failing.write_text("import sys\n" + "assert sys.hexversion != 0\n" + "def test_foo(): pass")
    else:
        failing.write_text("import sys\n" + "def test_foo(): assert sys.hexversion != 0")

    btf = BadTestsFinder(test_dir, trace=print)

    assert not btf.test({failing})

    assert {seq2p(3)} == btf.find_culprit(failing)

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

def test_find_failed_test_running():
    # from python3 -m pytest -x -qq --disable-warnings coverup-tests
    output = """\
.................................................................................................................................................................................. [  9%]
......................F
======================================================================================== FAILURES ========================================================================================
_____________________________________________________________________________ test_v2_runner_item_on_failed ______________________________________________________________________________

callback_module = <ansible.plugins.callback.default.CallbackModule object at 0x7f90afb786d0>, task_result = <ansible.executor.task_result.TaskResult object at 0x7f90aeafbdc0>

    def test_v2_runner_item_on_failed(callback_module, task_result):
        with patch('ansible.plugins.callback.default.C') as mock_color:
            mock_color.COLOR_ERROR = 'red'
>           callback_module.v2_runner_item_on_failed(task_result)

coverup-tests/test_coverup_1117.py:39: 
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
lib/ansible/plugins/callback/default.py:292: in v2_runner_item_on_failed
    self._print_task_banner(result._task)
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 

self = <ansible.plugins.callback.default.CallbackModule object at 0x7f90afb786d0>, task = TASK: fake_name

    def _print_task_banner(self, task):
        # args can be specified as no_log in several places: in the task or in
        # the argument spec.  We can check whether the task is no_log but the
        # argument spec can't be because that is only run on the target
        # machine and we haven't run it thereyet at this time.
        #
        # So we give people a config option to affect display of the args so
        # that they can secure this if they feel that their stdout is insecure
        # (shoulder surfing, logging stdout straight to a file, etc).
        args = ''
        if not task.no_log and C.DISPLAY_ARGS_TO_STDOUT:
            args = u', '.join(u'%s=%s' % a for a in task.args.items())
            args = u' %s' % args
    
        prefix = self._task_type_cache.get(task._uuid, 'TASK')
    
        # Use cached task name
        task_name = self._last_task_name
        if task_name is None:
            task_name = task.get_name().strip()
    
>       if task.check_mode and self.check_mode_markers:
E       AttributeError: 'CallbackModule' object has no attribute 'check_mode_markers'

lib/ansible/plugins/callback/default.py:211: AttributeError
================================================================================ short test summary info =================================================================================
FAILED coverup-tests/test_coverup_1117.py::test_v2_runner_item_on_failed - AttributeError: 'CallbackModule' object has no attribute 'check_mode_markers'
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
    assert Path("coverup-tests/test_coverup_1117.py") == BadTestsFinder.find_failed_test(output)
