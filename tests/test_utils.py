import coverup
import pytest
from hypothesis import given
import hypothesis.strategies as st


def test_format_ranges():
    assert "" == coverup.format_ranges(set(), set())
    assert "1-5" == coverup.format_ranges({1,2,3,5}, set())
    assert "1-3, 5" == coverup.format_ranges({1,2,3,5}, {4})
    assert "1-6, 10-11, 30" == coverup.format_ranges({1,2,4,5,6,10,11,30}, {8,14,15,16,17})


@given(st.integers(0, 10000), st.integers(1, 10000))
def test_format_ranges_is_sorted(a, b):
    b = a+b
    assert f"{a}-{b}" == coverup.format_ranges({i for i in range(a,b+1)}, set())


def test_lines_branches_do():
    assert "line 123 does" == coverup.lines_branches_do({123}, set(), set())
    assert "lines 123-125, 199 do" == coverup.lines_branches_do({123,124,125,199}, {128}, set())
    assert "branch 1->5 does" == coverup.lines_branches_do(set(), set(), {(1,5)})
    assert "branches 1->2, 1->5 do" == coverup.lines_branches_do(set(), set(), {(1,5),(1,2)})
    assert "line 123 and branches 1->exit, 1->2 do" == coverup.lines_branches_do({123}, set(), {(1,2),(1,0)})
    assert "lines 123-124 and branch 1->exit do" == coverup.lines_branches_do({123, 124}, set(), {(1,0)})
    assert "lines 123-125 and branches 1->exit, 1->2 do" == coverup.lines_branches_do({123,124,125}, set(), {(1,2),(1,0)})

    # if a line doesn't execute, neither do the branches that touch it...
    assert "lines 123-125 do" == coverup.lines_branches_do({123,124,125}, set(), {(123,124), (10,125)})



def test_clean_error_failure():
    error = """
F                                                                        [100%]
=================================== FAILURES ===================================
_____________ test_find_package_path_namespace_package_first_path ______________

    def test_find_package_path_namespace_package_first_path():
        # Create a dummy namespace package
        os.makedirs('namespace_package/submodule', exist_ok=True)
        open('namespace_package/__init__.py', 'a').close()
        open('namespace_package/submodule/__init__.py', 'a').close()
    
        # Add the current directory to sys.path
        import sys
        sys.path.append(os.getcwd())
    
        # Test the function
>       assert _find_package_path('namespace_package.submodule') == os.getcwd() + '/namespace_package'
E       AssertionError: assert '/Users/juan/tmp/flask' == '/Users/juan/...space_package'
E         - /Users/juan/tmp/flask/namespace_package
E         + /Users/juan/tmp/flask

tests/coverup_tmp_k076ps1h.py:19: AssertionError
=========================== short test summary info ============================
FAILED tests/coverup_tmp_k076ps1h.py::test_find_package_path_namespace_package_first_path
"""

    assert coverup.clean_error(error) == """
    def test_find_package_path_namespace_package_first_path():
        # Create a dummy namespace package
        os.makedirs('namespace_package/submodule', exist_ok=True)
        open('namespace_package/__init__.py', 'a').close()
        open('namespace_package/submodule/__init__.py', 'a').close()
    
        # Add the current directory to sys.path
        import sys
        sys.path.append(os.getcwd())
    
        # Test the function
>       assert _find_package_path('namespace_package.submodule') == os.getcwd() + '/namespace_package'
E       AssertionError: assert '/Users/juan/tmp/flask' == '/Users/juan/...space_package'
E         - /Users/juan/tmp/flask/namespace_package
E         + /Users/juan/tmp/flask

tests/coverup_tmp_k076ps1h.py:19: AssertionError
""".lstrip("\n")


def test_clean_error_error():
    error = """
==================================== ERRORS ====================================
________________ ERROR collecting tests/coverup_tmp_dkd55qhh.py ________________
ImportError while importing test module '/Users/juan/tmp/flask/tests/coverup_tmp_dkd55qhh.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Cellar/python@3.11/3.11.4_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
../../project/slipcover2/src/slipcover/importer.py:162: in exec_wrapper
    exec(obj, g)
tests/coverup_tmp_dkd55qhh.py:4: in <module>
    from flask.json import JSONProvider
E   ImportError: cannot import name 'JSONProvider' from 'flask.json' (/Users/juan/tmp/flask/src/flask/json/__init__.py)
=========================== short test summary info ============================
ERROR tests/coverup_tmp_dkd55qhh.py
!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!
"""
    assert coverup.clean_error(error) == """
ImportError while importing test module '/Users/juan/tmp/flask/tests/coverup_tmp_dkd55qhh.py'.
Hint: make sure your test modules/packages have valid Python names.
Traceback:
/opt/homebrew/Cellar/python@3.11/3.11.4_1/Frameworks/Python.framework/Versions/3.11/lib/python3.11/importlib/__init__.py:126: in import_module
    return _bootstrap._gcd_import(name[level:], package, level)
../../project/slipcover2/src/slipcover/importer.py:162: in exec_wrapper
    exec(obj, g)
tests/coverup_tmp_dkd55qhh.py:4: in <module>
    from flask.json import JSONProvider
E   ImportError: cannot import name 'JSONProvider' from 'flask.json' (/Users/juan/tmp/flask/src/flask/json/__init__.py)
""".lstrip("\n")


def test_compute_cost():
    assert pytest.approx(2.10, abs=.1) == \
           coverup.compute_cost({'prompt_tokens':60625, 'completion_tokens':4731}, 'gpt-4')

    # unknown model
    assert None == coverup.compute_cost({'prompt_tokens':60625, 'completion_tokens':4731}, 'unknown')

    # unknown token types
    assert None  == coverup.compute_cost({'blue_tokens':60625, 'red_tokens':4731}, 'gpt-4')


def test_find_imports():
    assert ['abc', 'bar', 'baz', 'cba', 'foo', 'xy'] == sorted(coverup.find_imports("""\
import foo, bar.baz
from baz.zab import a, b, c
from ..xy import yz

def foo_func():
    import abc
    from cba import xyzzy
"""))

    assert [] == coverup.find_imports("not a Python program")


def test_missing_imports():
    assert not coverup.missing_imports(['ast', 'dis', 'sys'])
    assert not coverup.missing_imports([])
    assert coverup.missing_imports(['sys', 'idontexist'])
