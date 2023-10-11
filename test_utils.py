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


def test_lines_branches_do():
    assert "line 123 does" == coverup.lines_branches_do({123}, {})
    assert "lines 123-125, 199 do" == coverup.lines_branches_do({123,124,125,199}, {})
    assert "branch 1->5 does" == coverup.lines_branches_do({}, {(1,5)})
    assert "branches 1->2, 1->5 do" == coverup.lines_branches_do({}, {(1,5),(1,2)})
    assert "line 123 and branches 1->exit, 1->2 do" == coverup.lines_branches_do({123}, {(1,2),(1,0)})
    assert "lines 123-124 and branch 1->exit do" == coverup.lines_branches_do({123, 124}, {(1,0)})
    assert "lines 123-125 and branches 1->exit, 1->2 do" == coverup.lines_branches_do({123,124,125}, {(1,2),(1,0)})


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
