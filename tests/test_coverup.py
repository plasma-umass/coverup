from pathlib import Path
from coverup import coverup
import pytest

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


def test_find_imports():
    assert ['abc', 'bar', 'baz', 'cba', 'foo'] == sorted(coverup.find_imports("""\
import foo, bar.baz
from baz.zab import a, b, c
from ..xy import yz         # relative, package likely present
from . import blah          # relative, package likely present
import __main__

def foo_func():
    import abc
    from cba import xyzzy
"""))

    assert [] == coverup.find_imports("not a Python program")


def test_missing_imports():
    assert not coverup.missing_imports(['ast', 'dis', 'sys'])
    assert not coverup.missing_imports([])
    assert coverup.missing_imports(['sys', 'idontexist'])


def test_extract_python():
    assert "foo()\n\nbar()\n" == coverup.extract_python("""\
```python
foo()

bar()
```
""")

    assert "foo()\n\nbar()\n" == coverup.extract_python("""\
```python
foo()

bar()
```""")

    assert "foo()\n\nbar()\n" == coverup.extract_python("""\
```python
foo()

bar()
""")


@pytest.mark.parametrize("pythonpath_exists", [True, False])
def test_add_to_pythonpath(pythonpath_exists):
    import sys, os
    saved_environ = dict(os.environ.items())
    saved_syspath = list(sys.path)

    try:
        if pythonpath_exists:
            os.environ['PYTHONPATH'] = 'foo:bar'
        elif 'PYTHONPATH' in os.environ:
            del os.environ['PYTHONPATH']

        coverup.add_to_pythonpath(Path("baz"))

        if pythonpath_exists:
            assert os.environ['PYTHONPATH'] == 'baz:foo:bar'
        else:
            assert os.environ['PYTHONPATH'] == 'baz'

        assert sys.path == ['baz'] + saved_syspath

    finally:
        os.environ = saved_environ
        sys.path = saved_syspath
