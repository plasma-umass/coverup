from pathlib import Path
from coverup import prompt


def test_get_module_name():
    fpath = Path('src/flask/json/provider.py').resolve()
    srcpath = Path('src/flask').resolve()

    assert 'flask.json.provider' == prompt.get_module_name(fpath, srcpath)

    assert None == prompt.get_module_name(fpath, Path('./tests').resolve())


