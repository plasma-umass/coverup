import setuptools
from pathlib import Path
import re

try:
    import tomllib
except ImportError:
    import tomli as tomllib

def get_version():
    import re
    v = re.findall(r"^__version__ *= *\"([^\"]+)\"", Path("src/coverup/version.py").read_text())[0]
    return v

def get_url():
    return tomllib.loads(Path("pyproject.toml").read_text())['project']['urls']['Repository']

def long_description():
    text = Path("README.md").read_text(encoding="utf-8")

    # Rewrite any relative paths to version-specific absolute paths,
    # so that they work from within PyPI
    sub = r'\1' + get_url() + "/blob/v" + get_version() + r'/\2'
    text = re.sub(r'(src=")((?!https?://))', sub, text)
    text = re.sub(r'(\[.*?\]\()((?!https?://))', sub, text)

    return text

if __name__ == "__main__":
    setuptools.setup(
        version=get_version(),
        long_description=long_description(),
        long_description_content_type="text/markdown",
    )
