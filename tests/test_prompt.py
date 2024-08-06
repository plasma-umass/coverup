from pathlib import Path
from coverup import prompt


def test_get_module_name():
    fpath = Path('src/flask/json/provider.py').resolve()
    srcpath = Path('src/flask').resolve()

    assert 'flask.json.provider' == prompt.get_module_name(fpath, srcpath)

    assert None == prompt.get_module_name(fpath, Path('./tests').resolve())


def test_gpt4_v1_relative_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "lib" / "ansible").mkdir(parents=True)
    (tmp_path / "lib" / "ansible" / "__init__.py").touch()
    (tmp_path / "tests").mkdir(parents=True)

    from coverup.coverup import parse_args
    args = parse_args(["--source", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    from coverup.segment import CodeSegment
    segment = CodeSegment(
                    filename=Path("lib") / "ansible" / "context.py",
                    name="foo",
                    begin=10, end=20, 
                    lines_of_interest=set(), missing_lines=set(), executed_lines=set(), missing_branches=set(),
                    context=[], imports=[]
    )

    monkeypatch.setattr(CodeSegment, "get_excerpt", lambda self, tag_lines=True: '<excerpt>')

    p = prompt.Gpt4PrompterV1(args)
    initial = p.initial_prompt(segment)
    assert "extracted from ansible/context.py," in initial[0]['content']

    assert "when tested, it does not execute." in initial[0]['content']


def test_claude_relative_file_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "lib" / "ansible").mkdir(parents=True)
    (tmp_path / "lib" / "ansible" / "__init__.py").touch()
    (tmp_path / "tests").mkdir(parents=True)

    from coverup.coverup import parse_args
    args = parse_args(["--source", "lib/ansible", "--tests", "tests", "--model", "anthropic/claude-3-sonnet-20240229"])

    from coverup.segment import CodeSegment
    segment = CodeSegment(
                    filename=Path("lib") / "ansible" / "context.py",
                    name="foo",
                    begin=10, end=20, 
                    lines_of_interest=set(), missing_lines=set(), executed_lines=set(), missing_branches=set(),
                    context=[], imports=[]
    )

    monkeypatch.setattr(CodeSegment, "get_excerpt", lambda self, tag_lines=True: '<excerpt>')

    p = prompt.ClaudePrompter(args)
    initial = p.initial_prompt(segment)
    assert '<file path="ansible/context.py"' in initial[1]['content']

    assert "when tested, it does not execute." in initial[1]['content']
