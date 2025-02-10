import pytest
from pathlib import Path
from coverup.segment import CodeSegment
from coverup.prompt.prompter import get_module_name
from coverup.prompt.gpt_v1 import GptV1Prompter
from coverup.prompt.gpt_v2_ablated import GptV2Prompter, GptV2AblatedPrompter
from coverup.prompt.gpt_v2_fully_ablated import GptV2FullyAblatedPrompter
from coverup.prompt.claude import ClaudePrompter
from coverup.coverup import parse_args
import coverup.codeinfo


def test_get_module_name():
    fpath = Path('src/flask/json/provider.py').resolve()
    base_path = Path('src').resolve()

    assert 'flask.json.provider' == get_module_name(fpath, base_path)

    assert None == get_module_name(fpath, Path('./tests').resolve())


class MockSegment(CodeSegment):
    def __init__(self):
        super().__init__(
            filename=Path("lib") / "ansible" / "context.py",
            name="foo",
            begin=10, end=20, 
            lines_of_interest=set(),
            missing_lines=set(),
            executed_lines=set(),
            missing_branches=set(),
            context=[],
            imports=[]
        )

    def lines_branches_missing_do(self) -> str:
        return "mock lines and branches do"


    def get_excerpt(self, *, tag_lines=True, add_imports=True) -> str:
        return f"""\
            fake excerpt
            {add_imports=}
            {tag_lines=}
        """


@pytest.fixture
def pkg_fixture(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    (tmp_path / "lib" / "ansible").mkdir(parents=True)
    (tmp_path / "lib" / "ansible" / "__init__.py").touch()
    (tmp_path / "tests").mkdir(parents=True)

    def mock_parse_file(*args, **kwargs):
        return None

    def mock_get_info(module, name, **kwargs):
        return f"get_info {name=} generate_imports={kwargs.get('generate_imports', True)}"

    monkeypatch.setattr(coverup.codeinfo, 'parse_file', mock_parse_file)
    monkeypatch.setattr(coverup.codeinfo, 'get_info', mock_get_info)


def test_gpt_v1_relative_file_name(pkg_fixture):
    args = parse_args(["--source", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])
    print(f"{args.src_base_dir=}")

    p = GptV1Prompter(args)
    initial = p.initial_prompt(MockSegment())
    assert "extracted from ansible/context.py," in initial[0]['content']

    assert "when tested, it does not execute." in initial[0]['content']


def test_claude_relative_file_name(pkg_fixture):
    args = parse_args(["--source", "lib/ansible", "--tests", "tests", "--model", "anthropic/claude-3-sonnet-20240229"])

    p = ClaudePrompter(args)
    initial = p.initial_prompt(MockSegment())
    assert '<file path="ansible/context.py"' in initial[1]['content']

    assert "when tested, it does not execute." in initial[1]['content']


def test_ablated_nothing_ablated(pkg_fixture):
    args = parse_args(["--package", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    s = MockSegment()
    v2 = GptV2Prompter(args)
    ablated = GptV2AblatedPrompter(args)

    assert [ablated.get_info] == ablated.get_functions()
    assert v2.initial_prompt(s) == ablated.initial_prompt(s)
    assert v2.error_prompt(s, "the error") == ablated.error_prompt(s, "the error")
    assert v2.missing_coverage_prompt(s, {1,2,3}, set()) == ablated.missing_coverage_prompt(s, {1,2,3}, set())

    assert "get_info name='foo' generate_imports=True" in ablated.get_info(s, "foo")

def test_ablated_everything_ablated(pkg_fixture):
    args = parse_args(["--package", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    s = MockSegment()
    v2 = GptV2FullyAblatedPrompter(args)
    ablated = GptV2AblatedPrompter(
        args,
        with_coverage=False,
        with_get_info=False,
        with_imports=False,
        with_error_fixing=False
    )

    assert [] == ablated.get_functions()
    assert v2.initial_prompt(s) == ablated.initial_prompt(s)
    assert v2.error_prompt(s, "the error") == ablated.error_prompt(s, "the error")
    assert v2.missing_coverage_prompt(s, {1,2,3}, set()) == ablated.missing_coverage_prompt(s, {1,2,3}, set())


def test_ablated_coverage(pkg_fixture):
    args = parse_args(["--package", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    s = MockSegment()
    v2 = GptV2Prompter(args)
    ablated = GptV2AblatedPrompter(args, with_coverage=False)

    assert [ablated.get_info] == ablated.get_functions()
    assert "mock lines and branches" not in ablated.initial_prompt(s)[0]['content']
    assert "tag_lines=False" in ablated.initial_prompt(s)[0]['content']
    assert v2.error_prompt(s, "the error") == ablated.error_prompt(s, "the error")
    assert ablated.missing_coverage_prompt(s, {1,2,3}, set()) is None

    assert "get_info name='foo' generate_imports=True" in ablated.get_info(s, "foo")


def test_ablated_imports(pkg_fixture):
    args = parse_args(["--package", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    s = MockSegment()
    v2 = GptV2Prompter(args)
    ablated = GptV2AblatedPrompter(args, with_imports=False)

    assert [ablated.get_info] == ablated.get_functions()
    assert "mock lines and branches" in ablated.initial_prompt(s)[0]['content']
    assert "tag_lines=True" in ablated.initial_prompt(s)[0]['content']
    assert "add_imports=False" in ablated.initial_prompt(s)[0]['content']
    assert v2.error_prompt(s, "the error") == ablated.error_prompt(s, "the error")
    assert v2.missing_coverage_prompt(s, {1,2,3}, set()) == ablated.missing_coverage_prompt(s, {1,2,3}, set())

    assert "get_info name='foo' generate_imports=False" in ablated.get_info(s, "foo")


def test_ablated_get_info(pkg_fixture):
    args = parse_args(["--package", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    s = MockSegment()
    ablated = GptV2AblatedPrompter(args, with_get_info=False)

    assert ablated.get_functions() == []
    assert "get_info" not in ablated.initial_prompt(s)[0]['content']
    assert "mock lines and branches" in ablated.initial_prompt(s)[0]['content']
    assert "tag_lines=True" in ablated.initial_prompt(s)[0]['content']
    assert "add_imports=True" in ablated.initial_prompt(s)[0]['content']
    assert "get_info" not in ablated.error_prompt(s, "the error")[0]['content']
    assert "get_info" not in ablated.missing_coverage_prompt(s, {1,2,3}, set())[0]['content']


def test_ablated_error_fixing(pkg_fixture):
    args = parse_args(["--package", "lib/ansible", "--tests", "tests", "--model", "gpt-4o"])

    s = MockSegment()
    v2 = GptV2Prompter(args)
    ablated = GptV2AblatedPrompter(args, with_error_fixing=False)

    assert [ablated.get_info] == ablated.get_functions()
    assert v2.initial_prompt(s) == ablated.initial_prompt(s)
    assert ablated.error_prompt(s, "the error") is None
    assert v2.missing_coverage_prompt(s, {1,2,3}, set()) == ablated.missing_coverage_prompt(s, {1,2,3}, set())

    assert "get_info name='foo' generate_imports=True" in ablated.get_info(s, "foo")
