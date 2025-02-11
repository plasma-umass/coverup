import asyncio
import json

import argparse
import subprocess
import re
import sys
import typing as T

from pathlib import Path
from datetime import datetime

from . import llm
from .segment import *
from .prompt.prompter import Prompter
from .testrunner import *
from .version import __version__
from .utils import summary_coverage


def get_prompters() -> dict[str, T.Callable[[T.Any], Prompter]]:
    # in the future, we may dynamically load based on file names.

    from .prompt.gpt_v1 import GptV1Prompter
    from .prompt.gpt_v2 import GptV2Prompter
    from .prompt.gpt_v2_ablated import GptV2AblatedPrompter
    from .prompt.claude import ClaudePrompter

    return {
        "gpt-v1": GptV1Prompter,
        "gpt-v2": GptV2Prompter,
        "gpt-v2-no-coverage": lambda cmd_args: GptV2AblatedPrompter(cmd_args, with_coverage=False),
        "gpt-v2-no-code-context": lambda cmd_args: GptV2AblatedPrompter(cmd_args, with_get_info=False, with_imports=False),
        "gpt-v2-no-error-fixing": lambda cmd_args: GptV2AblatedPrompter(cmd_args, with_error_fixing=False),
        "gpt-v2-ablated": \
            lambda cmd_args: GptV2AblatedPrompter(cmd_args,
                with_coverage=False, with_get_info=False, with_imports=False, with_error_fixing=False),
        "claude": ClaudePrompter
    }


prompter_registry = get_prompters()


def parse_args(args=None):
    ap = argparse.ArgumentParser(prog='CoverUp',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument('source_files', type=Path, nargs='*',
                    help='only process certain source file(s)')

    def Path_dir(value):
        path_dir = Path(value).resolve()
        if not path_dir.is_dir(): raise argparse.ArgumentTypeError(f"\"{value}\" must be a directory")
        return path_dir

    ap.add_argument('--tests-dir', type=Path_dir, default='tests',
                    help='directory where tests reside')

    g = ap.add_mutually_exclusive_group(required=False)
    g.add_argument('--package-dir', type=Path_dir,
                    help='directory with the package sources (e.g., src/flask)')
    g.add_argument('--source-dir', type=Path_dir, dest='package_dir', help=argparse.SUPPRESS)

    ap.add_argument('--checkpoint', type=Path, 
                    help=f'path to save progress to (and to resume it from)')
    ap.add_argument('--no-checkpoint', action='store_const', const=None, dest='checkpoint', default=argparse.SUPPRESS,
                    help=f'disables checkpoint')

    def default_model():
        if 'OPENAI_API_KEY' in os.environ:
            return "gpt-4o"
        if 'ANTHROPIC_API_KEY' in os.environ:
            return "anthropic/claude-3-sonnet-20240229"
        if 'AWS_ACCESS_KEY_ID' in os.environ:
            return "bedrock/anthropic.claude-3-sonnet-20240229-v1:0"

    ap.add_argument('--model', type=str, default=default_model(),
                    help='OpenAI model to use')

    ap.add_argument('--prompt', '--prompt-family', type=str,
                    choices=list(prompter_registry.keys()),
                    default='gpt-v2',
                    help='Prompt style to use')

    ap.add_argument('--model-temperature', type=float, default=0,
                    help='Model "temperature" to use')

    ap.add_argument('--line-limit', type=int, default=50,
                    help='attempt to keep code segment(s) at or below this limit')

    ap.add_argument('--rate-limit', type=int,
                    help='max. tokens/minute to send in prompts')

    ap.add_argument('--max-attempts', type=int, default=3,
                    help='max. number of prompt attempts for a code segment')

    ap.add_argument('--max-backoff', type=int, default=64,
                    help='max. number of seconds for backoff interval')

    ap.add_argument('--dry-run', default=False,
                    action=argparse.BooleanOptionalAction,
                    help=f'whether to actually prompt the model; used for testing')

    ap.add_argument('--show-details', default=False,
                    action=argparse.BooleanOptionalAction,
                    help=f'show details of lines/branches after each response')

    ap.add_argument('--log-file', default=f"coverup-log",
                    help='log file to use')

    ap.add_argument('--pytest-args', type=str, default='',
                    help='extra arguments to pass to pytest')

    ap.add_argument('--install-missing-modules', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='attempt to install any missing modules')

    ap.add_argument('--prefix', type=str, default='coverup',
                    help='prefix to use for test file names')

    ap.add_argument('--write-requirements-to', type=Path,
                    help='append the name of any missing modules to the given file')

    ap.add_argument('--repeat-tests', type=int, default=5,
                    help='number of times to repeat test execution to help detect flaky tests')
    ap.add_argument('--no-repeat-tests', action='store_const', const=0, dest='repeat_tests', help=argparse.SUPPRESS)

    ap.add_argument('--disable-polluting', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='look for tests causing others to fail and disable them')

    ap.add_argument('--disable-failing', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='look for failing tests and disable them')

    ap.add_argument('--prompt-for-tests', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='prompt LLM for new tests')

    ap.add_argument('--isolate-tests', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='run tests in isolation (to work around any state pollution) when measuring suite coverage')

    ap.add_argument('--debug', '-d', default=False,
                    action=argparse.BooleanOptionalAction,
                    help='print out debugging messages.')

    ap.add_argument('--add-to-pythonpath', default=True,
                    action=argparse.BooleanOptionalAction,
                    help='add (parent of) source directory to PYTHONPATH')

    ap.add_argument('--branch-coverage', default=True,
                    action=argparse.BooleanOptionalAction,
                    help=argparse.SUPPRESS)

    def positive_int(value):
        ivalue = int(value)
        if ivalue < 0: raise argparse.ArgumentTypeError("must be a number >= 0")
        return ivalue

    ap.add_argument('--max-concurrency', type=positive_int, default=50,
                    help='maximum number of parallel requests; 0 means unlimited')

    ap.add_argument('--save-coverage-to', type=Path_dir,
                    help='save each new test\'s coverage to given directory')

    ap.add_argument('--version', action='version',
                    version=f"%(prog)s v{__version__} (Python {'.'.join(map(str, sys.version_info[:3]))})")

    args = ap.parse_args(args)

    for i in range(len(args.source_files)):
        args.source_files[i] = args.source_files[i].resolve()

    if args.disable_failing and args.disable_polluting:
        ap.error('Specify only one of --disable-failing and --disable-polluting')

    if not args.model:
        ap.error('Specify the model to use with --model')

    if args.package_dir:
        # use parent of package dir as base so that its name is included in module paths
        args.src_base_dir = args.package_dir.parent
        if not list(args.package_dir.glob("*.py")):
            sources = sorted(args.package_dir.glob("**/*.py"), key=lambda p: len(p.parts))
            suggestion = sources[0].parent if sources else None

            try:
                args.package_dir = args.package_dir.relative_to(Path.cwd())
                suggestion = suggestion.relative_to(Path.cwd()) if suggestion else None
            except ValueError:
                pass

            ap.error(f'No Python sources found in "{args.package_dir}"' +
                     (f'; did you mean "{suggestion}"?' if suggestion else '.'))

    elif args.source_files:
        if len({p.parent for p in args.source_files}) > 1:
            ap.error('All source files must be in the same directory unless --package-dir is given.')
        # use the directory itself as base, as these file(s) are not obviously part of anything
        args.src_base_dir = args.source_files[0].parent
    else:
        ap.error('Specify either --package-dir or a file name')

    return args


def test_file_path(args, test_seq: int) -> Path:
    """Returns the Path for a test's file, given its sequence number."""
    return args.tests_dir / f"test_{args.prefix}_{test_seq}.py"


test_seq: int = 1
def new_test_file(args: argparse.Namespace):
    """Creates a new test file, returning its Path."""

    global test_seq

    while True:
        p = test_file_path(args, test_seq)
        if not (p.exists() or (p.parent / ("disabled_" + p.name)).exists()):
            try:
                p.touch(exist_ok=False)
                return p
            except FileExistsError:
                pass

        test_seq += 1


def clean_error(error: str) -> str:
    """Conservatively removes pytest-generated (and possibly other) output not needed by GPT,
       to cut down on token use.  Conservatively: if the format isn't recognized, leave it alone."""

    if (match := re.search("=====+ (?:FAILURES|ERRORS) ===+\n" +\
                           "___+ [^\n]+ _+___\n" +\
                           "\n?" +\
                           "(.*)", error,
                           re.DOTALL)):
        error = match.group(1)

    if (match := re.search("(.*\n)" +\
                           "===+ short test summary info ===+", error,
                           re.DOTALL)):
        error = match.group(1)

    return error


log_file = None
def log_write(args: argparse.Namespace, seg: CodeSegment, m: str) -> None:
    """Writes to the log file, opening it first if necessary."""

    global log_file
    if not log_file:
        log_file = open(args.log_file, "a", buffering=1)    # 1 = line buffered

    log_file.write(f"---- {datetime.now().isoformat(timespec='seconds')} {seg} ----\n{m}\n")


def check_whole_suite(args: argparse.Namespace) -> None:
    """Check whole suite and disable any polluting/failing tests."""
    import pytest_cleanslate.reduce as reduce

    pytest_args = (*(("--count", str(args.repeat_tests)) if args.repeat_tests else ()), *args.pytest_args.split())

    while True:
        print("Checking test suite...  ", end='', flush=True)
        try:
            results = reduce.run_pytest(args.tests_dir,
                                        pytest_args=(*pytest_args, *(('-x',) if args.disable_polluting else ())),
                                        trace=args.debug)
            if not results.get_first_failed():
                print("tests ok!")
                return

        except subprocess.CalledProcessError as e:
            print(str(e) + "\n" + str(e.stdout, 'UTF-8', errors='ignore'))
            sys.exit(1)

        if args.disable_failing:
            failed = list(results.get_failed())
            print(f"{len(failed)} test(s)/module(s) failed, disabling...")
            to_disable = {Path(reduce.get_module(t)) for t in failed}

        else:
            try:
                reduction = reduce.reduce(tests_path=args.tests_dir, results=results,
                                          pytest_args=pytest_args, trace=args.debug)
            except subprocess.CalledProcessError as e:
                print(str(e) + "\n" + str(e.stdout, 'UTF-8', errors='ignore'))
                sys.exit(1)

            if 'error' in reduction:
                sys.exit(1)

            # FIXME add check for disabling too much
            # FIXME could just disable individual tests rather than always entire modules
            to_disable = {Path(reduce.get_module(m)) for m in (reduction['modules'] + reduction['tests'])}

        for t in to_disable:
            print(f"Disabling {t}")
            t.rename(t.parent / ("disabled_" + t.name))


def find_imports(python_code: str) -> T.List[str]:
    """Collects a list of packages needed by a program by examining its 'import' statements"""
    import ast

    try:
        t = ast.parse(python_code)
    except SyntaxError:
        return []

    modules = []

    for n in ast.walk(t):
        if isinstance(n, ast.Import):
            for name in n.names:
                if isinstance(name, ast.alias):
                    modules.append(name.name.split('.')[0])

        elif isinstance(n, ast.ImportFrom):
            if n.module and n.level == 0:
                modules.append(n.module.split('.')[0])

    return [m for m in modules if m != '__main__']


module_available = dict()
def missing_imports(modules: T.List[str]) -> T.List[str]:
    # TODO handle GPT sometimes generating 'from your_module import XYZ', asking us to modify

    import importlib.util

    for module in modules:
        if module not in module_available:
            spec = importlib.util.find_spec(module)
            module_available[module] = 0 if spec is None else 1

    return [m for m in modules if not module_available[m]]


def install_missing_imports(args: argparse.Namespace, seg: CodeSegment, modules: T.List[str]) -> bool: 
    global module_available

    import importlib.metadata

    all_ok = True
    for module in modules:
        try:
            # FIXME we probably want to limit the module(s) installed to an "approved" list
            p = subprocess.run((f"{sys.executable} -m pip install {module}").split(),
                               check=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
            version = importlib.metadata.version(module)
            module_available[module] = 2    # originally unavailable, but now added
            print(f"Installed module {module} {version}")
            log_write(args, seg, f"Installed module {module} {version}")

            if args.write_requirements_to:
                with args.write_requirements_to.open("a") as f:
                    f.write(f"{module}=={version}\n")
        except subprocess.CalledProcessError as e:
            log_write(args, seg, f"Unable to install module {module}:\n{str(e.stdout, 'UTF-8', errors='ignore')}")
            all_ok = False

    return all_ok


def get_required_modules() -> T.List[str]:
    """Returns a list of the modules found missing (and not installed)"""
    return [m for m in module_available if not module_available[m]]


PROGRESS_COUNTERS=['G', 'F', 'U', 'R']  # good, failed, useless, retry
class Progress:
    """Tracks progress, showing a tqdm-based bar."""

    def __init__(self, total, initial):
        import tqdm
        from collections import OrderedDict

        self._bar = tqdm.tqdm(total=total, initial=initial)

        self._postfix = OrderedDict()
        for p in [*PROGRESS_COUNTERS, 'cost']:
            self._postfix[p] = ''  # to establish order

        self._bar.set_postfix(ordered_dict=self._postfix)

    def update_cost(self, cost: float):
        self._postfix['cost'] = f'~${cost:.02f}'
        self._bar.set_postfix(ordered_dict=self._postfix)

    def update_counters(self, counters: dict):
        """Updates the counters display."""
        for k in counters:
            self._postfix[k] = counters[k]

        self._bar.set_postfix(ordered_dict=self._postfix)

    def signal_one_completed(self):
        """Signals an item completed."""
        self._bar.update()

    def close(self):
        """Closes the underlying tqdm bar."""
        self._bar.close()


class State:
    def __init__(self, initial_coverage: dict):
        """Initializes the CoverUp state."""
        from collections import defaultdict

        self._done : T.Dict[Path, T.Set[T.Tuple[int, int]]] = defaultdict(set)
        self._coverage = initial_coverage
        self._cost = 0.0
        self._counters = {k:0 for k in PROGRESS_COUNTERS}
        self._final_coverage: dict|None = None
        self._bar: Progress|None = None


    def get_initial_coverage(self) -> dict:
        """Returns the coverage initially measured."""
        return self._coverage


    def set_final_coverage(self, cov: dict) -> None:
        """Adds the final coverage obtained, so it can be saved in a checkpoint."""
        self._final_coverage = cov


    def set_progress_bar(self, bar: Progress):
        """Specifies a progress bar to update."""
        self._bar = bar
        if bar is not None:
            self._bar.update_cost(self._cost)
            self._bar.update_counters(self._counters)


    def add_cost(self, cost: float) -> None:
        self._cost += cost
        if self._bar:
            self._bar.update_cost(self._cost)


    def inc_counter(self, key: str):
        """Increments a progress counter."""
        self._counters[key] += 1

        if self._bar:
            self._bar.update_counters(self._counters)


    def mark_done(self, seg: CodeSegment):
        """Marks a segment done."""
        self._done[seg.path].add((seg.begin, seg.end))


    def is_done(self, seg: CodeSegment):
        """Returns whether a segment is done."""
        return (seg.begin, seg.end) in self._done[seg.path]


    @staticmethod
    def load_checkpoint(ckpt_file: Path):   # -> State
        """Loads state from a checkpoint."""
        try:
            with ckpt_file.open("r") as f:
                ckpt = json.load(f)
                if ckpt['version'] != 2: return None
                assert 'done' in ckpt
                assert 'cost' in ckpt
                assert 'counters' in ckpt
                assert 'coverage' in ckpt, "coverage information missing in checkpoint"

                state = State(ckpt['coverage'])
                state._cost = ckpt['cost']
                state._counters = ckpt['counters']
                for filename, done_list in ckpt['done'].items():
                    state._done[Path(filename).resolve()] = set(tuple(d) for d in done_list)
                return state

        except FileNotFoundError:
            return None


    def save_checkpoint(self, ckpt_file: Path):
        """Saves this state to a checkpoint file."""
        ckpt = {
            'version': 2,
            'done': {str(k):list(v) for k,v in self._done.items() if len(v)}, # cannot serialize 'Path' or 'set' as-is
            'cost': self._cost,
            'counters': self._counters,
            'coverage': self._coverage,
            **({'final_coverage': self._final_coverage} if self._final_coverage else {})
            # FIXME save missing modules
        }

        with ckpt_file.open("w") as f:
            json.dump(ckpt, f)


def extract_python(response: str) -> str:
    # This regex accepts a truncated code block... this seems fine since we'll try it anyway
    m = re.search(r'```python\n(.*?)(?:```|\Z)', response, re.DOTALL)
    if not m: raise RuntimeError(f"Unable to extract Python code from response {response}")
    return m.group(1)


state: State

async def improve_coverage(
    args: argparse.Namespace,
    chatter: llm.Chatter,
    prompter: Prompter,
    seg: CodeSegment
) -> bool:
    """Works to improve coverage for a code segment."""

    messages = prompter.initial_prompt(seg)
    attempts = 0

    if args.dry_run:
        return True

    while True:
        attempts += 1
        if (attempts > args.max_attempts):
            log_write(args, seg, "Too many attempts, giving up")
            break

        if not (response := await chatter.chat(messages, ctx=seg)):
            log_write(args, seg, "giving up")
            break

        response_message = response["choices"][0]["message"]
        messages.append(response_message)

        if '```python' in response_message['content']:
            last_test = extract_python(response_message['content'])
        else:
            log_write(args, seg, "No Python code in GPT response, giving up")
            break

        if missing := missing_imports(find_imports(last_test)):
            log_write(args, seg, f"Missing modules {' '.join(missing)}")
            if not args.install_missing_modules or not install_missing_imports(args, seg, missing):
                return False # not finished: needs a missing module

        try:
            pytest_args = (f"--count={args.repeat_tests} " if args.repeat_tests else "") + args.pytest_args
            coverage = await measure_test_coverage(test=last_test, tests_dir=args.tests_dir,
                                                 pytest_args=pytest_args,
                                                 isolate_tests=args.isolate_tests,
                                                 branch_coverage=args.branch_coverage,
                                                 log_write=lambda msg: log_write(args, seg, msg))

        except subprocess.TimeoutExpired:
            log_write(args, seg, "measure_coverage timed out")
            # FIXME is the built-in timeout reasonable? Do we prompt for a faster test?
            # We don't want slow tests, but there may not be any way around it.
            return True

        except subprocess.CalledProcessError as e:
            state.inc_counter('F')
            error = clean_error(str(e.stdout, 'UTF-8', errors='ignore'))
            if not (prompts := prompter.error_prompt(seg, error)):
                log_write(args, seg, "Test failed:\n\n" + error)
                break

            messages.extend(prompts)
            continue

        result = coverage['files'].get(seg.filename, None)
        new_lines = set(result['executed_lines']) if result else set()
        new_branches = set(tuple(b) for b in result['executed_branches']) if (result and \
                                                                              'executed_branches' in result) \
                       else set()
        gained_lines = seg.missing_lines.intersection(new_lines)
        gained_branches = seg.missing_branches.intersection(new_branches)

        if args.show_details:
            print(seg.identify())
            print(f"Originally missing: {sorted(seg.missing_lines)}")
            print(f"                    {list(format_branches(seg.missing_branches))}")
            print(f"Gained:             {sorted(gained_lines)}")
            print(f"                    {list(format_branches(gained_branches))}")

        if not gained_lines and not gained_branches:
            state.inc_counter('U')
            if not (prompts := prompter.missing_coverage_prompt(seg, seg.missing_lines, seg.missing_branches)):
                log_write(args, seg, "Test doesn't improve coverage")
                break

            messages.extend(prompts)
            continue

        asked = {'lines': sorted(seg.missing_lines), 'branches': sorted(seg.missing_branches)}
        gained = {'lines': sorted(gained_lines), 'branches': sorted(gained_branches)}

        new_test = new_test_file(args)
        new_test.write_text(f"# file: {seg.identify()}\n" +\
                            f"# asked: {json.dumps(asked)}\n" +\
                            f"# gained: {json.dumps(gained)}\n\n" +\
                            last_test)

        log_write(args, seg, f"Saved as {new_test}\n")
        state.inc_counter('G')

        if args.save_coverage_to:
            with (args.save_coverage_to / (new_test.stem + ".json")).open("w") as f:
                json.dump(coverage, f)

        # TODO re-add segment with remaining missing coverage?
        break

    return True # finished


def add_to_pythonpath(dir: Path):
    import os
    os.environ['PYTHONPATH'] = str(dir) + (f":{os.environ['PYTHONPATH']}" if 'PYTHONPATH' in os.environ else "")
    sys.path.insert(0, str(dir))


def main():
    from collections import defaultdict
    import os

    global state
    args = parse_args()

    if not args.tests_dir.exists():
        print(f'Directory "{args.tests_dir}" does not exist. Please specify the correct one or create it.')
        return 1

    # add source dir to paths so that the module doesn't need to be installed to be worked on
    if args.add_to_pythonpath:
        add_to_pythonpath(args.src_base_dir)

    if args.prompt_for_tests:
        try:
            chatter = llm.Chatter(model=args.model)
            chatter.set_log_msg(lambda ctx, msg: log_write(args, ctx, msg))
            chatter.set_log_json(lambda ctx, j: log_write(args, ctx, json.dumps(j, indent=2)))
            chatter.set_signal_retry(lambda: state.inc_counter('R'))

            chatter.set_model_temperature(args.model_temperature)
            chatter.set_max_backoff(args.max_backoff)

            if args.rate_limit:
                chatter.set_token_rate_limit((args.rate_limit, 60))

            prompter = prompter_registry[args.prompt](cmd_args=args)
            for f in prompter.get_functions():
                chatter.add_function(f)

        except llm.ChatterError as e:
            print(e)
            return 1

        log_write(args, 'startup', f"Command: {' '.join(sys.argv)}")

        # --- (1) load or measure initial coverage, figure out segmentation ---

        if args.checkpoint and (state := State.load_checkpoint(args.checkpoint)):
            print("Continuing from checkpoint;  coverage: ", end='', flush=True)
            coverage = state.get_initial_coverage()
        else:
            if args.disable_polluting or args.disable_failing:
                # check and clean up suite before measuring coverage
                check_whole_suite(args)

            try:
                print("Measuring coverage...  ", end='', flush=True)
                coverage = measure_suite_coverage(tests_dir=args.tests_dir, source_dir=args.package_dir,
                                                  pytest_args=args.pytest_args,
                                                  isolate_tests=args.isolate_tests,
                                                  branch_coverage=args.branch_coverage,
                                                  trace=(print if args.debug else None))
                state = State(coverage)

            except subprocess.CalledProcessError as e:
                print("Error measuring coverage:\n" + str(e.stdout, 'UTF-8', errors='ignore'))
                return 1

        print(summary_coverage(coverage, args.source_files))
        # TODO also show running coverage estimate

        chatter.set_add_cost(state.add_cost)

        segments = sorted(get_missing_coverage(state.get_initial_coverage(), line_limit=args.line_limit),
                          key=lambda seg: seg.missing_count(), reverse=True)

        # save initial coverage so we don't have to redo it next time
        if args.checkpoint:
            state.save_checkpoint(args.checkpoint)

        # --- (2) prompt for tests ---

        print(f"Prompting {args.model} for tests to increase coverage...")
        print("(in the following, G=good, F=failed, U=useless and R=retry)")

        async def work_segment(seg: CodeSegment) -> None:
            if await improve_coverage(args, chatter, prompter, seg):
                # Only mark done if was able to complete (True return),
                # so that it can be retried after installing any missing modules
                state.mark_done(seg)

            if args.checkpoint:
                state.save_checkpoint(args.checkpoint)
            progress.signal_one_completed()

        worklist = []
        seg_done_count = 0
        for seg in segments:
            if not seg.path.is_relative_to(args.src_base_dir):
                continue

            if args.source_files and seg.path not in args.source_files:
                continue

            if state.is_done(seg):
                seg_done_count += 1
            else:
                worklist.append(work_segment(seg))

        progress = Progress(total=len(worklist)+seg_done_count, initial=seg_done_count)
        state.set_progress_bar(progress)

        async def run_it():
            if args.max_concurrency:
                semaphore = asyncio.Semaphore(args.max_concurrency)

                async def sem_coro(coro):
                    async with semaphore:
                        return await coro

                await asyncio.gather(*(sem_coro(c) for c in worklist))
            else:
                await asyncio.gather(*worklist)

        try:
            asyncio.run(run_it())
        except KeyboardInterrupt:
            print("Interrupted.")
            if args.checkpoint:
                state.save_checkpoint(args.checkpoint)
            return 1

        progress.close()

    # --- (3) clean up resulting test suite ---

    if args.disable_polluting or args.disable_failing:
        check_whole_suite(args)

    # --- (4) show final coverage

    if args.prompt_for_tests:
        try:
            print("Measuring coverage...  ", end='', flush=True)
            coverage = measure_suite_coverage(tests_dir=args.tests_dir, source_dir=args.package_dir,
                                              pytest_args=args.pytest_args,
                                              isolate_tests=args.isolate_tests,
                                              branch_coverage=args.branch_coverage,
                                              trace=(print if args.debug else None))

        except subprocess.CalledProcessError as e:
            print("Error measuring coverage:\n" + str(e.stdout, 'UTF-8', errors='ignore'))
            return 1

        print(summary_coverage(coverage, args.source_files))

    # --- (5) save state and show missing modules, if appropriate

        if args.checkpoint:
            state.set_final_coverage(coverage)
            state.save_checkpoint(args.checkpoint)

        if not args.install_missing_modules and (required := get_required_modules()):
            print(f"Some modules seem to be missing:  {', '.join(str(m) for m in required)}")
            if args.write_requirements_to:
                with args.write_requirements_to.open("a") as f:
                    for module in required:
                        f.write(f"{module}\n")

    return 0
