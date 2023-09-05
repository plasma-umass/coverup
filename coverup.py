import openai
import json
import subprocess
import os
import ast
from collections import defaultdict
from pathlib import Path
import typing


PREFIX = 'coverup'
openai.key=os.environ['OPENAI_API_KEY']
openai.organization=os.environ['OPENAI_PLASMA_ORG'] # FIXME

total_tokens = 0


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(prog='CoverUp')
    ap.add_argument('cov_json', type=Path)

    # "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
    ap.add_argument('--model', type=str, default='gpt-4')

    # FIXME derive this somehow?
    ap.add_argument('--tests-dir', type=Path, default='tests')

    # FIXME derive this somehow?
    ap.add_argument('--source-dir', type=str, default='src')

    ap.add_argument('--source-file', type=str)

    ap.add_argument('--checkpoint', action='store_true')

    return ap.parse_args()

args = parse_args()

log = open(PREFIX + "-log", "w", buffering=1)    # 1 = line buffered

test_seq = None
def get_test_path(advance = False):
    global test_seq

    if advance or test_seq is None:
        test_seq = 1 if test_seq is None else test_seq+1

        while True:
            p = args.tests_dir / f"test_{PREFIX}_{test_seq}.py"
            if not p.exists():
                break

            test_seq += 1

        return p

    return args.tests_dir / f"test_{PREFIX}_{test_seq}.py"


def delete_last_test():
    try:
        get_test_path().unlink()
    except FileNotFoundError:
        pass


class CodeSegment:
    def __init__(self, filename: Path, name: str, begin: int, end: int, missing_lines: typing.Set[int],
                 context: typing.List[typing.Tuple[int, int]]):
        self.filename = filename
        self.name = name
        self.begin = begin
        self.end = end
        self.missing_lines = missing_lines
        self.context = context


    def __str__(self):
        return f"CodeSegment({self.name}, \"{self.filename}\" {self.begin}-{self.end-1})"


    def get_excerpt(self, base_path = ''):
        excerpt = []
        with open(base_path + self.filename, "r") as src:
            code = src.readlines()

            for b, e in self.context:
                for i in range(b, e):
                    excerpt.extend([f"{i:10}: ", code[i-1]])

            for i in range(self.begin, self.end):
                excerpt.extend([f"{i:10}: ", code[i-1]])

        return ''.join(excerpt)


    def get_missing(self):
        # FIXME compress by using ranges
        return ", ".join(map(str, self.missing_lines))


# FIXME make size_limit configurable; base it on tokens, not lines?
def get_missing_coverage(jsonfile, base_path = '', size_limit=100):
    """Processes a JSON SlipCover output and generates a list of Python code segments,
    such as functions or classes, which have less than 100% coverage.
    """
    with open(jsonfile, "r") as f:
        cov = json.load(f)

    code_segs = []

    def find_enclosing(root, line):
        for node in ast.walk(root):
            if node is root:
                continue

            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and \
               hasattr(node, "lineno") and node.lineno <= line <= node.end_lineno:
                return node

    def get_line_range(node):
        begin = node.lineno
        for d in node.decorator_list: # skip back to include decorators, in case they matter
            begin = min(begin, d.lineno)

        return (begin, node.end_lineno+1)   # +1 for range() style

    for fname in cov['files']:
        with open(base_path + fname, "r") as src:
            tree = ast.parse(src.read(), base_path + fname)

        code_this_file = defaultdict(set)
        ctx_this_file = defaultdict(set)

        for line in cov['files'][fname]['missing_lines']:
            if node := find_enclosing(tree, line):
                begin, end = get_line_range(node)

                context = []

                while end - begin > size_limit:
                    child = find_enclosing(node, line)
                    if not child:
                        break

                    context.append((begin, node.lineno+1))
                    begin, end = get_line_range(child)
                    node = child

                #print(f"{fname} line {line} -> {str(node)} {begin}..{node.end_lineno}")
                code_this_file[(node.name, begin, end)].add(line)
                ctx_this_file[(node.name, begin, end)] = context

        if code_this_file:
            for it in code_this_file:
                name, begin, end = it
                missing = code_this_file[it]
                code_segs.append(CodeSegment(fname, name, begin, end, missing, ctx_this_file[it]))

    return code_segs


def measure_coverage(test: str):
    import sys

    test_path = get_test_path()
    json_path = Path("/tmp") / f"{test_path.stem}.json"     # FIXME use temporary for json

    with test_path.open("w") as test_file:
        test_file.write(test)

    # -qq to cut down on tokens
    p = subprocess.run(f"{sys.executable} -m slipcover --json --out {json_path} -m pytest -qq {test_path}".split(),
                       check=True, capture_output=True, timeout=60)
    log.write(str(p.stdout, 'UTF-8') + "\n")

    with json_path.open() as j:
        cov = json.load(j)

    return cov["files"]


def improve_coverage(seg: CodeSegment):
    print(f"\n=== {seg.name} ({seg.filename}) ===")

    def pl(item, singular, plural = None):
        if len(item) <= 1:
            return singular
        return plural if plural is not None else f"{singular}s"

    messages = [{"role": "user",
                 "content": f"""
The code below, extracted from {seg.filename}, does not achieve full line coverage:
when tested, {pl(seg.missing_lines,'line')} {seg.get_missing()} {pl(seg.missing_lines,'does','do')} not execute.
Create a new pytest test function that executes these missing lines, always making
sure that the new test is correct and indeed improves coverage.  Always send entire Python
test scripts when proposing a new test or correcting one you previously proposed.
Respond ONLY with the Python code enclosed in backticks, without any explanation.
```python
{seg.get_excerpt()}
```
"""
            }]

    log.write(messages[0]['content'] + "\n---\n")

    attempts = 0

    while True:
        if (attempts > 5):
            log.write("Too many attempts, giving up\n---\n")
            print("giving up")
            delete_last_test()
            break

        attempts += 1

        print("sent request, waiting on GPT...")
        sleep = 1
        while True:
            try:
                completion_args = {
                    'model': args.model,
                    'messages': messages,
                    'temperature': 0
                }

                response = openai.ChatCompletion.create(**completion_args)
                break

            except (openai.error.ServiceUnavailableError,
                    openai.error.RateLimitError,
                    openai.error.Timeout) as e:
                print(e)
                print(f"waiting {sleep}s)")
                import time
                time.sleep(sleep)
                sleep *= 2

            except openai.error.InvalidRequestError as e:
                # usually "maximum context length" XXX check for this?
                log.write(f"Received {e}: giving up\n")
                print(f"Received {e}: giving up")
                delete_last_test()
                return

        response_message = response["choices"][0]["message"]

        if response_message['content']:
            global total_tokens
            tokens = response['usage']['total_tokens']
            print(f"received response; usage: {total_tokens}+{tokens} = {total_tokens+tokens}")
            total_tokens += tokens
            log.write(response_message['content'] + "\n---\n")
            log.write(f"usage: {total_tokens}+{tokens} = {total_tokens+tokens}\n")

        messages.append(response_message)

        if '```python' in response_message['content']:
            import re
            m = re.search('^```python(.*?)^```$', response_message['content'], re.M|re.S)
            if m:
                last_test = m.group(1)
        else:
            log.write("No Python code in GPT response, giving up\n---\n")
            print("No Python code in GPT response, giving up")
            delete_last_test()
            break

        try:
            result = measure_coverage(last_test)

            new_covered = set(result[seg.filename]['executed_lines']) if seg.filename in result else set()
            now_missing = seg.missing_lines - new_covered

            print(f"Originally missing: {list(seg.missing_lines)}")
            print(f"Still missing:      {list(now_missing)}")

            # XXX insist on len(now_missing) == 0 while saving best test?
            if len(now_missing) < len(seg.missing_lines):
                # good 'nough
                # FIXME change this to save_test and writing to temporary files while trying
                get_test_path(advance=True)
                break

            messages.append({
                "role": "user",
                "content": f"""
This test still lacks coverage: {'lines' if len(now_missing)>1 else 'line'}
{", ".join(map(str, now_missing))} still {'do' if len(now_missing)>1 else 'does'} not execute.
"""
            })
            log.write(messages[-1]['content'] + "\n---\n")

        except subprocess.TimeoutExpired:
            log.write("measure_coverage timed out, giving up\n---\n")
            print("measure_coverage timed out, giving up")
            delete_last_test()
            break

        except subprocess.CalledProcessError as e:
            print("causes error.")
            messages.append({
                "role": "user",
                "content": "Executing the test yields an error:\n\n" + str(e.output, 'UTF-8')
            })
            log.write(messages[-1]['content'] + "\n---\n")


if __name__ == "__main__":
    segments = sorted(get_missing_coverage(args.cov_json), key=lambda seg: len(seg.missing_lines), reverse=True)
    total = sum(map(lambda seg: len(seg.missing_lines), segments))

    checkpoint_file = Path(PREFIX + "-checkpoint.json")
    done = set()

    if args.checkpoint:
        try:
            with checkpoint_file.open("r") as f:
                ckpt = json.load(f)
                done = set(ckpt['done'])
                total_tokens = ckpt['total_tokens']
        except json.decoder.JSONDecodeError:
            pass
        except FileNotFoundError:
            pass

    for seg in segments:
        if not seg.filename.startswith(args.source_dir) or \
           seg.missing_lines.issubset(done):
            continue

        if args.source_file and args.source_file not in seg.filename:
            print(f"skipping {seg}")
            continue

        improve_coverage(seg)

        done.update(seg.missing_lines)
        if args.checkpoint:
            with checkpoint_file.open("w") as f:
                json.dump({
                    'done': list(done),
                    'total_tokens': total_tokens
                }, f)

        print(f"{len(done)}/{total}  {len(done)/total:.0%}")
