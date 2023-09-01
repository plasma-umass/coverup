import openai
import json
import subprocess
import os
import ast
from collections import defaultdict
from pathlib import Path


PREFIX = 'coverup'
openai.key=os.environ['OPENAI_API_KEY']
openai.organization=os.environ['OPENAI_PLASMA_ORG'] # FIXME


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


def get_missing_coverage(jsonfile, base_path = ''):
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

        return (begin, node.end_lineno)

    for fname in cov['files']:
        with open(base_path + fname, "r") as src:
            tree = ast.parse(src.read(), base_path + fname)

        code_this_file = defaultdict(set)
        ctx_this_file = defaultdict(set)

        for line in cov['files'][fname]['missing_lines']:
            if node := find_enclosing(tree, line):
                begin, end = get_line_range(node)

                context = []

                # FIXME make configurable; base it on tokens, not lines
                while end - begin > 100:
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
                name, first, last = it
                missing = code_this_file[it]
                code_segs.append([fname, name, (first, last), list(missing), ctx_this_file[it]])

    return code_segs


def measure_coverage(test: str):
    import sys

    test_path = get_test_path()
    json_path = Path("/tmp") / f"{test_path.stem}.json"     # FIXME use temporary for json

    with test_path.open("w") as test_file:
        test_file.write(test)

    p = subprocess.run(f"{sys.executable} -m slipcover --json --out {json_path} -m pytest {test_path}".split(),
                       check=True, capture_output=True, timeout=60)
    log.write(str(p.stdout, 'UTF-8') + "\n")

    with json_path.open() as j:
        cov = json.load(j)

    return cov["files"]


def improve_coverage(missing, base_path = ''):
    fname, objname, line_range, missing_lines, context = missing
    print(f"\n=== {objname} ({fname}) ===")

    excerpt = []
    with open(base_path + fname, "r") as src:
        code = src.readlines()

        for b, e in context:
            for i in range(b, e):
                excerpt.extend([f"{i:10}: ", code[i-1]])

        b, e = line_range
        for i in range(b, e+1):
            excerpt.extend([f"{i:10}: ", code[i-1]])

    excerpt = ''.join(excerpt)

    messages = [{"role": "user",
                 "content": f"""
The code below, extracted from {fname}, does not achieve full line coverage:
when tested, {'lines' if len(missing_lines)>1 else 'line'} {", ".join(map(str, missing_lines))}
{'do' if len(missing_lines)>1 else 'does'} not execute.
Create a new pytest test function that executes these missing lines, always checking
using the provided function that the new version is correct and indeed improves
coverage. Do not propose a new version without first checking that it is correct
and that it provides improved coverage. Always send entire Python test scripts
when proposing a new test or correcting one you previously proposed.
```python
{excerpt}
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
                    openai.error.RateLimitError) as e:
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
            print(f"received \"{response_message['content'][:75]}...\"")
            log.write(response_message['content'] + "\n---\n")

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
            result = result[fname]

            orig_missing = set(missing_lines)
            new_covered = set(result['executed_lines'])
            now_missing = orig_missing - new_covered

            print(f"Previously missing: {list(orig_missing)}")
            print(f"Still missing:      {list(now_missing)}")

            if len(now_missing) < len(orig_missing):
                # good 'nough
                get_test_path(advance=True)
                break

            messages.append({
                "role": "user",
                "content": f"""
This test does not improve coverage: {'lines' if len(missing_lines)>1 else 'line'}
{", ".join(map(str, missing_lines))} still {'do' if len(missing_lines)>1 else 'does'} not execute.
"""
            })
            log.write(messages[-1]['content'] + "\n---\n")

        except subprocess.TimeoutExpired:
            log.write("measure_coverage timed out, giving up\n---\n")
            print("measure_coverage timed out, giving up")
            delete_last_test()
            break

        except subprocess.CalledProcessError as e:
            messages.append({
                "role": "user",
                "content": "Executing the test yields an error:\n\n" + str(e.output, 'UTF-8')
            })
            log.write(messages[-1]['content'] + "\n---\n")


for m in get_missing_coverage(args.cov_json):
    fname, objname, line_range, missing_lines, context = m
    if not fname.startswith(args.source_dir):
        continue

    if args.source_file and args.source_file not in fname:
        print(f"skipping {m}")
        continue

    improve_coverage(m)
