import re
import json
from pathlib import Path


TERMINAL_EVENTS=('G', 'M', 'T', '-', '*', 'f', 'u')


def is_same_as_P(content, begin, end):
    """Attempts to detect 'C' (context) prompts that were only issued as context prompts
       because their 'def' executed when their module was loaded to 'P' (initial) prompts,
       which is what they should have been."""

    begin, end = int(begin), int(end)

    if (rng := re.search(r'^when tested, lines (\d+)-(\d+) do not execute', content, re.M)) and \
       (py := re.search('```python\n(.*)```', content, re.S)):
        first, last = int(rng.group(1)), int(rng.group(2))

        def del_line_markup(s):
            # CoverUp "   nnn: " line markup
            line_markup_len = 12
            s = '\n'.join(l[line_markup_len:] for l in s.splitlines())

            import textwrap
            return textwrap.dedent(s)

        try:
            import ast
            tree = ast.parse(del_line_markup(py.group(1)))
            block = next(iter(node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
            if block is None:
                block = next(iter(node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)))
        except Exception:
            print("----")
            print(py.group(1))
            print("----")
            print(del_line_markup(py.group(1)))
            print("----")
            raise

        if ast.get_docstring(block, clean=False) is not None:
            block.body.pop(0)

        for first_stmt in block.body:
            if not isinstance(first_stmt, (ast.Global, ast.Nonlocal)):
                break

        last_stmt = block.body[-1]

        block.lineno = min([block.lineno, *(d.lineno for d in block.decorator_list)])

        # CoverUp may include "class" statements in a segment, or it may start a segment
        # within the class and just emit "class" statements for context. We have no way
        # of knowing which one we're seeing, so we need to look for both cases (below).

        # We also need to allow the last line to fall anywhere within the last statement
        # because of things like:
        #     return (
        #         foo
        #     )

        if (begin - 1 + first_stmt.lineno == first and
              (begin - 1 + last_stmt.lineno <= last and
               begin - 1 + last_stmt.end_lineno >= last)):
            # segment started at line 1
            return True
        elif (begin - block.lineno + first_stmt.lineno == first and
            (begin - block.lineno + last_stmt.lineno <= last and
             begin - block.lineno + last_stmt.end_lineno >= last)):
            # segment started with the 'def', with other lines for context
            return True
#            else:
#                print("----")
#                print(f"{first=} {last=} {rng.group(0)=}")
#                print(f"{block.lineno=} {begin=}")
#                print(f"{first_stmt=} {begin - block.lineno + first_stmt.lineno} {first}")
#                print(f"{last_stmt=} {begin - block.lineno + last_stmt.end_lineno} {last}")
#                print(py.group(1))

    return False


def parse_log_raw(log_content: str):
    """Yields raw items from a CoverUp log."""
    for m in re.finditer('---- (?:(\S+) )?([\S+ ]+) ----\n\n?(.*?)(?=\n---- |\Z)',
                         log_content, re.DOTALL):
        timestamp, ctx, content = m.groups()
        yield timestamp, ctx, content


def parse_log(log_content: str, check_c_p_equivalence=False):
    """Parses a CoverUp log, categorizing them in different events and extracting information from JSON messages."""

    for timestamp, ctx, content in parse_log_raw(log_content):
        if ctx == 'startup':
            yield timestamp, 'startup', None, content
            continue

        if not (m := re.match('(\S+):(\d+)-(\d+)', ctx)):
            continue

        py, begin, end = m.groups()
        seginfo = (py, int(begin), int(end))

        def what(content: str):
            content = content.lstrip()
            if content.startswith(("The code below,", "You are an expert")): # prompt
                if "\nwhen tested, it does not execute." in content:
                    return 'P'
                if check_c_p_equivalence and is_same_as_P(content, begin, end):
                    return 'p'
                return 'C'
            elif content.startswith("Executing the test yields an error"):
                return 'F'
            elif content.startswith("Test failed"):
                return 'f'
            elif content.startswith("Executing the test along with"): # side effect
                return 'S'
            elif content.startswith("```python"): # response
                return 'R'
            elif content.startswith("This test still lacks coverage"):
                return 'U'
            elif content.startswith("Test doesn't improve coverage"):
                return 'u'
            elif content.startswith("Saved as"): # success
                return 'G'
            elif content.startswith("Missing modules"):
                return 'M'
            elif content.startswith("measure_coverage timed out"):
                return 'T'
            elif content.startswith("No Python code in GPT response"):
                return '-'
            elif content.startswith("Too many attempts"): # gave up
                return '*'

            return '?'

        if content.startswith("{"):
            j = json.loads(content)
            if 'choices' in j:
                if tool_calls := j['choices'][0]['message'].get("tool_calls", None):
                    for call in tool_calls:
                        args = json.loads(call['function']['arguments'])
                        funccall = call['function']['name'] + "(" +\
                                   ", ".join(f"{k}={json.dumps(v)}" for k, v in args.items()) +\
                                   ")"
                        yield timestamp, 'n', seginfo, funccall

                else:
                    content = j['choices'][0]['message']['content']
                    yield timestamp, what(content), seginfo, content

                continue

            if 'messages' not in j:
                continue
            
            messages = j['messages']

            start = max([0, *[i+1 for i in range(len(messages)) if messages[i]['role'] == 'assistant']])
            for msg in messages[start:]:
                if msg['role'] == 'tool':
                    yield timestamp, 'N', seginfo, msg['content']
                else:
                    yield timestamp, what(msg['content']), seginfo, msg['content']
        else:
            yield timestamp, what(content), seginfo, content


def get_sequences(log_content: str, check_c_p_equivalence=True):
    """Collates log events by segment"""

    from collections import defaultdict
    segs: dict[str, list[tuple[str, str, str]]] = defaultdict(list)

    def yield_sequences():
        for seg, seq in list(segs.items()):
            if seq[-1][0] in TERMINAL_EVENTS:
                    yield seg, seq
        segs.clear()

    for ts, ev, details, content in parse_log(log_content, check_c_p_equivalence=check_c_p_equivalence):
        if ev == 'startup':
            # starting over, so flush what we have
            yield from yield_sequences()
            continue

        (py, begin, end) = details

        if ev not in ('?'):    # skip "total usage"
            seg = f"{py}:{begin}-{end}"
            segs[seg].append((ev, ts, content))

    yield from yield_sequences()


def parse_args():
    import argparse
    ap = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ap.add_argument('--errors-only', action="store_true", help='show errors only')

    ap.add_argument('--check-c-p', action="store_true",
                    help='check for C prompts that are equivalent to a C prompt')

    ap.add_argument('logs', type=Path, nargs='+', help='log file(s) to process')
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()

#    max_calls = 0
#    max_seq = ''

    for log in args.logs:
        for seg, seq in get_sequences(log.read_text(), check_c_p_equivalence=args.check_c_p):
            if args.errors_only and seq[-1][0] == 'G':
                continue

            print("=======", seg, ''.join(ev[0] for ev in seq))

            for ev in seq:
                print(f"--- {ev[1]} {ev[0]} ---")
                print(ev[2])

#            evseq = ''.join(ev[0] for ev in seq)
#            for funccalls in re.findall(r'(?:nN)+', evseq):
#                if len(funccalls)//2 > max_calls:
#                    max_calls = len(funccalls)//2
#                    max_seq = evseq

#print(f"{max_calls=}  {max_seq=}")
