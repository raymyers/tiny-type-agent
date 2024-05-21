"""
# Tiny Type Agent

Minimal example of a safe Autonomous DevTool for nopilot.dev blog.

Adapted from the Menderbot project by Ray Myers.
https://github.com/craftvscruft/menderbot

LICENSE MIT or Apache-2 at your option
"""

from dataclasses import dataclass
import itertools
import logging
import os
from pathlib import Path
import re

import sys
import tempfile
from typing import Iterable
import subprocess
import argparse


from charset_normalizer import from_path

# Tree Sitter is used here as a potential cross-language approach.
# For a Python-only tool, LibCST or Rope would be a better choice.
from tree_sitter import Language, Parser
import tree_sitter_python as tspython

import openai


openai.api_key = os.getenv("OPENAI_API_KEY")

if os.getenv("OPENAI_API_BASE"):
    openai.api_base = os.getenv("OPENAI_API_BASE")

if os.getenv("OPENAI_ORG"):
    openai.organization = os.getenv("OPENAI_ORG")

INSTRUCTIONS = (
    """You are helpful electronic assistant with knowledge of Software Engineering."""
)

TEMPERATURE = 0.5
MAX_TOKENS = 500
FREQUENCY_PENALTY = 0
PRESENCE_PENALTY = 0.6
# limits how many questions we include in the prompt
MAX_CONTEXT_QUESTIONS = 10


def is_test_override() -> bool:
    return (
        os.getenv("OPENAI_API_KEY")
        == "sk-TEST00000000000000000000000000000000000000000000"
    )


def override_response_for_test(messages) -> str:
    del messages
    return "<LLM Output>"


def get_response(
    instructions: str, new_question: str
) -> str:
    """Get a response from ChatCompletion

    Args:
        instructions: The instructions for the chat bot - this determines how it will behave
        previous_questions_and_answers: Chat history
        new_question: The new question to ask the bot

    Returns:
        The response text
    """
    messages = [
        {"role": "system", "content": instructions},
    ]
    messages.append({"role": "user", "content": new_question})

    if is_test_override():
        return override_response_for_test(messages)
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        top_p=1,
        frequency_penalty=FREQUENCY_PENALTY,
        presence_penalty=PRESENCE_PENALTY,
    )
    return completion.choices[0].message.content


def run_check(command: str) -> tuple[bool, str]:
    try:
        return (
            True,
            subprocess.check_output(
                command, stderr=subprocess.STDOUT, shell=True, text=True
            ),
        )
    except subprocess.CalledProcessError as e:
        return (False, e.output)


logger = logging.getLogger("typing")
PY_LANGUAGE = Language(tspython.language())


def parse_source_to_tree(source: bytes, language: Language):
    parser = Parser()
    parser.set_language(language)

    return parser.parse(source)

@dataclass
class Insertion:
    text: str
    line_number: int
    label: str
    col: int = -1  # Use with `inline`
    inline: bool = False  # Insert into existing line instead of adding new line

def partition(pred, iterable):
    "Use a predicate to partition entries into false entries and true entries"
    # partition(is_odd, range(10)) --> 0 2 4 6 8   and  1 3 5 7 9
    t1, t2 = itertools.tee(iterable)
    return itertools.filterfalse(pred, t1), filter(pred, t2)


def insert_in_lines(lines: Iterable[str], insertions: Iterable[Insertion]):
    """
    Performs the insertions on the given lines.
    Handles both full line insertions and inline insertions.
    """
    lines = iter(lines)
    last_line = 1
    insertion_groups = itertools.groupby(insertions, key=lambda ins: ins.line_number)
    for line_number, insertion_group in insertion_groups:
        for line in itertools.islice(lines, line_number - last_line):
            yield line
            last_line += 1
        full_insertions, inline_insertions = partition(
            lambda ins: ins.inline, insertion_group
        )
        for insertion in full_insertions:
            yield insertion.text + "\n"

        line_to_edit = None
        col_offset = 0
        for insertion in inline_insertions:
            if not line_to_edit:
                line_to_edit = next(lines, "")
            col = insertion.col + col_offset
            col_offset += len(insertion.text)
            line_to_edit = line_to_edit[:col] + insertion.text + line_to_edit[col:]
        if line_to_edit:
            yield line_to_edit
            last_line += 1
    yield from lines


class SourceFile:
    def __init__(self, path: str):
        self.path = path
        self.encoding = None
        self._initial_modified_time = os.path.getmtime(path)

    def load_source_as_utf8(self):
        loaded = from_path(self.path)
        best_guess = loaded.best()
        self.encoding = best_guess.encoding
        return best_guess.output(encoding="utf_8")

    def is_unicode(self):
        return self.encoding.startswith("utf")

    def update_file(self, insertions: Iterable[Insertion], suffix: str) -> None:
        """
        Perform all given insertions on the updated file.
        """
        path_obj = Path(self.path)
        with path_obj.open("r", encoding=self.encoding) as filehandle:
            if self.modified_after_loaded():
                raise Exception(
                    f"File was externally modified, try again. {self.path}"
                )
            new_lines = list(insert_in_lines(lines=filehandle, insertions=insertions))
            out_file = path_obj.with_suffix(f"{path_obj.suffix}{suffix}")
            self._write_result(new_lines, out_file)

    def _write_result(self, lines: list, output_file: Path) -> None:
        with tempfile.TemporaryDirectory() as tempdir:
            my_tempfile: Path = Path(tempdir) / "output.txt"
            with my_tempfile.open("w") as filehandle:
                for line in lines:
                    filehandle.write(line)
            my_tempfile.replace(output_file)

    def modified_after_loaded(self) -> bool:
        return os.path.getmtime(self.path) > self._initial_modified_time



class PythonLanguageStrategy():
    def function_has_comment(self, node) -> bool:
        """Checks if function has a docstring. Example node:
        (function_definition name: (identifier) parameters: (parameters)
           body: (block (expression_statement
             (string string_content: (string_content))) (pass_statement)))
        """
        body_node = node.child_by_field_name("body")
        if body_node and body_node.type == "block":
            if (
                body_node.child_count > 0
                and body_node.children[0].type == "expression_statement"
            ):
                expression_statement_node = body_node.children[0]
                if expression_statement_node.child_count > 0:
                    return expression_statement_node.children[0].type == "string"
        return False

    def parse_source_to_tree(self, source: bytes):
        return parse_source_to_tree(source, PY_LANGUAGE)

    def get_node_declarator_name(self, node) -> str:
        name_node = node.child_by_field_name("name")
        return str(name_node.text, encoding="utf-8")

    def get_function_nodes(self, tree) -> list:
        query = PY_LANGUAGE.query(
            """
        (function_definition name: (identifier)) @function
        """
        )
        captures = query.captures(tree.root_node)
        return [capture[0] for capture in captures]

    function_doc_line_offset = 1

def node_str(node) -> str:
    return str(node.text, encoding="utf-8")


def parse_type_hint_answer(text: str) -> list:
    def line_to_tuple(line: str) -> tuple:
        [ident, new_type] = line.split(":")
        new_type = re.sub(r"\bList\b", "list", new_type)
        return (ident.strip(), new_type.strip())

    lines = text.strip().splitlines()
    hints = [line_to_tuple(line) for line in lines if ":" in line]
    return [hint for hint in hints if hint[0] != "self" and hint[1].lower() != "any"]


def add_type_hints(function_node, hints: list) -> list:
    function_name = node_str(function_node.child_by_field_name("name"))
    params_node = function_node.child_by_field_name("parameters")
    return_type_node = function_node.child_by_field_name("return_type")
    insertions = []

    for ident, new_type in hints:
        for param_node in params_node.children:
            if param_node.type in ["identifier"] and node_str(param_node) == ident:
                line = param_node.end_point[0] + 1
                col = param_node.end_point[1]

                insertions.append(
                    Insertion(
                        text=f": {new_type}",
                        line_number=line,
                        col=col,
                        inline=True,
                        label=function_name,
                    )
                )
        if ident == "return" and not return_type_node:
            line = params_node.end_point[0] + 1
            col = params_node.end_point[1]
            insertions.append(
                Insertion(
                    text=f" -> {new_type}",
                    line_number=line,
                    col=col,
                    inline=True,
                    label=function_name,
                )
            )
    return insertions


def process_untyped_functions(source_file: SourceFile):
    """
    returns list of tuple (node, function_text, needs_typing)
    """
    path = source_file.path
    logger.info('Processing "%s"...', path)
    _, file_extension = os.path.splitext(path)
    if not file_extension == ".py":
        logger.info('"%s" is not a Python file, skipping.', path)
        return
    language_strategy = PythonLanguageStrategy()
    source = source_file.load_source_as_utf8()
    tree = language_strategy.parse_source_to_tree(source)
    return process_untyped_functions_in_tree(tree, language_strategy)


def process_untyped_functions_in_tree(tree, language_strategy):
    """yields (tree, node, function_text, needs_typing)"""
    for node in language_strategy.get_function_nodes(tree):
        name = node_str(node.child_by_field_name("name"))
        params_node = node.child_by_field_name("parameters")
        return_type_node = node.child_by_field_name("return_type")
        needs_typing = [
            node_str(n) for n in params_node.children if n.type in ["identifier"]
        ]
        needs_typing = [n for n in needs_typing if n != "self"]
        # Add "default_parameter" later probably.
        return_type_text = ""
        if return_type_node:
            return_type_text = " -> " + node_str(return_type_node)
        elif name != "__init__":
            needs_typing.append("return")
        params = node_str(params_node)
        print()
        print(f"def {name}{params}{return_type_text}")

        if needs_typing:
            function_text = node_str(node)
            # Should make an object
            yield (node, function_text, needs_typing)

        # https://github.com/tree-sitter/tree-sitter-python/blob/master/grammar.js
        # print(node.sexp())


def type_prompt(function_text: str, needs_typing: list, previous_error: str) -> str:
    """Create prompt for guessing types"""
    needs_typing_text = ",".join(needs_typing)
    # Do not assume the existence any unreferenced classes outside the standard library unless you see.
    return f"""
Please infer these missing Python type hints. 
If you cannot determine the type with confidence, use 'any'. 
The lowercase built-in types available include: int, str, list, set, dict, tuple. 
You will be shown a previous error message from the type-checker with useful clues.

Input:
```
def foo(a, b: int, unk):
return a + b
```
Previous error: 
```
error: Argument 3 to "foo" has incompatible type "LightBulb"; expected "NoReturn"  [arg-type]
```
Infer: a, unk, return
Output:
a: int
unk: LightBulb
return: int

Input:
```
{function_text}
```
Previous error:
```
{previous_error}
```
Infer: {needs_typing_text}
Output:
"""

def try_function_type_hints(source_file, function_node, function_text, needs_typing, check):
    
    mypy_args = "--no-error-summary --soft-error-limit 10"
    check_command = (
        f"{check} {mypy_args} --shadow-file {source_file.path} {source_file.path}.shadow"
    )
    max_tries = 2
    check_output = None
    # First set them all to wrong type, to produce an error message.
    hints = [(ident, "None") for ident in needs_typing]
    insertions_for_function = add_type_hints(function_node, hints)
    if insertions_for_function:
        source_file.update_file(insertions_for_function, suffix=".shadow")
        (success, pre_check_output) = run_check(check_command)
        if not success:
            check_output = pre_check_output
    for try_num in range(0, max_tries):
        if try_num > 0:
            print("Retrying")
        prompt = type_prompt(function_text, needs_typing, previous_error=check_output)
        answer = get_response(INSTRUCTIONS, prompt)
        hints = parse_type_hint_answer(answer)

        insertions_for_function = add_type_hints(function_node, hints)
        if insertions_for_function:
            print(f"[cyan]Bot[/cyan]: {hints}")
            source_file.update_file(insertions_for_function, suffix=".shadow")
            (success, check_output) = run_check(check_command)
            if success:
                print("[green]Type checker passed[/green], keeping")
                return insertions_for_function
            else:
                print("[red]Type checker failed[/red], discarding")
        else:
            print("[cyan]Bot[/cyan]: No changes")
            # No retry if it didn't try to hint anything.
            return []
    return []



def run(file, check):
    """Insert type hints (Python only)"""
    print(openai.Model.list())
    print("Running type-checker baseline")
    (success, check_output) = run_check(check)
    if not success:
        print(check_output)
        print("Baseline failed, aborting.")
        return
    source_file = SourceFile(file)
    insertions = []
    for function_node, function_text, needs_typing in process_untyped_functions(
        source_file
    ):
        insertions += try_function_type_hints(
            source_file, function_node, function_text, needs_typing, check=check
        )
    if not insertions:
        print(f"No changes for '{file}.")
        return
    sys.stdout.write(f"Write '{file}'?")
    choice = input().lower()
    if not choice == 'y':
        print("Skipping.")
        return
    source_file.update_file(insertions, suffix="")
    print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Tiny Type Agent to add type hints to a Python file.")
    parser.add_argument("file", type=str, help="A Python file to process, e.g. main.py")
    parser.add_argument("check", type=str, help="The command to run the type checker, e.g. './venv/bin/python3 -m mypy *.py'")
    args = parser.parse_args()
    run(args.file, args.check)
