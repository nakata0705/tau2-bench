#!/usr/bin/env python3
"""Cheap, LLM-free test runner for the ``business_interview`` domain.

This is the *default* dev workflow for business_interview changes: it runs the
domain's deterministic pytest (which makes **no live LLM API calls at all**).
Live-LLM verification is opt-in, expensive, and must be requested explicitly
with the ``smoke`` mode.

Usage
-----
    uv run python scripts/test_business_interview.py quick [-- PYTEST_ARGS...]
    uv run python scripts/test_business_interview.py file PATH [-- PYTEST_ARGS...]
    uv run python scripts/test_business_interview.py smoke [-- SMOKE_ARGS...]

Modes
-----
quick  (DEFAULT for development: cheap, deterministic, zero LLM API calls)
    Runs every deterministic pytest under
    ``tests/test_domains/test_business_interview/`` with compact output
    (``-q --tb=short``) and fail-fast (stop at the first failure). Pass extra
    pytest args after ``--`` to narrow further — ``-k KEYWORD`` matches test
    names, e.g.::

        uv run python scripts/test_business_interview.py quick -- -k concept

file    (deterministic, targeted)
    Runs ONE specific file or directory under ``tests/``. A safety check
    rejects any target that resolves outside ``tests/`` so arbitrary repo
    code outside the test tree is never executed::

        uv run python scripts/test_business_interview.py file \\
            tests/test_domains/test_business_interview/test_graph_business_interview.py \\
            -- -k evidence

smoke   (EXPENSIVE, explicit opt-in, live LLM, ONE run only)
    Runs the real end-to-end pipeline (Interview Agent <-> Stakeholder LLM)
    via ``scripts/business_interview_real_llm_smoke.py`` for exactly ONE run.
    It makes live, non-deterministic API calls and costs money/tokens. It is
    never invoked automatically and never as part of ``make test``. Use it
    only to validate end-to-end behaviour that deterministic tests cannot
    cover, and never repeat it for the same code state without reason.

    The wrapper always forces ``--runs 1`` regardless of what you pass, so a
    mistaken ``--runs N`` can never multiply token cost. Put extra smoke args
    after ``--``::

        uv run python scripts/test_business_interview.py smoke -- --seed-base 2000
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
TESTS_ROOT = REPO_ROOT / "tests"
BI_TEST_DIR = TESTS_ROOT / "test_domains" / "test_business_interview"
SMOKE_SCRIPT = REPO_ROOT / "scripts" / "business_interview_real_llm_smoke.py"

# Minimal ANSI styling (safe to print without a terminal).
_BOLD = "\x1b[1m"
_RED = "\x1b[31m"
_GREY = "\x1b[90m"
_END = "\x1b[0m"


def _split_extra(argv: list[str]) -> tuple[list[str], list[str]]:
    """Split ``argv`` at the first ``--`` into (our-args, passthrough-args)."""
    if "--" in argv:
        idx = argv.index("--")
        return argv[:idx], argv[idx + 1 :]
    return argv, []


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="test_business_interview",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="mode", metavar="MODE", required=True)
    sub.add_parser(
        "quick",
        help="cheap DEFAULT: deterministic business_interview pytest (no LLM API)",
    )
    file_p = sub.add_parser(
        "file", help="run one deterministic test file/dir under tests/"
    )
    file_p.add_argument("path", help="file or dir under tests/ to run")
    sub.add_parser(
        "smoke",
        help="EXPENSIVE opt-in live-LLM end-to-end smoke test (forced to 1 run)",
    )
    return parser


def _is_within(path: Path, parent: Path) -> bool:
    """True if ``path`` is ``parent`` or resolves inside it."""
    return path.is_relative_to(parent)


def _run(command: list[str]) -> int:
    print(
        f"\n{_RED}Running:{_END} {' '.join(command)}\n",
        file=sys.stderr,
    )
    return subprocess.run(command, check=False).returncode


def _quick(extra: list[str]) -> int:
    if not BI_TEST_DIR.is_dir():
        raise SystemExit(f"business_interview test dir not found: {BI_TEST_DIR}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        str(BI_TEST_DIR),
        "-q",  # compact output
        "--tb=short",  # compact tracebacks
        "-x",  # fail-fast: stop at the first failure/error
        *extra,
    ]
    print(
        f"{_BOLD}quick{_END} :: deterministic pytest only — no LLM API calls.\n"
        f"{_GREY}  dir     : {BI_TEST_DIR.relative_to(REPO_ROOT)}{_END}",
        file=sys.stderr,
    )
    if extra:
        print(
            f"{_GREY}  pytest  : {' '.join(extra)}{_END}",
            file=sys.stderr,
        )
    return _run(command)


def _file(path: str, extra: list[str]) -> int:
    raw = Path(path)
    target = raw if raw.is_absolute() else (REPO_ROOT / raw)
    target = target.resolve()
    tests_root = TESTS_ROOT.resolve()
    if not _is_within(target, tests_root):
        raise SystemExit(
            f"refusing to run {target}: {_GREY}must resolve inside tests/{_END}"
        )
    if not target.exists():
        raise SystemExit(f"target does not exist: {target}")
    command = [sys.executable, "-m", "pytest", str(target), *extra]
    print(
        f"{_BOLD}file{_END} :: deterministic pytest target — no LLM API calls.\n"
        f"{_GREY}  target     : {target.relative_to(REPO_ROOT)}{_END}",
        file=sys.stderr,
    )
    if extra:
        print(f"{_GREY}  pytest   : {' '.join(extra)}{_END}", file=sys.stderr)
    return _run(command)


def _strip_runs(extra: list[str]) -> list[str]:
    """Drop any user ``--runs`` (or ``--runs=N``) so the wrapper can force 1."""
    out: list[str] = []
    i = 0
    while i < len(extra):
        if extra[i] == "--runs":
            i += 2  # skip value
            continue
        if extra[i].startswith("--runs="):
            i += 1
            continue
        out.append(extra[i])
        i += 1
    return out


def _smoke(extra: list[str]) -> int:
    print(
        f"\n{_RED}{_BOLD}WARNING: live-LLM end-to-end smoke test.{_END}\n"
        f"{_RED}This makes real, non-deterministic API calls and {_BOLD}costs "
        f"money/tokens{_END}{_RED}.{_END}\n"
        f"{_RED}The wrapper forces exactly ONE run and it is never run "
        f"automatically.{_END}\n"
        f"{_GREY}target   : {SMOKE_SCRIPT.name} (task=quotation_workflow_1, "
        f"agent + stakeholder on the env-configured provider){_END}",
        file=sys.stderr,
    )
    command = [sys.executable, str(SMOKE_SCRIPT), *_strip_runs(extra), "--runs", "1"]
    return _run(command)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args, extra = _split_extra(argv)
    parser = _build_parser()
    ns = parser.parse_args(args)
    if ns.mode == "quick":
        return _quick(extra)
    if ns.mode == "file":
        return _file(ns.path, extra)
    if ns.mode == "smoke":
        return _smoke(extra)
    parser.error(f"unknown mode: {ns.mode!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
