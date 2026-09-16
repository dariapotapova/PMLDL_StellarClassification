"""Scan submission content for common identity and absolute-path leaks.

References
----------
The implementation in this file is original project code; no external source
was used.
"""

from __future__ import annotations

import re
from pathlib import Path

EXCLUDED_DIRECTORIES = {
    ".git",
    ".venv",
    ".pytest_cache",
    ".ruff_cache",
    "__pycache__",
    "reproduction_outputs",
}
TEXT_SUFFIXES = {
    "",
    ".csv",
    ".dockerignore",
    ".gitignore",
    ".ipynb",
    ".json",
    ".md",
    ".py",
    ".sh",
    ".txt",
    ".yml",
    ".yaml",
}
PATTERNS = {
    "email address": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "macOS personal path": re.compile("/" + r"Users/[^/\s\"']+"),
    "Linux home path": re.compile("/" + r"home/[^/\s\"']+"),
    "Windows personal path": re.compile(r"[A-Za-z]:\\\\" + r"Users\\\\[^\\\s\"']+"),
}


def main() -> None:
    """Fail if a likely leak is found in files eligible for submission."""

    root = Path(__file__).resolve().parents[1]
    findings = []
    for path in root.rglob("*"):
        if not path.is_file() or any(part in EXCLUDED_DIRECTORIES for part in path.parts):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for label, pattern in PATTERNS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{path.relative_to(root)}:{line}: {label}")
    if findings:
        raise SystemExit("Anonymity audit failed:\n" + "\n".join(findings))
    print("Anonymity audit passed for submission-eligible text files.")


if __name__ == "__main__":
    main()