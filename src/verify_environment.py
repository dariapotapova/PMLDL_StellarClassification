"""Fail fast unless the canonical interpreter and package lock are active.

References
----------
Python importlib.metadata documentation:
https://docs.python.org/3.11/library/importlib.metadata.html

The validation logic is original project code.
"""

from __future__ import annotations

import importlib.metadata
import sys
from pathlib import Path

from packaging.version import Version

EXPECTED_PYTHON = (3, 11, 0)
EXPECTED_PIP = "26.2.1"


def locked_versions(lock_path: Path) -> dict[str, str]:
    """Parse the project's simple name-and-version lock file."""

    versions = {}
    for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        name, version = line.split("==", maxsplit=1)
        versions[name] = version
    return versions


def main() -> None:
    """Validate Python, pip, and every locked distribution."""

    if sys.version_info[:3] != EXPECTED_PYTHON:
        raise SystemExit(
            f"Expected Python {'.'.join(map(str, EXPECTED_PYTHON))}, "
            f"received {sys.version.split()[0]}."
        )
    root = Path(__file__).resolve().parents[1]
    failures = []
    expected = locked_versions(root / "requirements.lock.txt")
    expected["pip"] = EXPECTED_PIP
    for distribution, expected_version in expected.items():
        try:
            actual = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            failures.append(f"{distribution}: missing")
            continue
        comparable = Version(actual).base_version if distribution == "torch" else actual
        if comparable != expected_version:
            failures.append(
                f"{distribution}: expected {expected_version}, received {actual}"
            )
    if failures:
        raise SystemExit("Environment verification failed:\n" + "\n".join(failures))
    print(
        f"Verified Python {sys.version.split()[0]} and {len(expected)} locked packages."
    )


if __name__ == "__main__":
    main()
