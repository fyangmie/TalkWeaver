#!/usr/bin/env python3
"""Report availability of TalkWeaver's optional runtime dependencies."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class OptionalDependency:
    display_name: str
    import_name: str
    purpose: str
    install: str


DEPENDENCIES = {
    "faster-whisper": OptionalDependency(
        display_name="faster-whisper",
        import_name="faster_whisper",
        purpose="real ASR inference",
        install="pip install -r requirements-optional.txt",
    ),
    "pyannote.audio": OptionalDependency(
        display_name="pyannote.audio",
        import_name="pyannote.audio",
        purpose="automatic speaker diarization",
        install="pip install -r requirements-optional.txt",
    ),
    "noisereduce": OptionalDependency(
        display_name="noisereduce",
        import_name="noisereduce",
        purpose="optional spectral denoising",
        install="pip install -r requirements-optional.txt",
    ),
}


def is_import_available(import_name: str) -> bool:
    """Check import metadata without importing heavyweight packages."""

    try:
        return importlib.util.find_spec(import_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def resolve_dependency(name: str) -> OptionalDependency:
    """Resolve known package aliases and arbitrary strict-check targets."""

    normalized = name.strip()
    if normalized in DEPENDENCIES:
        return DEPENDENCIES[normalized]
    import_name = normalized.replace("-", "_")
    return OptionalDependency(
        display_name=normalized,
        import_name=import_name,
        purpose="explicit dependency check",
        install=f"pip install {normalized}",
    )


def check_dependencies(
    names: list[str] | None = None,
) -> list[tuple[OptionalDependency, bool]]:
    selected = names or list(DEPENDENCIES)
    return [
        (dependency, is_import_available(dependency.import_name))
        for dependency in map(resolve_dependency, selected)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strict",
        nargs="*",
        metavar="PACKAGE",
        help=(
            "Exit nonzero when any requested package is missing. "
            "With no package names, check all known optional dependencies."
        ),
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    strict = args.strict is not None
    requested = args.strict if strict and args.strict else None
    results = check_dependencies(requested)

    print("TalkWeaver optional dependency check")
    print(f"Python: {sys.version.split()[0]}")
    for dependency, available in results:
        status = "AVAILABLE" if available else "MISSING"
        print(
            f"[{status}] {dependency.display_name}: "
            f"{dependency.purpose}"
        )
        if not available:
            print(f"  Install: {dependency.install}")

    missing = [
        dependency.display_name
        for dependency, available in results
        if not available
    ]
    if missing and strict:
        print(
            "Strict check failed; missing: " + ", ".join(missing),
            file=sys.stderr,
        )
        return 2
    if missing:
        print(
            "Optional dependencies are missing. Mock and reference-assisted "
            "workflows remain available."
        )
    else:
        print("All requested optional dependencies are available.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
