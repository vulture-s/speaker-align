"""The version the package reports must be the version it was published as.

Not hypothetical: this package's sibling, whisper-guard, shipped 0.3.0 to PyPI
with `Version: 0.3.0` in its metadata and `__version__ = "0.2.0"` inside the
wheel. The number lived in two places, one of them was bumped, and nothing
compared them — `pip show` and `whisper_guard.__version__` disagreed for two
months before anyone opened the wheel.

speaker-align had the same two-source layout. It is fixed here *before* the
first release, and these tests keep it fixed.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

import speaker_align


def test_installed_metadata_matches_dunder_version():
    from importlib.metadata import PackageNotFoundError, version

    try:
        installed = version("speaker-align")
    except PackageNotFoundError:
        pytest.skip("speaker-align is not installed in this environment")

    assert installed == speaker_align.__version__, (
        "installed metadata says %s but the package reports %s — "
        "the wheel was built from a different tree than it claims"
        % (installed, speaker_align.__version__)
    )


def test_pyproject_has_no_second_version_source():
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present (installed-only environment)")
    text = pyproject.read_text(encoding="utf-8")

    assert re.search(r"^version\s*=\s*[\"']", text, re.M) is None, (
        "pyproject.toml pins a literal version again — that is the second "
        "source that broke whisper-guard 0.3.0"
    )
    assert 'dynamic = ["version"]' in text
    assert "[tool.hatch.version]" in text


def test_sdist_excludes_local_environments():
    """A packaged virtualenv is how whisper-guard's sdist reached 1.5 MB."""
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    if not pyproject.exists():
        pytest.skip("pyproject.toml not present (installed-only environment)")
    text = pyproject.read_text(encoding="utf-8")

    assert "[tool.hatch.build.targets.sdist]" in text
    assert ".venv-*" in text


def test_version_is_pep440_sortable():
    assert re.fullmatch(r"\d+\.\d+\.\d+([abrc].*)?", speaker_align.__version__)
