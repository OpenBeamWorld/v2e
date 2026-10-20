# SPDX-FileCopyrightText: 2026 OpenBeam, LLC
# SPDX-License-Identifier: Apache-2.0
"""Layout and hygiene checks for the package scaffold.

These tests pin the package layout, the licence headers, the base-install dependency boundary
and the command surface. They make no claim about pipeline behaviour.
"""

from __future__ import annotations

import ast
import importlib
import pkgutil
import re
import sys
from pathlib import Path

import pytest

import openbeam_v2e
from openbeam_v2e import cli

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "openbeam_v2e"

EXPECTED_PACKAGES = {
    "openbeam_v2e.agents",
    "openbeam_v2e.backends",
    "openbeam_v2e.backends.openbeam",
    "openbeam_v2e.bench",
    "openbeam_v2e.compile",
    "openbeam_v2e.compile.dialects",
    "openbeam_v2e.controllers",
    "openbeam_v2e.fourdcb",
    "openbeam_v2e.generators",
    "openbeam_v2e.generators.levels",
    "openbeam_v2e.generators.props",
    "openbeam_v2e.generators.vehicle",
    "openbeam_v2e.media",
    "openbeam_v2e.perception",
    "openbeam_v2e.perception.adapters",
    "openbeam_v2e.perception.baseline",
    "openbeam_v2e.reconstruct",
    "openbeam_v2e.run",
    "openbeam_v2e.scene",
    "openbeam_v2e.score",
    "openbeam_v2e.search",
}

# Third-party imports a module may make at import time. Optional extras (MCP SDK, model
# libraries) must be imported lazily, inside the function that needs them.
BASE_DEPENDENCIES = {"numpy", "scipy"}

SPDX_LINES = (
    "# SPDX-FileCopyrightText: 2026 OpenBeam, LLC",
    "# SPDX-License-Identifier: Apache-2.0",
)


def _walk_modules() -> list[str]:
    names = ["openbeam_v2e"]
    for info in pkgutil.walk_packages(openbeam_v2e.__path__, prefix="openbeam_v2e."):
        if info.name.endswith(".__main__"):
            continue  # importing it runs the CLI
        names.append(info.name)
    return sorted(names)


def _source_files() -> list[Path]:
    files = sorted(p for p in ROOT.rglob("*.py") if "__pycache__" not in p.parts)
    files.append(ROOT / "pyproject.toml")
    return files


def test_version_is_pep440() -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+(?:(?:a|b|rc)\d+)?(?:\.post\d+)?(?:\.dev\d+)?", openbeam_v2e.__version__)


def test_expected_subpackages_exist() -> None:
    packages = {
        info.name
        for info in pkgutil.walk_packages(openbeam_v2e.__path__, prefix="openbeam_v2e.")
        if info.ispkg
    }
    missing = EXPECTED_PACKAGES - packages
    assert not missing, f"missing subpackages: {sorted(missing)}"


@pytest.mark.parametrize("name", _walk_modules())
def test_module_imports_and_has_docstring(name: str) -> None:
    module = importlib.import_module(name)
    assert module.__doc__ and module.__doc__.strip(), f"{name} needs a docstring"


@pytest.mark.parametrize("path", _source_files(), ids=lambda p: str(p.relative_to(ROOT)))
def test_source_file_has_apache_spdx_header(path: Path) -> None:
    head = path.read_text(encoding="utf-8").splitlines()[:2]
    assert tuple(head) == SPDX_LINES, f"{path.relative_to(ROOT)} must start with the Apache-2.0 SPDX header"


@pytest.mark.parametrize("path", sorted(SRC.rglob("*.py")), ids=lambda p: str(p.relative_to(ROOT)))
def test_import_time_dependencies_stay_in_the_base_install(path: Path) -> None:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    allowed = set(sys.stdlib_module_names) | BASE_DEPENDENCIES | {"openbeam_v2e", "__future__"}
    offenders = []
    for node in tree.body:  # module level only: lazy imports inside functions are allowed
        if isinstance(node, ast.Import):
            roots = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots = [node.module.split(".")[0]]
        else:
            continue
        offenders += [r for r in roots if r not in allowed]
    assert not offenders, f"{path.relative_to(ROOT)} imports {offenders} at import time"


def test_first_release_ships_only_the_openbeam_connector_package() -> None:
    # The fake test connector is a module of `backends`; any further engine connector is a later
    # release and should arrive as a plugin or a reviewed change to this test.
    connectors = {
        info.name
        for info in pkgutil.iter_modules([str(SRC / "backends")])
        if info.ispkg
    }
    assert connectors == {"openbeam"}


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0
    assert capsys.readouterr().out.strip() == f"openbeam-v2e {openbeam_v2e.__version__}"


def test_cli_declares_the_pipeline_commands(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for command in ("reconstruct", "scene", "compile", "run", "export", "score", "inspect", "bench", "mcp"):
        assert command in out


def test_unimplemented_command_exits_2(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["annotate"]) == 2
    assert "not implemented yet" in capsys.readouterr().err
