"""
io/serializer.py — File I/O for TolTransform project files.

Provides save_project() and load_project() for persisting and restoring
ProjectModel instances as human-readable, diff-friendly JSON.

Error handling:
    All failures (file not found, malformed JSON, schema validation, version
    mismatch, dangling references) raise ProjectLoadError with a human-readable
    one-sentence message — not a raw stack trace. The underlying exception is
    always attached as __cause__ for debugging.
"""

from __future__ import annotations

import pydantic

from io.schema import ProjectModel

EXPECTED_SCHEMA_VERSION = 1


class ProjectLoadError(Exception):
    """Raised when a project file cannot be loaded or fails validation."""
    pass


def save_project(project: ProjectModel, path: str) -> None:
    """Serialize a ProjectModel to a JSON file.

    The output is formatted with indent=2 for human readability and clean diffs.

    Parameters
    ----------
    project : ProjectModel
    path    : str — destination file path (will be created or overwritten)
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(project.model_dump_json(indent=2))


def load_project(path: str) -> ProjectModel:
    """Load and validate a ProjectModel from a JSON file.

    Performs three sequential checks:
      1. File exists and is readable.
      2. Content is valid JSON and passes Pydantic schema validation (including
         the cross-reference validator in ProjectModel.validate_references).
      3. schema_version matches EXPECTED_SCHEMA_VERSION.

    All failures raise ProjectLoadError with a clear, actionable message.

    Parameters
    ----------
    path : str — path to the .json project file

    Returns
    -------
    ProjectModel — fully validated

    Raises
    ------
    ProjectLoadError
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = f.read()
    except FileNotFoundError as exc:
        raise ProjectLoadError(
            f"Project file not found: '{path}'"
        ) from exc

    try:
        project = ProjectModel.model_validate_json(raw)
    except pydantic.ValidationError as exc:
        raise ProjectLoadError(
            f"Project file '{path}' failed schema validation:\n{exc}"
        ) from exc
    except Exception as exc:
        raise ProjectLoadError(
            f"Could not parse project file '{path}': {exc}"
        ) from exc

    if project.schema_version != EXPECTED_SCHEMA_VERSION:
        raise ProjectLoadError(
            f"Project file uses schema_version {project.schema_version}, but this "
            f"tool expects schema_version {EXPECTED_SCHEMA_VERSION}. "
            "The file may have been created by a different version of TolTransform."
        )

    return project
