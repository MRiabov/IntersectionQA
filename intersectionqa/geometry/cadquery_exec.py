"""CadQuery execution availability checks.

Untrusted source execution belongs in isolated workers. The MVP exposes this
module only to report local dependency availability and avoid accidental
in-process CADEvolve execution.
"""

from __future__ import annotations

import importlib.util


def cadquery_available() -> bool:
    return importlib.util.find_spec("cadquery") is not None


def cadquery_version() -> str | None:
    if not cadquery_available():
        return None
    import cadquery  # type: ignore[import-not-found]

    return getattr(cadquery, "__version__", None)
