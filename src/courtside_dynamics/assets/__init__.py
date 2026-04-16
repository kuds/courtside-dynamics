"""MJCF model files for the Courtside Dynamics environments."""
from __future__ import annotations

from importlib.resources import as_file, files
from pathlib import Path


def asset_path(name: str) -> str:
    """Return an absolute filesystem path to a bundled MJCF asset.

    Parameters
    ----------
    name:
        File name of the asset, e.g. ``"ball_balance.xml"``.

    Notes
    -----
    MuJoCo's loader requires a real filesystem path. For editable installs
    (``pip install -e .``) the asset is already on disk and we can return
    the path directly. For wheel installs the resource may live inside a
    zip; in that case we still materialize it to a real file via
    ``importlib.resources.as_file``.
    """
    resource = files(__name__).joinpath(name)
    try:
        return str(Path(str(resource)).resolve(strict=True))
    except (FileNotFoundError, OSError):
        with as_file(resource) as path:
            return str(Path(path).resolve(strict=True))


__all__ = ["asset_path"]
