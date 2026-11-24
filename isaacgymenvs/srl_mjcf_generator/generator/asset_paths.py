"""Resolve MJCF filenames for the Isaac Gym importer."""
from pathlib import Path


def gym_asset_path(asset_root, asset_file):
    # Gym dispatches on the asset filename. A leading ".." can be interpreted
    # as its extension, so re-root experiment assets at their mjcf directory.
    if asset_file.startswith('mjcf/'):
        return asset_root, asset_file
    path = (Path(asset_root) / asset_file).resolve()
    for parent in path.parents:
        if parent.name == 'mjcf':
            return str(parent.parent), str(path.relative_to(parent.parent))
    return str(path.parent), path.name
