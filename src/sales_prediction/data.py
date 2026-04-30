"""Dataset access helpers.

The raw dataset lives on Google Drive. We download it once and cache it on the
local filesystem to keep subsequent reruns offline and deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .config import DATASET_DIRECT_URL, DATASET_FILENAME, RAW_DIR


def _download_to(target: Path) -> Path:
    """Download the raw CSV to *target* using :mod:`gdown`.

    Kept private so callers go through :func:`load_raw` which adds caching.
    """

    try:
        import gdown
    except ImportError as exc:  # pragma: no cover - import guarded by requirements.txt
        raise RuntimeError(
            "`gdown` is required to fetch the dataset. Install it with `pip install gdown`."
        ) from exc

    target.parent.mkdir(parents=True, exist_ok=True)
    gdown.download(DATASET_DIRECT_URL, str(target), quiet=True)

    if not target.exists() or target.stat().st_size == 0:
        raise RuntimeError(
            f"Download failed for {DATASET_DIRECT_URL}. "
            f"Either Google Drive is rate limiting, or the file id has changed."
        )

    return target


def load_raw(
    path: Path | str | None = None,
    *,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Return the raw sales dataset as a DataFrame.

    Parameters
    ----------
    path:
        Optional explicit path to the CSV file. When *None*, the default cache
        location under ``data/raw/`` is used.
    download_if_missing:
        When True (default) the CSV is downloaded from Google Drive if it is
        not already present on disk. Set to False to force a hard failure on a
        missing cache, which is useful in CI.
    """

    csv_path = Path(path) if path is not None else RAW_DIR / DATASET_FILENAME

    if not csv_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}. Either download it manually or "
                f"call load_raw(download_if_missing=True)."
            )
        _download_to(csv_path)

    return pd.read_csv(csv_path)
