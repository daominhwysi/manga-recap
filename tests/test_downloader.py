"""Tests for Nyaa.si URL parsing, selective volume matching, and CBZ extraction."""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.downloaders.nyaa import NyaaDownloader, parse_nyaa_view_id


def test_parse_nyaa_view_id() -> None:
    assert parse_nyaa_view_id("https://nyaa.si/view/1246645") == "1246645"
    assert parse_nyaa_view_id("1246645") == "1246645"


def test_volume_filter_matching() -> None:
    entries = [
        (0, "Pandora Hearts/v13_Pandora.cbz", 100),
        (1, "Pandora Hearts/v01_Pandora.cbz", 100),
        (2, "Pandora Hearts/v02_Pandora.cbz", 100),
    ]
    matched = NyaaDownloader._match_volume_indices(entries, "vol1")
    assert matched == [1]


def test_extract_cbz_archive(tmp_path: Path) -> None:
    cbz = tmp_path / "v01.cbz"
    with zipfile.ZipFile(cbz, "w") as zf:
        zf.writestr("002.jpg", b"img2")
        zf.writestr("001.jpg", b"img1")

    extracted = NyaaDownloader.extract_archives([cbz], tmp_path / "out")
    vol_dir = extracted / "v01"
    assert (vol_dir / "p0001.jpg").read_bytes() == b"img1"
    assert (vol_dir / "p0002.jpg").read_bytes() == b"img2"
