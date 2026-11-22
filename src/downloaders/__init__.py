"""Manga downloaders (Nyaa.si torrent + selective volume extraction)."""

from src.downloaders.nyaa import NyaaDownloader, parse_nyaa_view_id

__all__ = ["NyaaDownloader", "parse_nyaa_view_id"]
