"""Nyaa.si torrent downloader using libtorrent with selective volume filtering."""

from __future__ import annotations

import re
import time
import zipfile
from pathlib import Path

import httpx
import libtorrent as lt

from src.core.protocols import MangaDownloader
from src.cv.loader import IMAGE_EXTS, natural_sort_key


def parse_nyaa_view_id(source: str) -> str:
    """Extract numeric Nyaa view ID from a URL or raw ID string."""
    s = source.strip()
    if s.isdigit():
        return s
    m = re.search(r"nyaa\.si/(?:view|download)/(\d+)", s)
    if m:
        return m.group(1)
    raise ValueError(f"Unsupported Nyaa source '{source}'. Expected https://nyaa.si/view/<id> or <id>.")


class NyaaDownloader(MangaDownloader):
    """Downloads manga archives from Nyaa.si via .torrent + libtorrent."""

    name = "nyaa"

    def __init__(
        self,
        base_url: str = "https://nyaa.si",
        timeout_sec: int = 600,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout_sec = timeout_sec

    def fetch_torrent_bytes(self, view_id: str) -> bytes:
        url = f"{self.base_url}/download/{view_id}.torrent"
        resp = httpx.get(url, timeout=30.0, follow_redirects=True)
        resp.raise_for_status()
        return resp.content

    @staticmethod
    def list_torrent_files(torrent_bytes: bytes) -> list[tuple[int, str, int]]:
        info = lt.torrent_info(lt.bdecode(torrent_bytes))
        files = info.files()
        return [
            (idx, files.file_path(idx), files.file_size(idx))
            for idx in range(files.num_files())
        ]

    @staticmethod
    def _match_volume_indices(
        file_entries: list[tuple[int, str, int]],
        volume_filter: str | None,
    ) -> list[int]:
        if not volume_filter:
            return [idx for idx, _, _ in file_entries]
        needle = volume_filter.lower().strip()
        # Support flexible patterns like "v01", "vol1", "vol 1", "volume 01"
        num_match = re.search(r"(\d+)", needle)
        patterns = [needle]
        if num_match:
            n = int(num_match.group(1))
            patterns.extend([f"v{n:02d}", f"v{n:03d}", f"vol{n:02d}", f"vol {n}", f"volume {n:02d}"])

        matched: list[int] = []
        for idx, path, _ in file_entries:
            low = path.lower()
            if any(p in low for p in patterns):
                matched.append(idx)
        return matched

    @staticmethod
    def extract_archives(archive_paths: list[Path], extract_root: Path) -> Path:
        """Extract downloaded CBZ/ZIP archives into chapter/volume folders."""
        extract_root.mkdir(parents=True, exist_ok=True)
        for idx, arc in enumerate(sorted(archive_paths, key=lambda p: natural_sort_key(p.name)), start=1):
            if arc.suffix.lower() not in {".cbz", ".zip"}:
                continue
            target_dir = extract_root / arc.stem
            target_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(arc, "r") as zf:
                members = sorted(
                    [m for m in zf.namelist() if not m.endswith("/") and Path(m).suffix.lower() in IMAGE_EXTS],
                    key=natural_sort_key,
                )
                for p_idx, member in enumerate(members, start=1):
                    ext = Path(member).suffix.lower()
                    out_file = target_dir / f"p{p_idx:04d}{ext}"
                    if not out_file.exists():
                        out_file.write_bytes(zf.read(member))
        return extract_root

    def download(
        self,
        source: str,
        output_dir: Path,
        volume_filter: str | None = None,
        extract: bool = True,
    ) -> Path:
        """Download torrent (optionally filtered to a single volume) and extract CBZ."""
        view_id = parse_nyaa_view_id(source)
        output_dir.mkdir(parents=True, exist_ok=True)

        torrent_path = output_dir / f"{view_id}.torrent"
        if torrent_path.exists():
            torrent_bytes = torrent_path.read_bytes()
        else:
            torrent_bytes = self.fetch_torrent_bytes(view_id)
            torrent_path.write_bytes(torrent_bytes)

        info = lt.torrent_info(lt.bdecode(torrent_bytes))
        entries = self.list_torrent_files(torrent_bytes)
        selected = self._match_volume_indices(entries, volume_filter)
        if not selected:
            raise ValueError(
                f"No files matched volume_filter={volume_filter!r} in torrent {view_id}."
            )

        selected_paths = [output_dir / entries[i][1] for i in selected]
        if all(p.exists() and p.stat().st_size == entries[i][2] for i, p in zip(selected, selected_paths)):
            if extract:
                return self.extract_archives(selected_paths, output_dir / "extracted")
            return selected_paths[0].parent

        ses = lt.session({"listen_interfaces": "0.0.0.0:6881"})
        params = lt.add_torrent_params()
        params.ti = info
        params.save_path = str(output_dir)
        handle = ses.add_torrent(params)

        priorities = [0] * info.num_files()
        for idx in selected:
            priorities[idx] = 4
        handle.prioritize_files(priorities)

        deadline = time.monotonic() + self.timeout_sec
        while time.monotonic() < deadline:
            status = handle.status()
            # Check if all selected files are complete on disk
            done = all(
                p.exists() and p.stat().st_size >= entries[i][2]
                for i, p in zip(selected, selected_paths)
            )
            if done and status.progress >= 0.999 / max(1, info.num_files()) * len(selected):
                # Also verify piece progress for selected files
                fp = handle.file_progress()
                if all(fp[i] >= entries[i][2] for i in selected):
                    break
            time.sleep(1.0)
        else:
            raise TimeoutError(
                f"Torrent download timed out after {self.timeout_sec}s for nyaa:{view_id}"
            )

        if extract:
            return self.extract_archives(selected_paths, output_dir / "extracted")
        return selected_paths[0].parent
