"""Input loader supporting image folders, multi-chapter folders, and CBZ/ZIP archives."""

from __future__ import annotations

import re
import zipfile
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def natural_sort_key(text: str) -> list[int | str]:
    """Sort strings containing numbers naturally (e.g. page_2 before page_10)."""
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


def _extract_cbz(cbz_path: Path, extract_dir: Path) -> list[Path]:
    extract_dir.mkdir(parents=True, exist_ok=True)
    images: list[Path] = []
    with zipfile.ZipFile(cbz_path, "r") as zf:
        members = sorted(
            [m for m in zf.namelist() if not m.endswith("/") and Path(m).suffix.lower() in IMAGE_EXTS],
            key=natural_sort_key,
        )
        for idx, member in enumerate(members, start=1):
            ext = Path(member).suffix.lower()
            out_path = extract_dir / f"p{idx:04d}{ext}"
            if not out_path.exists():
                out_path.write_bytes(zf.read(member))
            images.append(out_path)
    return images


def collect_chapters(
    input_path: Path,
    work_dir: Path,
    max_pages_per_chapter: int | None = None,
) -> list[tuple[str, str, list[Path]]]:
    """Discover chapters and sorted image paths from a directory or CBZ file.

    Returns list of (chapter_id, chapter_title, image_paths).
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    chapters: list[tuple[str, str, list[Path]]] = []

    if input_path.is_file() and input_path.suffix.lower() in {".cbz", ".zip"}:
        imgs = _extract_cbz(input_path, work_dir / "extracted" / "ch01")
        if max_pages_per_chapter:
            imgs = imgs[:max_pages_per_chapter]
        if imgs:
            chapters.append(("ch01", input_path.stem, imgs))
        return chapters

    if input_path.is_dir():
        cbz_files = sorted(
            [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in {".cbz", ".zip"}],
            key=lambda p: natural_sort_key(p.name),
        )
        if cbz_files:
            for idx, cbz in enumerate(cbz_files, start=1):
                cid = f"ch{idx:02d}"
                imgs = _extract_cbz(cbz, work_dir / "extracted" / cid)
                if max_pages_per_chapter:
                    imgs = imgs[:max_pages_per_chapter]
                if imgs:
                    chapters.append((cid, cbz.stem, imgs))
            return chapters

        subdirs = sorted(
            [d for d in input_path.iterdir() if d.is_dir() and not d.name.startswith(".")],
            key=lambda d: natural_sort_key(d.name),
        )
        if subdirs:
            for idx, sub in enumerate(subdirs, start=1):
                imgs = sorted(
                    [p for p in sub.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                    key=lambda p: natural_sort_key(p.name),
                )
                if max_pages_per_chapter:
                    imgs = imgs[:max_pages_per_chapter]
                if imgs:
                    chapters.append((sub.name, sub.name, imgs))
            if chapters:
                return chapters

        imgs = sorted(
            [p for p in input_path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
            key=lambda p: natural_sort_key(p.name),
        )
        if max_pages_per_chapter:
            imgs = imgs[:max_pages_per_chapter]
        if imgs:
            chapters.append(("ch01", input_path.name, imgs))
            return chapters

    raise ValueError(f"No supported manga images or CBZ archives found in {input_path}")
