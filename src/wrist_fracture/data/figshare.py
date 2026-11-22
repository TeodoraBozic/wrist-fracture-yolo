"""Inspect Figshare metadata and explicitly download selected original files.

Downloads restart from zero after interruption. Completed files are never
overwritten; a mismatching existing file requires manual investigation.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import re
import time
from typing import Any

import requests

LOGGER = logging.getLogger(__name__)
ARTICLE_ID = 14825193
API_ROOT = "https://api.figshare.com/v2/articles"
CHUNK_SIZE = 1024 * 1024


def utc_now() -> str:
    """Return an unambiguous UTC acquisition timestamp."""
    return datetime.now(timezone.utc).isoformat()


def _retryable(error: requests.RequestException) -> bool:
    response = error.response
    return response is None or response.status_code in {408, 429} or response.status_code >= 500

#Komunicira sa Figshare API-jem i čita informacije o datasetu
def fetch_article(
    article_id: int = ARTICLE_ID,
    *,
    version: int | None = None,
    attempts: int = 3,
    timeout: float = 60,
) -> dict[str, Any]:
    """Fetch actual public article metadata, optionally pinned to a version."""
    if article_id < 1 or (version is not None and version < 1):
        raise ValueError("Article ID and version must be positive")
    if attempts < 1 or timeout <= 0:
        raise ValueError("Attempts and timeout must be positive")
    url = f"{API_ROOT}/{article_id}"
    if version is not None:
        url += f"/versions/{version}"
    for attempt in range(attempts):
        try:
            with requests.get(url, timeout=(10, timeout)) as response:
                response.raise_for_status()
                article = response.json()
            if not isinstance(article, dict) or not isinstance(article.get("files"), list):
                raise ValueError("Figshare response must be an article with a files list")
            if article.get("id") != article_id:
                raise ValueError("Figshare returned an unexpected article ID")
            return article
        except requests.RequestException as error:
            if attempt + 1 == attempts or not _retryable(error):
                raise
            LOGGER.warning("Metadata request failed; retrying: %s", error)
            time.sleep(2 ** attempt)
    raise AssertionError("Unreachable")

#Odabira fajlove koje želiš da preuzmeš
def select_files(article: dict[str, Any], names: list[str]) -> list[dict[str, Any]]:
    """Resolve exact filenames against the API; never select all implicitly."""
    if not names:
        raise ValueError("Select at least one exact filename")
    selected = []
    for name in dict.fromkeys(names):
        matches = [item for item in article["files"] if item.get("name") == name]
        if len(matches) != 1:
            raise ValueError(f"Filename not uniquely available in article: {name!r}")
        selected.append(matches[0])
    return selected

#Pronalazi kontrolni hash koji je objavio izvor
def file_md5(item: dict[str, Any]) -> str | None:
    """Prefer Figshare's computed MD5, falling back to the supplied checksum."""
    for key in ("computed_md5", "supplied_md5"):
        value = item.get(key)
        if isinstance(value, str) and re.fullmatch(r"[0-9a-fA-F]{32}", value):
            return value.lower()
    return None


def verify_file(path: Path, item: dict[str, Any]) -> dict[str, Any]:
    """Verify exact source size and MD5; record SHA-256 for local provenance."""
    if path.stat().st_size != item["size"]:
        raise ValueError(f"Size mismatch for {path.name}")
    md5 = hashlib.md5(usedforsecurity=False)
    sha256 = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(CHUNK_SIZE):
            md5.update(chunk)
            sha256.update(chunk)
    expected = file_md5(item)
    if expected and md5.hexdigest() != expected:
        raise ValueError(f"MD5 mismatch for {path.name}")
    if expected is None:
        LOGGER.warning("No source MD5 for %s; only size can be source-verified", path.name)
    return {"md5": md5.hexdigest(), "sha256": sha256.hexdigest(),
            "source_md5_verified": expected is not None, "bytes": path.stat().st_size}


def _safe_name(name: str) -> None:
    if (not name or name in {".", ".."} or name.endswith((".", " "))
            or re.search(r'[<>:"/\\|?*\x00-\x1f]', name)
            or name.split(".")[0].upper() in {
                "CON", "PRN", "AUX", "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }):
        raise ValueError(f"Unsafe source filename: {name!r}")


def _write_record(directory: Path, record: dict[str, Any]) -> None:
    """Write one immutable event per completed or verified download."""
    from uuid import uuid4

    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{uuid4().hex}.json"
    with target.open("x", encoding="utf-8") as stream:
        json.dump(record, stream, indent=2, ensure_ascii=False)
        stream.write("\n")


def download_file(
    article: dict[str, Any],
    item: dict[str, Any],
    output_dir: Path,
    *,
    attempts: int = 3,
    timeout: float = 60,
) -> Path:
    """Stream one API-listed file, verify it, then atomically publish it.

    A per-file exclusive lock prevents competing writers. Partial files are
    removed on failure; retrying restarts the transfer. No archives are extracted.
    """
    if attempts < 1 or timeout <= 0:
        raise ValueError("Attempts and timeout must be positive")
    if item not in article["files"] or article.get("download_disabled"):
        raise ValueError("File unavailable for download in this article")
    name = item["name"]
    _safe_name(name)
    if item.get("is_link_only") or not item.get("download_url", "").startswith("https://"):
        raise ValueError("Only downloadable HTTPS files are supported")
    if not isinstance(item.get("size"), int) or item["size"] < 0:
        raise ValueError("Source file size must be a nonnegative integer")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    destination = output_dir / name
    partial = output_dir / f".{name}.part"
    lock = output_dir / f".{name}.lock"
    # Exclusive creation also refuses a symlink at the lock location.
    with lock.open("x"):
        pass
    try:
        if destination.is_symlink() or partial.is_symlink():
            raise ValueError("Refusing symlink destination or partial file")
        if destination.exists():
            verification = verify_file(destination, item)
            action = "verified_existing"
        else:
            for attempt in range(attempts):
                try:
                    with requests.get(item["download_url"], stream=True,
                                      timeout=(10, timeout)) as response:
                        response.raise_for_status()
                        if response.status_code != 200:
                            raise ValueError("Expected a complete HTTP 200 response")
                        with partial.open("wb") as stream:
                            count = 0
                            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                                count += len(chunk)
                                if count > item["size"]:
                                    raise ValueError("Download exceeds source size")
                                stream.write(chunk)
                    verification = verify_file(partial, item)
                    if destination.exists():
                        raise FileExistsError(destination)
                    partial.rename(destination)
                    break
                except (requests.RequestException, ValueError) as error:
                    partial.unlink(missing_ok=True)
                    permanent_http_error = (
                        isinstance(error, requests.RequestException) and not _retryable(error)
                    )
                    if attempt + 1 == attempts or permanent_http_error:
                        raise
                    LOGGER.warning("Download failed; restarting %s: %s", name, error)
                    time.sleep(2 ** attempt)
            action = "downloaded"
        _write_record(output_dir / ".provenance", {
            "schema_version": 1, "recorded_at_utc": utc_now(), "action": action,
            "article": article, "file": item, "verification": verification,
            "local_path": str(destination.resolve()),
        })
        LOGGER.info("%s: %s", action, destination)
        return destination
    finally:
        # A symlink encountered above is left untouched for investigation.
        if not partial.is_symlink():
            partial.unlink(missing_ok=True)
        lock.unlink(missing_ok=True)


def main() -> int:
    """Run the metadata-only inspect or explicit file download command."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-id", type=int, default=ARTICLE_ID)
    parser.add_argument("--version", type=int, help="Pin an official article version")
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60, help="Read timeout in seconds")
    commands = parser.add_subparsers(dest="command", required=True)
    inspect_parser = commands.add_parser("inspect", help="Fetch metadata only")
    inspect_parser.add_argument("--save", type=Path, help="Save timestamped metadata snapshot")
    download_parser = commands.add_parser("download", help="Download explicitly selected files")
    download_parser.add_argument("--file", action="append", required=True, dest="names")
    download_parser.add_argument("--output-dir", type=Path, default=Path("data/raw/grazpedwri-dx"))
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        article = fetch_article(args.article_id, version=args.version,
                                attempts=args.attempts, timeout=args.timeout)
        if args.command == "inspect":
            print(f"{article.get('title')} | version: {article.get('version', 'unavailable')}")
            total = sum(item["size"] for item in article["files"])
            for item in article["files"]:
                print(f"{item['name']} | {item['size']:,} bytes | {item.get('download_url')}")
            print(f"Total: {total:,} bytes ({total / 1024 ** 3:.3f} GiB)")
            if args.save:
                args.save.parent.mkdir(parents=True, exist_ok=True)
                with args.save.open("x", encoding="utf-8") as stream:
                    snapshot = {"retrieved_at_utc": utc_now(), "article": article}
                    json.dump(snapshot, stream, indent=2)
        else:
            for item in select_files(article, args.names):
                download_file(article, item, args.output_dir,
                              attempts=args.attempts, timeout=args.timeout)
        return 0
    except (requests.RequestException, OSError, ValueError, KeyError, TypeError) as error:
        LOGGER.error("Acquisition failed: %s", error)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
