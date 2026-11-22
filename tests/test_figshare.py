"""Offline acquisition tests: no network access or real dataset required."""

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

import requests

from wrist_fracture.data import figshare


def response(content=b"abc", status=200):
    result = MagicMock()
    result.__enter__.return_value = result
    result.status_code = status
    result.iter_content.return_value = iter([content])
    return result


class AcquisitionTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.item = {
            "id": 1, "name": "dataset.csv", "size": 3,
            "download_url": "https://example.org/file",
            "computed_md5": hashlib.md5(b"abc").hexdigest(),
        }
        self.article = {"id": 14825193, "title": "test", "version": 2,
                        "files": [self.item]}

    @patch.object(figshare.requests, "get")
    def test_fetch_actual_response_and_version_endpoint(self, get):
        reply = response()
        reply.json.return_value = self.article
        get.return_value = reply
        self.assertEqual(figshare.fetch_article(version=2), self.article)
        self.assertTrue(get.call_args.args[0].endswith("/14825193/versions/2"))
        reply.raise_for_status.assert_called_once()

    @patch.object(figshare.requests, "get")
    def test_invalid_metadata_rejected(self, get):
        reply = response()
        reply.json.return_value = {"id": 14825193}
        get.return_value = reply
        with self.assertRaises(ValueError):
            figshare.fetch_article()

    def test_selection_is_explicit_and_exact(self):
        self.assertEqual(figshare.select_files(self.article, ["dataset.csv"]), [self.item])
        for names in ([], ["missing.zip"]):
            with self.assertRaises(ValueError):
                figshare.select_files(self.article, names)

    @patch.object(figshare.requests, "get")
    def test_stream_integrity_provenance_and_reuse(self, get):
        get.return_value = response()
        path = figshare.download_file(self.article, self.item, self.output)
        self.assertEqual(path.read_bytes(), b"abc")
        self.assertTrue(get.call_args.kwargs["stream"])
        figshare.download_file(self.article, self.item, self.output)
        self.assertEqual(get.call_count, 1)
        records = [json.loads(p.read_text()) for p in (self.output / ".provenance").glob("*.json")]
        self.assertEqual(len(records), 2)
        self.assertTrue(all(r["verification"]["source_md5_verified"] for r in records))
        self.assertEqual(records[0]["article"], self.article)

    @patch.object(figshare.requests, "get")
    def test_existing_mismatch_is_not_overwritten(self, get):
        (self.output / "dataset.csv").write_bytes(b"bad")
        with self.assertRaisesRegex(ValueError, "MD5 mismatch"):
            figshare.download_file(self.article, self.item, self.output)
        get.assert_not_called()
        self.assertEqual((self.output / "dataset.csv").read_bytes(), b"bad")

    @patch.object(figshare.time, "sleep")
    @patch.object(figshare.requests, "get")
    def test_interruption_restarts_cleanly(self, get, sleep):
        broken = response()
        def interrupted(*args, **kwargs):
            yield b"a"
            raise requests.ConnectionError("interrupted")
        broken.iter_content.side_effect = interrupted
        get.side_effect = [broken, response()]
        path = figshare.download_file(self.article, self.item, self.output)
        self.assertEqual(path.read_bytes(), b"abc")
        self.assertEqual(get.call_count, 2)
        self.assertFalse((self.output / ".dataset.csv.part").exists())

    @patch.object(figshare.requests, "get")
    def test_corrupt_or_truncated_download_not_published(self, get):
        for content in (b"ab", b"bad", b"abcd"):
            with self.subTest(content=content):
                get.return_value = response(content)
                with self.assertRaises(ValueError):
                    figshare.download_file(self.article, self.item, self.output, attempts=1)
                self.assertFalse((self.output / "dataset.csv").exists())
                self.assertFalse((self.output / ".dataset.csv.part").exists())
                self.assertFalse((self.output / ".dataset.csv.lock").exists())

    @patch.object(figshare.requests, "get")
    def test_permanent_http_error_not_retried(self, get):
        reply = response(status=404)
        reply.raise_for_status.side_effect = requests.HTTPError(response=reply)
        get.return_value = reply
        with self.assertRaises(requests.HTTPError):
            figshare.download_file(self.article, self.item, self.output)
        self.assertEqual(get.call_count, 1)

    @patch.object(figshare.requests, "get")
    def test_unsafe_names_rejected_before_network(self, get):
        for name in ("../escape", "C:\\escape", "CON", "file:stream", "..", "file."):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.item["name"] = name
                figshare.download_file(self.article, self.item, self.output)
        get.assert_not_called()

    @patch.object(figshare.requests, "get")
    def test_absent_source_checksum_is_recorded(self, get):
        self.item.pop("computed_md5")
        get.return_value = response()
        figshare.download_file(self.article, self.item, self.output)
        record = json.loads(next((self.output / ".provenance").glob("*.json")).read_text())
        self.assertFalse(record["verification"]["source_md5_verified"])

    @patch.object(figshare.requests, "get")
    def test_lock_prevents_competing_download(self, get):
        (self.output / ".dataset.csv.lock").touch()
        with self.assertRaises(FileExistsError):
            figshare.download_file(self.article, self.item, self.output)
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
