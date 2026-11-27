"""Pinned Hiking data can hydrate an empty cache without external requests."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np

from src.wmac_perceptive import locomotion_data as source


MOTION = "parkour_motion_without_run_retargetted.npz"
SELECTION = "parkour_motion_without_run.yaml"
README = "README.md"
ARCHIVE = "hiking-in-the-wild_Data_Model.zip"


class HikingBundledSourceTest(unittest.TestCase):
    @contextmanager
    def fixture(self):
        contents = {
            MOTION: b"pinned motion fixture",
            SELECTION: b"selected_files:\n- parkour_motion_without_run_retargetted.npz\n",
            README: b"unmodified author fixture readme\n",
        }
        digests = {name: hashlib.sha256(value).hexdigest() for name, value in contents.items()}
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bundle, cache = root/"bundle", root/"cache"
            bundle.mkdir()
            for name, value in contents.items():
                (bundle/name).write_bytes(value)
            with patch.object(source, "HIKING_BUNDLED_SOURCE", bundle), \
                    patch.object(source, "HIKING_SOURCE_SHA256", digests):
                yield root, bundle, cache, contents

    def assert_source(self, directory, expected):
        for name, content in expected.items():
            self.assertEqual((directory/name).read_bytes(), content)

    def test_empty_cache_is_populated_from_verified_bundle_without_network(self):
        with self.fixture() as (_, bundle, cache, contents), \
                patch("urllib.request.urlopen", side_effect=AssertionError("external request")) as request:
            result = source.ensure_hiking_source(cache)
            self.assertEqual(result, cache/MOTION)
            self.assert_source(cache, contents)
            self.assert_source(bundle, contents)
            self.assertFalse((cache/ARCHIVE).exists())
            request.assert_not_called()

    def test_complete_verified_cache_does_not_need_bundle_or_archive(self):
        with self.fixture() as (root, _, cache, contents):
            cache.mkdir()
            for name, value in contents.items():
                (cache/name).write_bytes(value)
            with patch.object(source, "HIKING_BUNDLED_SOURCE", root/"absent"), \
                    patch("urllib.request.urlopen", side_effect=AssertionError("external request")) as request:
                self.assertEqual(source.ensure_hiking_source(cache), cache/MOTION)
                self.assert_source(cache, contents)
                request.assert_not_called()

    def test_partial_cache_is_filled_without_replacing_existing_source(self):
        with self.fixture() as (_, _, cache, contents), \
                patch("urllib.request.urlopen", side_effect=AssertionError("external request")) as request:
            cache.mkdir()
            (cache/MOTION).write_bytes(contents[MOTION])
            self.assertEqual(source.ensure_hiking_source(cache), cache/MOTION)
            self.assert_source(cache, contents)
            request.assert_not_called()

    def test_corrupt_bundle_is_rejected_before_copying_or_requesting(self):
        with self.fixture() as (_, bundle, cache, _), \
                patch("urllib.request.urlopen", side_effect=AssertionError("external request")) as request:
            (bundle/README).write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "SHA256|modified"):
                source.ensure_hiking_source(cache)
            self.assertFalse((cache/MOTION).exists())
            request.assert_not_called()

    def test_corrupt_existing_cache_is_never_silently_overwritten(self):
        with self.fixture() as (_, _, cache, contents), \
                patch("urllib.request.urlopen", side_effect=AssertionError("external request")) as request:
            cache.mkdir()
            for name, value in contents.items():
                (cache/name).write_bytes(value)
            (cache/MOTION).write_bytes(b"changed cached pose")
            with self.assertRaisesRegex(ValueError, "SHA256|modified"):
                source.ensure_hiking_source(cache)
            self.assertEqual((cache/MOTION).read_bytes(), b"changed cached pose")
            request.assert_not_called()

    def test_existing_invalid_archive_cannot_be_hidden_by_valid_bundle(self):
        with self.fixture() as (_, _, cache, _), \
                patch("urllib.request.urlopen", side_effect=AssertionError("external request")) as request:
            cache.mkdir()
            (cache/ARCHIVE).write_bytes(b"changed archive")
            with self.assertRaisesRegex(ValueError, "SHA256|modified"):
                source.ensure_hiking_source(cache)
            request.assert_not_called()

    def test_absent_bundle_preserves_pinned_download_fallback(self):
        with self.fixture() as (root, _, cache, contents):
            buffer = io.BytesIO()
            members = {
                source.HIKING_MOTION_MEMBER: contents[MOTION],
                "data&model/parkour_motion_reference/"+SELECTION: contents[SELECTION],
                "data&model/"+README: contents[README],
                "../../unwanted.py": b"not extracted",
            }
            with zipfile.ZipFile(buffer, "w") as package:
                for name, value in members.items():
                    package.writestr(name, value)
            payload = buffer.getvalue()
            with patch.object(source, "HIKING_BUNDLED_SOURCE", root/"absent"), \
                    patch.object(source, "HIKING_ARCHIVE_SHA256", hashlib.sha256(payload).hexdigest()), \
                    patch("urllib.request.urlopen", return_value=io.BytesIO(payload)) as request:
                self.assertEqual(source.ensure_hiking_source(cache), cache/MOTION)
                self.assert_source(cache, contents)
                request.assert_called_once()
                self.assertFalse((root/"unwanted.py").exists())

    def test_source_verifier_rejects_missing_and_changed_required_files(self):
        with self.fixture() as (_, bundle, _, _):
            self.assertEqual(source.verify_hiking_source_files(bundle), bundle/MOTION)
            (bundle/SELECTION).unlink()
            with self.assertRaises(FileNotFoundError):
                source.verify_hiking_source_files(bundle)

    def test_installed_reference_has_pinned_bytes_and_full_official_motion(self):
        bundle = source.DEFAULT_HIKING_SOURCE
        if not (bundle/MOTION).is_file():
            self.skipTest("Install the author-provided Hiking reference to run the full asset check")
        self.assertEqual(source.verify_hiking_source_files(bundle), bundle/MOTION)
        self.assertEqual(set(source.HIKING_SOURCE_SHA256), {MOTION, SELECTION, README})
        self.assertEqual(hashlib.sha256((bundle/MOTION).read_bytes()).hexdigest(),
                         source.HIKING_MOTION_SHA256)
        with np.load(bundle/MOTION, allow_pickle=False) as data:
            self.assertEqual(data["joint_pos"].shape, (18982, 29))
            self.assertEqual(data["base_pos_w"].shape, (18982, 3))
            self.assertEqual(data["base_quat_w"].shape, (18982, 4))
            self.assertEqual(float(data["framerate"]), 50.)
            for key in ("joint_pos", "base_pos_w", "base_quat_w"):
                self.assertTrue(np.isfinite(data[key]).all())


if __name__ == "__main__":
    unittest.main()
