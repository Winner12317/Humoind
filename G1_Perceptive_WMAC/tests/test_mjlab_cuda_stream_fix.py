import importlib.metadata
from pathlib import Path
import runpy
import stat
import tempfile
import unittest

import torch
import warp as wp

from reproduce import fix_mjlab_cuda_stream as fix


class TestMjlabCudaStreamFix(unittest.TestCase):
    def setUp(self):
        # Use the wheel's source, on systems where the installer may already
        # have applied this patch. All writes in these tests go to a temp dir.
        source = Path(importlib.metadata.distribution("mjlab").locate_file(fix.MODULE)).read_bytes()
        self.original = source.replace(fix.NEW_RETURN, fix.OLD_RETURN, 1)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.target = self.root / fix.MODULE
        self.target.parent.mkdir(parents=True)
        self.target.write_bytes(self.original)
        self.info = self.root / "mjlab-1.2.0.dist-info"
        self.info.mkdir()
        (self.info / "METADATA").write_text("Metadata-Version: 2.1\nName: mjlab\nVersion: 1.2.0\n")
        (self.info / "RECORD").write_text(f"{fix.MODULE},,\n")
        self.distribution = importlib.metadata.PathDistribution(self.info)

    def test_preserves_other_source_and_permissions_and_applies_once(self):
        original = self.target.stat()
        self.assertEqual(fix.check_installation(self.distribution), "patch available")
        self.assertEqual(self.target.stat().st_mtime_ns, original.st_mtime_ns)
        self.target.chmod(0o640)
        self.assertEqual(fix.check_installation(self.distribution, apply=True), "patched")
        self.assertEqual(self.target.read_bytes(), self.original.replace(fix.OLD_RETURN, fix.NEW_RETURN, 1))
        self.assertEqual(stat.S_IMODE(self.target.stat().st_mode), 0o640)
        inode = self.target.stat().st_ino
        self.assertEqual(fix.check_installation(self.distribution, apply=True), "already patched")
        self.assertEqual(self.target.stat().st_ino, inode)

    def test_rejects_other_source_or_version(self):
        changed = self.original + b"# local edit\n"
        self.target.write_bytes(changed)
        with self.assertRaisesRegex(ValueError, "Unrecognized"):
            fix.check_installation(self.distribution, apply=True)
        self.assertEqual(self.target.read_bytes(), changed)
        (self.info / "METADATA").write_text("Name: mjlab\nVersion: 2.0.0\n")
        with self.assertRaisesRegex(ValueError, "Expected mjlab"):
            fix.check_installation(self.distribution, apply=True)

    def test_rejects_file_not_owned_by_distribution(self):
        (self.info / "RECORD").write_text("another.py,,\n")
        with self.assertRaisesRegex(ValueError, "does not own"):
            fix.check_installation(self.distribution, apply=True)
        self.assertEqual(self.target.read_bytes(), self.original)

    def check_stream_owner(self, ordinal):
        fix.check_installation(self.distribution, apply=True)
        torch_array = runpy.run_path(str(self.target))["TorchArray"]
        wp.init()
        with torch.cuda.device(0):
            array = wp.zeros(2, dtype=wp.float32, device=f"cuda:{ordinal}")
            wrapped = torch_array(array)
            self.assertEqual(wrapped._torch_stream.device, torch.device("cuda", ordinal))
            self.assertEqual(wrapped._torch_stream.cuda_stream, wp.get_stream(array.device).cuda_stream)
            self.assertEqual(torch.cuda.current_device(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_stream_owner_on_primary_device(self):
        self.check_stream_owner(0)

    @unittest.skipUnless(torch.cuda.device_count() >= 2, "requires two CUDA devices")
    def test_stream_owner_on_other_device(self):
        self.check_stream_owner(1)
