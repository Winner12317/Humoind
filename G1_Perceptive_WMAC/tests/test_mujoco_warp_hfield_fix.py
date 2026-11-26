"""Offline dependency-patch checks; writes only temporary source copies."""
from __future__ import annotations

import ast
from contextlib import redirect_stdout
import io
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from scripts import fix_mujoco_warp_hfield as fix


def pinned_sources():
    """Use the installed audited source or the installer's exact old backup."""
    try:
        location, _ = fix.installed_files()
        source = location.read_bytes()
        if fix.sha256(source) != fix.ORIGINAL_SHA256:
            backup = location.with_name(location.name+".before_hfield_"+fix.UPSTREAM_COMMIT[:8]+"_"+fix.ORIGINAL_SHA256[:12]+".bak")
            source = backup.read_bytes()
        core = location.with_name("collision_core.py").read_bytes()
    except (OSError,fix.importlib.metadata.PackageNotFoundError) as error:
        raise unittest.SkipTest("Audited MuJoCo-Warp source fixture is unavailable") from error
    if fix.sha256(source) != fix.ORIGINAL_SHA256 or fix.sha256(core) != fix.CORE_SHA256:
        raise unittest.SkipTest("Installed MuJoCo-Warp differs from the audited old fixture")
    return source,core


class HfieldFixTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.source,cls.core = pinned_sources()
        cls.patched = fix.transformed_source(cls.source,cls.core)

    def fixture(self,folder):
        path = Path(folder)/"collision_convex.py"
        path.write_bytes(self.source)
        path.with_name("collision_core.py").write_bytes(self.core)
        path.chmod(0o640)
        return path

    def test_transform_is_pinned_and_changes_only_heightfield_builder(self):
        self.assertEqual(fix.sha256(self.patched),fix.PATCHED_SHA256)
        old,new = map(ast.parse,(self.source,self.patched))
        old_other = [node for node in old.body if not isinstance(node,ast.FunctionDef)
                     or node.name != "ccd_hfield_kernel_builder"]
        new_other = [node for node in new.body if not isinstance(node,ast.FunctionDef)
                     or node.name != "ccd_hfield_kernel_builder"]
        self.assertEqual([ast.dump(node) for node in old_other],[ast.dump(node) for node in new_other])
        self.assertTrue(fix.has_upstream_fix(self.patched))
        self.assertFalse(fix.has_upstream_fix(self.source))
        self.assertEqual(self.patched.count(b"height field collision overflow"),1)

    def test_filter_matches_old_contact_writer_for_positive_zero_and_negative_gap(self):
        core_tree = ast.parse(self.core)
        writer = next(node for node in core_tree.body if isinstance(node,ast.FunctionDef) and node.name == "write_contact")
        active = next(node.value for node in writer.body if isinstance(node,ast.Assign)
                      and any(isinstance(target,ast.Name) and target.id == "active" for target in node.targets))
        core_active = compile(ast.Expression(body=active),"old_contact_active","eval")
        builder = fix._builder(self.patched.decode())
        condition = next(node.test for node in ast.walk(builder) if isinstance(node,ast.If)
                         and ast.unparse(node.test) == "ncontact == 0 or dist >= margin")
        cache_skip = compile(ast.Expression(body=condition),"backported_cache_skip","eval")
        for margin in (-.1,0.,.1):
            for gap in (-.2,0.,.2):
                for distance in (-.3,-.1,-.00001,0.,.00001,.1,.3):
                    expected = eval(core_active,{},dict(dist_in=distance,margin_in=margin,gap_in=gap))
                    actual = not eval(cache_skip,{},dict(ncontact=1,dist=distance,margin=margin,gap=gap))
                    self.assertEqual(actual,expected,(margin,gap,distance))
                    self.assertTrue(eval(cache_skip,{},dict(ncontact=0,dist=distance,margin=margin,gap=gap)))

    def test_complete_fix_recognition_does_not_accept_distance_filter_alone(self):
        incomplete = self.source.replace(b"if ncontact == 0:",b"if ncontact == 0 or dist >= margin:",1)
        self.assertFalse(fix.has_upstream_fix(incomplete))
        newer = self.patched.replace(b"if ncontact == 0 or dist >= margin:",
                                     b"if ncontact == 0 or dist >= margin + gap:",1)
        self.assertEqual(fix.installation_status(newer,b"different newer helper","9.9"),"upstream_fixed")
        no_warning = newer.replace(b"if noverflow > 0:\n      wp.printf(",
                                   b"if noverflow > 0:\n      ignore_warning(",1)
        self.assertFalse(fix.has_upstream_fix(no_warning))

    def test_original_version_and_contact_helper_are_both_required(self):
        self.assertEqual(fix.installation_status(self.source,self.core,"3.8.1"),"needs_patch")
        for source,core,version in (
                (self.source,self.core,"3.8.2"),
                (self.source+b"\n",self.core,"3.8.1"),
                (self.source,self.core+b"\n","3.8.1"),
                (self.patched,self.core,"3.8.2"),
                (self.patched,self.core+b"\n","3.8.1")):
            with self.assertRaises(fix.PatchError):
                fix.installation_status(source,core,version)
        with self.assertRaises(fix.PatchError):
            fix.transformed_source(self.source+b"\n",self.core)
        with self.assertRaises(fix.PatchError):
            fix.transformed_source(self.source,self.core+b"\n")

    def test_check_and_default_cli_are_read_only_and_diff_is_reviewable(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder)
            before = path.stat()
            report = fix.process_installation(path,"3.8.1")
            self.assertEqual(report["status"],"needs_patch")
            self.assertFalse(report["changed"])
            with patch.object(fix,"installed_files",return_value=(path,"3.8.1")):
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(fix.main([]),0)
                    self.assertEqual(fix.main(["--check","--diff"]),0)
            self.assertIn("dist >= margin",output.getvalue())
            self.assertEqual(path.read_bytes(),self.source)
            self.assertEqual(path.stat().st_mtime_ns,before.st_mtime_ns)
            self.assertEqual(sorted(p.name for p in Path(folder).iterdir()),["collision_convex.py","collision_core.py"])

    def test_apply_is_atomic_backed_up_idempotent_and_restorable_with_same_mode(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder)
            applied = fix.process_installation(path,"3.8.1","apply")
            backup = Path(applied["backup"])
            self.assertTrue(applied["changed"] and applied["restart_required"])
            self.assertEqual(path.read_bytes(),self.patched)
            self.assertEqual(backup.read_bytes(),self.source)
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o640)
            self.assertEqual(stat.S_IMODE(backup.stat().st_mode),0o640)
            self.assertFalse(fix.process_installation(path,"3.8.1","apply")["changed"])
            restored = fix.process_installation(path,"3.8.1","restore")
            self.assertTrue(restored["changed"])
            self.assertEqual(path.read_bytes(),self.source)
            self.assertFalse(fix.process_installation(path,"3.8.1","restore")["changed"])
            self.assertEqual(stat.S_IMODE(path.stat().st_mode),0o640)
            self.assertFalse(any(p.suffix == ".tmp" for p in Path(folder).iterdir()))

    def test_corrupt_backup_and_subsequent_dependency_changes_are_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder)
            report = fix.process_installation(path,"3.8.1","check")
            backup = Path(report["backup"])
            backup.write_bytes(b"unrelated contents")
            with self.assertRaises(fix.PatchError):
                fix.process_installation(path,"3.8.1","apply")
            self.assertEqual(path.read_bytes(),self.source)
            backup.unlink()
            fix.process_installation(path,"3.8.1","apply")
            backup.write_bytes(b"corrupt backup")
            with self.assertRaises(fix.PatchError):
                fix.process_installation(path,"3.8.1","restore")
            self.assertEqual(path.read_bytes(),self.patched)
            backup.write_bytes(self.source)
            modified = self.patched+b"\n# dependency changed afterwards\n"
            path.write_bytes(modified)
            with self.assertRaises(fix.PatchError):
                fix.process_installation(path,"3.8.1","restore")
            self.assertEqual(path.read_bytes(),modified)

    def test_unsupported_core_and_symlink_are_never_replaced(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.fixture(folder)
            path.with_name("collision_core.py").write_bytes(self.core+b"\n")
            with self.assertRaises(fix.PatchError):
                fix.process_installation(path,"3.8.1","apply")
            self.assertEqual(path.read_bytes(),self.source)
            link = path.with_name("linked.py")
            link.symlink_to(path)
            with self.assertRaises(fix.PatchError):
                fix.process_installation(link,"3.8.1","apply")
            self.assertTrue(link.is_symlink())


if __name__ == "__main__":
    unittest.main()
