#!/usr/bin/env python3
"""Restore the exact framework from this document and a separately obtained licensed local v3 baseline."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tarfile

PUBLIC_REPOSITORY = 'https://github.com/ccrpRepo/AMP_mjlab.git'
PUBLIC_COMMIT = '6c7a2947fccc973e4af8e6d90e550400f1b6fcfc'
ARCHIVE_SHA256 = '4b6378a91139818b5d459e34a6ee73d01126c105f8ddbdca0a2ce02c2b1460b5'


def checked_members(package):
    members = package.getmembers()
    paths = set()
    for member in members:
        path = PurePosixPath(member.name)
        if (path.is_absolute() or '..' in path.parts or not path.parts
                or path.parts[0] == '.git' or member.name in paths
                or not (member.isfile() or member.isdir())):
            raise ValueError('Unsafe or duplicate archive entry: ' + member.name)
        paths.add(member.name)
    return members


def unpack(package, destination):
    for member in checked_members(package):
        path = destination.joinpath(*PurePosixPath(member.name).parts)
        if member.isdir():
            path.mkdir(parents=True, exist_ok=True)
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or any(parent.is_symlink() for parent in path.parents):
            raise ValueError('Refusing to write through a symlink: ' + str(path))
        stream = package.extractfile(member)
        if stream is None:
            raise ValueError('Unreadable member: ' + member.name)
        path.write_bytes(stream.read())
        path.chmod(member.mode & 0o777)


def run(*command):
    subprocess.run(command, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--document', required=True, type=Path)
    parser.add_argument('--destination', required=True, type=Path)
    parser.add_argument('--skip-download', action='store_true',
                        help='Compatibility option; all restorations use a local checkout')
    parser.add_argument('--base-checkout', required=True, type=Path,
                        help='Licensed local checkout containing the original v3 G1 motion inputs at the pinned commit')
    parser.add_argument('--hiking-motion', required=True, type=Path,
                        help='Original walking NPZ obtained from the Hiking authors')
    args = parser.parse_args()
    source_motion = args.hiking_motion.resolve()
    if hashlib.sha256(source_motion.read_bytes()).hexdigest() != '7cfb7c1dcaa6f2a55a13c4849be9e17b4c960ce4015c500ac0ddfb9d77f4ba5b':
        raise ValueError('The supplied Hiking reference differs from the pinned author release')
    document = args.document.read_text(encoding='utf-8')
    pattern = (r'^WMAC_PAYLOAD_PART ([0-9]{4}) ([0-9]{4})\r?\n'
               r'([A-Za-z0-9+/=\r\n]+)\r?\nWMAC_PAYLOAD_END_PART\r?$')
    parts = re.findall(pattern, document, flags=re.MULTILINE)
    if not parts:
        raise ValueError('No payload blocks found. Use the complete original Markdown.')
    totals = {int(total) for _, total, _ in parts}
    if len(totals) != 1 or len(parts) != next(iter(totals)):
        raise ValueError('Payload is truncated or has duplicate blocks')
    numbered = {int(index): content for index, _, content in parts}
    if set(numbered) != set(range(1, len(parts) + 1)):
        raise ValueError('Payload block indices are missing or duplicated')
    encoded = ''.join(''.join(numbered[i].split()) for i in range(1, len(parts)+1))
    payload = base64.b64decode(encoded, validate=True)
    actual = hashlib.sha256(payload).hexdigest()
    if actual != ARCHIVE_SHA256:
        raise ValueError('Payload SHA256 mismatch: ' + actual)
    with tarfile.open(fileobj=io.BytesIO(payload), mode='r:xz') as overlay:
        members = checked_members(overlay)
        manifest_file = overlay.extractfile('reproduce/MANIFEST.json')
        if manifest_file is None:
            raise ValueError('Missing payload manifest')
        manifest = json.load(manifest_file)
        if (manifest['public_repository'] != PUBLIC_REPOSITORY
                or manifest['public_commit'] != PUBLIC_COMMIT):
            raise ValueError('Manifest baseline differs from this restorer')
        destination = args.destination.expanduser().absolute()
        if any(path.is_symlink() for path in [destination, *destination.parents]):
            raise ValueError('Destination and its parents must not be symlinks')
        if destination.exists() and (not destination.is_dir() or any(destination.iterdir())):
            raise ValueError('Destination must be absent or an empty directory')
        destination.mkdir(parents=True, exist_ok=True)
        baseline = subprocess.check_output(['git', '-C', str(args.base_checkout),
                                            'archive', '--format=tar', PUBLIC_COMMIT])
        with tarfile.open(fileobj=io.BytesIO(baseline), mode='r:') as package:
            unpack(package, destination)
        run('git', '-C', str(destination), 'init', '--quiet')
        # The English README incorporates the separate README from the public base.
        (destination / 'README_zh.md').unlink(missing_ok=True)
        unpack(overlay, destination)
        shutil.copyfile(source_motion, destination / 'src/assets/motions/g1/hiking_official/parkour_motion_without_run_retargetted.npz')
        for name, expected in manifest['source_files'].items():
            path = destination / name
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise ValueError('Reconstructed source differs: ' + name)
        for name, expected in manifest['overlay_files'].items():
            if hashlib.sha256((destination/name).read_bytes()).hexdigest() != expected:
                raise ValueError('Reconstructed helper differs: ' + name)
        # Local outputs must be real directories, never links to another host.
        for name in ('logs', '.perceptive_cache'):
            path = destination / name
            if path.is_symlink():
                raise ValueError('Unexpected output symlink: ' + name)
            path.mkdir(exist_ok=True)
        # The restorer stays outside its own payload to avoid a circular hash.
        # Copy the exact decoder used by the user into the restored checkout.
        restorer = destination / 'scripts/restore_framework_from_doc.py'
        restorer.write_bytes(Path(__file__).resolve().read_bytes())
        restorer.chmod(0o755)
        handbook = destination / 'reproduce/FRAMEWORK_REPRODUCTION.md'
        handbook.write_bytes(args.document.read_bytes())
        print('Verified',len(manifest['source_files']),'source/asset files.')
        print('Verified payload SHA256:',actual)
        print('Restored to:',destination)
        print('Next: bash',str(destination/'reproduce/install.sh'))


if __name__ == '__main__':
    main()
