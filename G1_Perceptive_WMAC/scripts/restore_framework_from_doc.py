#!/usr/bin/env python3
"""Restore the exact framework from this document and a public GitHub baseline."""
from __future__ import annotations
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
import tarfile

PUBLIC_REPOSITORY = 'https://github.com/ccrpRepo/AMP_mjlab.git'
PUBLIC_COMMIT = '6c7a2947fccc973e4af8e6d90e550400f1b6fcfc'
ARCHIVE_SHA256 = 'ecd718333366994ea664539efcc683941364af02d42b1e8a86c7f8954e9ceacc'


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
                        help='Use git archive from --base-checkout for local verification only')
    parser.add_argument('--base-checkout', type=Path,
                        help='Existing checkout containing the pinned public baseline commit')
    args = parser.parse_args()
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
        if args.skip_download:
            if args.base_checkout is None:
                raise ValueError('--skip-download requires --base-checkout')
            # git archive reads committed objects; dirty files are never copied.
            baseline = subprocess.check_output(['git', '-C', str(args.base_checkout),
                                                'archive', '--format=tar', PUBLIC_COMMIT])
            with tarfile.open(fileobj=io.BytesIO(baseline), mode='r:') as package:
                unpack(package, destination)
            run('git','-C',str(destination),'init','--quiet')
        else:
            run('git','-C',str(destination),'init','--quiet')
            run('git','-C',str(destination),'remote','add','origin',PUBLIC_REPOSITORY)
            run('git','-C',str(destination),'fetch','--depth=1','origin',PUBLIC_COMMIT)
            run('git','-C',str(destination),'checkout','--quiet','--detach','FETCH_HEAD')
        # The English README incorporates the separate README from the public base.
        (destination / 'README_zh.md').unlink(missing_ok=True)
        unpack(overlay, destination)
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
