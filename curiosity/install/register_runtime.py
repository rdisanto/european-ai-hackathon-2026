"""Register a shared SIF and overlay independently of model checkpoints.

Run as the installation owner. No download, permission grant or GPU allocation.
The destination must not exist. Existing legacy assets are left intact.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(16 * 1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def register(destination, sif, overlay, runtime_id):
    destination, sif, overlay = map(Path, (destination, sif, overlay))
    if destination.exists() or destination.is_symlink():
        raise ValueError('Runtime destination already exists')
    if any(path.is_symlink() for path in overlay.rglob('*')):
        raise ValueError('Overlay symlinks require review before registration')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix='.runtime-register-', dir=str(destination.parent)))
    try:
        target = temporary / 'image.sif'
        try:
            os.link(str(sif), str(target))
            method = 'hardlink'
        except OSError:
            shutil.copyfile(str(sif), str(target))
            target.chmod(0o444)
            method = 'copy'
        shutil.copytree(str(overlay), str(temporary / 'python-overlay'))
        hashes = {str(path.relative_to(temporary / 'python-overlay')): digest(path)
                  for path in (temporary / 'python-overlay').rglob('*') if path.is_file()}
        obj = {'schema_version': 1, 'id': runtime_id, 'sif': 'image.sif', 'overlay': 'python-overlay',
               'sif_size': target.stat().st_size, 'sif_sha256': digest(target), 'sif_registration': method,
               'overlay_files': hashes,
               'note': 'SIF is immutable installation data. A hardlink shares the original inode; do not modify either path in place.'}
        (temporary / 'runtime.json').write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
        os.rename(str(temporary), str(destination))
    except BaseException:
        shutil.rmtree(str(temporary))
        raise
    print('REGISTERED {}: SIF {}, {} overlay files.'.format(runtime_id, method, len(hashes)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--destination', required=True)
    p.add_argument('--sif', required=True)
    p.add_argument('--overlay', required=True)
    p.add_argument('--id', required=True)
    a = p.parse_args()
    register(a.destination, a.sif, a.overlay, a.id)
