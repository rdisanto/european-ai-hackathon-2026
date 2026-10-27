"""Copy hash-pinned runtime overlay files without replacing existing customizations.

Runs on the login/CPU node, before registration. The copied sitecustomize hook
itself runs only inside the vLLM container, whose Python supports AST end_lineno.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys

from register_runtime import digest


def reuse_runtime(base, serve, runtime_id, staging):
    """Reuse immutable installed assets; otherwise let the normal installer pull them."""
    base, serve, staging = Path(base), Path(serve), Path(staging)
    catalog = json.loads((serve / 'catalog.json').read_text())['runtimes']
    parent_id = catalog[runtime_id].get('reuse_from')
    if not parent_id or (staging / 'image.sif').exists():
        return False
    if catalog[parent_id]['image'] != catalog[runtime_id]['image']:
        raise ValueError('Reused runtime must have the same pinned image')
    if catalog[parent_id]['overlay_packages'] != catalog[runtime_id]['overlay_packages']:
        raise ValueError('Reused runtime must have the same pinned overlay packages')
    site = json.loads((base / 'inference-site.json').read_text())
    parent = base / site.get('runtime_paths', {}).get(parent_id, 'runtimes/' + parent_id)
    if not (parent / 'runtime.json').exists():
        return False
    record = json.loads((parent / 'runtime.json').read_text())
    sif, overlay = parent / record['sif'], parent / record['overlay']
    if record['id'] != parent_id or digest(sif) != record['sif_sha256']:
        raise ValueError('Reused runtime registration/hash mismatch')
    if any(p.is_symlink() for p in overlay.rglob('*')):
        raise ValueError('Reused overlay must not contain symlinks')
    for name, expected in record['overlay_files'].items():
        if digest(overlay / name) != expected:
            raise ValueError('Reused overlay hash mismatch: ' + name)
    # A private staging copy prevents the patch from modifying the shared base.
    destination = staging / 'python-overlay'
    if any(destination.iterdir()):
        raise ValueError('Reuse needs an empty staging overlay; inspect the interrupted install')
    for path in overlay.iterdir():
        target = destination / path.name
        if path.is_dir():
            shutil.copytree(str(path), str(target))
        else:
            shutil.copyfile(str(path), str(target))
    try:
        os.link(str(sif), str(staging / 'image.sif'))
    except OSError:
        shutil.copyfile(str(sif), str(staging / 'image.sif'))
    return True


def copy_overlay(serve, runtime_id, destination):
    serve, destination = Path(serve), Path(destination)
    runtime = json.loads((serve / 'catalog.json').read_text())['runtimes'][runtime_id]
    for item in runtime.get('overlay_files', []):
        source_name, target_name = Path(item['source']), Path(item['target'])
        if (any(p.is_absolute() or '..' in p.parts for p in (source_name, target_name))
                or len(target_name.parts) != 1):
            raise ValueError('Unsafe runtime overlay path')
        source, target = serve / source_name, destination / target_name
        if source.is_symlink() or serve.resolve() not in source.resolve().parents:
            raise ValueError('Overlay source must be a release file')
        if hashlib.sha256(source.read_bytes()).hexdigest() != item['sha256']:
            raise ValueError('Runtime overlay source hash mismatch: ' + item['source'])
        if target.is_symlink() or (target.exists() and target.read_bytes() != source.read_bytes()):
            raise ValueError('Refusing to replace runtime overlay file: ' + item['target'])
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(str(source), str(target))


if __name__ == '__main__':
    if sys.argv[1] == 'reuse':
        print('yes' if reuse_runtime(*sys.argv[2:]) else 'no')
    else:
        copy_overlay(*sys.argv[2:])
