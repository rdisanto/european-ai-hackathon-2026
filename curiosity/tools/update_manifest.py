"""Regenerate serve/RELEASE_MANIFEST.json after editing files in serve/.

inference.py refuses to prepare a run when a file differs from the manifest,
so every change to serve/ needs this. The manifest also decides which files
are copied into a private run directory. Python 3.6+, like the framework.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path

SERVE = Path(__file__).resolve().parents[1] / 'serve'


def build(release):
    files = {}
    for path in sorted(SERVE.rglob('*')):
        relative = path.relative_to(SERVE)
        if not path.is_file() or path.name == 'RELEASE_MANIFEST.json' or '__pycache__' in relative.parts:
            continue
        files[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return {'release': release, 'files': files}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--release', help='Release label recorded in each run and endpoint '
                        '(default: keep the current label, or inference/<today> if none)')
    parser.add_argument('--check', action='store_true', help='Only report whether the manifest is current')
    args = parser.parse_args()
    path = SERVE / 'RELEASE_MANIFEST.json'
    current = json.loads(path.read_text()) if path.exists() else {}
    release = args.release or current.get('release') or 'inference/' + datetime.date.today().isoformat()
    manifest = build(release)
    if args.check:
        if manifest['files'] != current.get('files'):
            raise SystemExit('serve/RELEASE_MANIFEST.json is stale: run python3 tools/update_manifest.py')
        print('Manifest current: {} files, release {}.'.format(len(manifest['files']), current['release']))
        return
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print('Wrote {} with {} files, release {}.'.format(path, len(manifest['files']), release))


if __name__ == '__main__':
    main()
