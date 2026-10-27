"""Shared endpoint registry in $BASE/<site endpoints dir>: discovery and one-server-per-name locks.

Records hold addresses only, never keys: everyone the team folder is shared with can read them.
A job claims its served model name when it starts, publishes the record after its API checks pass,
and withdraws both when it stops. A claim held by a job that Slurm no longer knows is taken over.
"""
import getpass
import json
import os
from pathlib import Path
import shutil
import subprocess

NAME_CHARS = frozenset('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-')


def check_name(name):
    if not name or name[0] in '._-' or not set(name) <= NAME_CHARS or len(name) > 160:
        raise ValueError('Invalid endpoint name: ' + name)
    return name


def directory(cfg):
    """The shared registry directory, or None when the site does not publish endpoints."""
    relative = cfg['site'].get('endpoints')
    if not relative:
        return None
    path = Path(cfg['base']) / relative
    # Created by `team.py init`; teammates get access through the ACLs `team.py share` sets.
    path.mkdir(parents=True, exist_ok=True)
    return path


def job_state(job):
    """'active', 'gone' or 'unknown'. Only 'gone' allows taking over a claim."""
    try:
        result = subprocess.run(['squeue', '-h', '-j', str(job), '-o', '%T'],
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return 'unknown'
    if result.returncode == 0:
        return 'active' if result.stdout.strip() else 'gone'
    return 'gone' if b'Invalid job id' in result.stderr else 'unknown'


def _owner(lock):
    try:
        return (lock / 'owner').read_text().strip()
    except OSError:
        return ''


def claim(root, name, job):
    """Reserve `name` for `job`. Raises ValueError if another live job holds it."""
    check_name(name)
    lock = root / ('.' + name + '.lock')
    for attempt in (1, 2):
        try:
            lock.mkdir()
        except FileExistsError:
            holder = _owner(lock)
            if holder == str(job):
                return
            if attempt == 1 and holder and job_state(holder) == 'gone':
                print('Taking over endpoint {} from finished job {}.'.format(name, holder), flush=True)
                shutil.rmtree(str(lock), ignore_errors=True)
                stale = root / (name + '.json')
                if stale.exists() and _record_job(stale) in ('', holder):
                    stale.unlink()
                continue
            raise ValueError('Endpoint {} is held by job {}. Stop that job first, or check it with squeue '
                             'before removing {}.'.format(name, holder or '(unknown)', lock))
        (lock / 'owner').write_text(str(job) + '\n')
        # Teammates' jobs must read the owner to take over a stale claim.
        (lock / 'owner').chmod(0o644)
        return


def _record_job(path):
    try:
        return str(json.loads(path.read_text()).get('job_id', ''))
    except (OSError, ValueError):
        return ''


def publish(root, name, record, job):
    lock = root / ('.' + check_name(name) + '.lock')
    if _owner(lock) != str(job):
        raise ValueError('Claim endpoint {} before publishing it'.format(name))
    temporary = root / ('.' + name + '.' + str(job) + '.tmp')
    temporary.write_text(json.dumps(dict(record, name=name, owner=getpass.getuser()), indent=2) + '\n')
    temporary.chmod(0o644)
    os.replace(str(temporary), str(root / (name + '.json')))


def withdraw(root, name, job):
    """Remove the record and the claim, but only if `job` owns them."""
    lock = root / ('.' + check_name(name) + '.lock')
    if _owner(lock) != str(job):
        return
    record = root / (name + '.json')
    if record.exists() and _record_job(record) == str(job):
        record.unlink()
    shutil.rmtree(str(lock), ignore_errors=True)


def records(root):
    found = []
    for path in sorted(root.glob('*.json')):
        try:
            found.append(json.loads(path.read_text()))
        except (OSError, ValueError):
            continue
    return found
