"""Shared, standard-library-only deployment and client CLI (Python 3.6+)."""
import argparse
import getpass
import hashlib
import json
import os
from pathlib import Path
import re
import secrets
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parent
# API checks that must pass before an endpoint is published. Profiles pick a suite with "checks".
CHECK_SUITES = {
    'chat_tools': ('unauthenticated_rejected', 'model_catalog', 'arithmetic', 'english_response',
                   'sse_streaming', 'forced_function_call', 'four_concurrent_requests'),
    'chat': ('unauthenticated_rejected', 'model_catalog', 'arithmetic', 'english_response',
             'sse_streaming', 'four_concurrent_requests'),
    'minimal': ('unauthenticated_rejected', 'model_catalog', 'any_completion'),
    'classify': ('unauthenticated_rejected', 'model_catalog', 'classification'),
}


def load(path):
    return json.loads(Path(path).read_text())


def catalog():
    return load(ROOT / 'catalog.json')


def runtime_kind(profile):
    """'docker' runtimes launch through the node's rootless docker; others through apptainer."""
    return catalog()['runtimes'][profile['runtime']].get('kind', 'singularity')


def check_suite(profile):
    return CHECK_SUITES[profile.get('checks', 'chat_tools')]


def save(path, obj):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w') as handle:
        json.dump(obj, handle, indent=2)
        handle.write('\n')
    os.replace(str(temporary), str(path))


def identifier(value):
    if not re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', value):
        raise ValueError('Invalid identifier: ' + value)
    return value


def profile_check(profile):
    if profile['schema_version'] != 1:
        raise ValueError('Unsupported profile schema')
    for name in ('id', 'checkpoint', 'runtime', 'served_model'):
        identifier(profile[name])
    resources, parallel = profile['resources'], profile['parallelism']
    if not parallel.get('data'): parallel['data']=1
    for name in ('nodes', 'gpus_per_node', 'cpus_per_task'):
        if type(resources[name]) is not int or resources[name] < 1:
            raise ValueError('Invalid resource: ' + name)
    if any(type(parallel[k]) is not int or parallel[k] < 1 for k in ('tensor', 'pipeline', 'data')):
        raise ValueError('Invalid parallelism')
    if parallel['tensor'] * parallel['pipeline'] * parallel['data'] != resources['nodes'] * resources['gpus_per_node']:
        raise ValueError('TP x PP x DP must match the allocated GPU count')
    if not re.fullmatch(r'[1-9][0-9]*(?:[KMGTP]B?)?', resources['memory']):
        raise ValueError('Invalid memory request')
    if not re.fullmatch(r'(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]', resources['time']):
        raise ValueError('Invalid time limit')
    reserved = {'--served-model-name', '--tensor-parallel-size', '--pipeline-parallel-size', '--data-parallel-size',
                '--nnodes', '--node-rank', '--master-addr', '--master-port', '--host', '--port',
                '--headless', '--api-key', '--config'}
    args = profile['engine_args']
    if not isinstance(args, list) or not all(isinstance(v, str) for v in args):
        raise ValueError('engine_args must be a string array')
    if any(v.split('=', 1)[0] in reserved for v in args):
        raise ValueError('Engine arguments override a managed connection or parallelism flag')
    if set(profile.get('hooks', [])) - {'nccl_ep', 'warm_engram'}:
        raise ValueError('Unknown pre-start hook')
    if 'nccl_ep' in profile.get('hooks', []) and (parallel['pipeline'] != 1 or '--enable-expert-parallel' not in args):
        raise ValueError('The EP smoke hook requires PP=1 and expert parallelism')
    # Parsers the docker launcher resolves from the image's own registry (stock images rename
    # them between versions; the diagnostics script showed the names must not be guessed).
    candidates = profile.get('parser_candidates', {})
    if not isinstance(candidates, dict) or set(candidates) - {'tool', 'reasoning'}:
        raise ValueError('parser_candidates must hold only "tool" and "reasoning" lists')
    for kind, names in candidates.items():
        parser_flag = '--tool-call-parser' if kind == 'tool' else '--reasoning-parser'
        if not isinstance(names, list) or not names or not all(
                isinstance(n, str) and re.fullmatch(r'[a-zA-Z0-9][a-zA-Z0-9_.-]*', n) for n in names):
            raise ValueError('parser_candidates.{} must be a non-empty list of parser names'.format(kind))
        if parser_flag in [value.split('=', 1)[0] for value in args]:
            raise ValueError('parser_candidates.{} conflicts with an explicit {} in engine_args'.format(kind, parser_flag))
    if not 1 <= profile['client']['local_port'] <= 65535 or profile['client']['timeout_seconds'] <= 0:
        raise ValueError('Invalid client port or timeout')
    if profile['startup_timeout_seconds'] <= 0:
        raise ValueError('Invalid startup timeout')
    if profile.get('checks', 'chat_tools') not in CHECK_SUITES:
        raise ValueError('Unknown check suite; use one of ' + ', '.join(sorted(CHECK_SUITES)))
    if profile['runtime'] not in catalog()['runtimes']:
        raise ValueError('Runtime is not in catalog.json: ' + profile['runtime'])
    env = profile.get('environment', {})
    if not isinstance(env, dict) or any(not re.fullmatch(r'[A-Z][A-Z0-9_]*', k) or not isinstance(v, str) for k, v in env.items()):
        raise ValueError('Invalid container environment')
    protected = {'VLLM_API_KEY', 'PYTHONPATH', 'VLLM_HOST_IP', 'CUDA_VISIBLE_DEVICES', 'HF_HOME', 'TMPDIR'}
    if protected.intersection(env):
        raise ValueError('Profile overrides a private/runtime environment variable')
    return profile


def verify_release(root=ROOT):
    manifest = load(root / 'RELEASE_MANIFEST.json')
    for name, expected in manifest['files'].items():
        path = root / name
        if Path(name).is_absolute() or '..' in Path(name).parts or path.is_symlink():
            raise ValueError('Unsafe release manifest path')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Release checksum mismatch: ' + name)
    return manifest


def private_dir(path):
    path = Path(path)
    if path.is_symlink():
        raise ValueError('Private directory must not be a symlink: ' + str(path))
    path.mkdir(mode=0o700, exist_ok=True)
    if path.stat().st_uid != os.getuid():
        raise ValueError('Private directory belongs to another account: ' + str(path))
    path.chmod(0o700)
    return path


def config():
    return load(ROOT / 'RUN_CONFIG.json')


def preflight(cfg):
    verify_release()
    profile_check(cfg['profile'])
    model = Path(cfg['model_path'])
    if (model / 'model.safetensors.index.json').exists():
        shards = sorted(set(load(model / 'model.safetensors.index.json')['weight_map'].values()))
    else:
        shards = ['model.safetensors']  # small checkpoints come as a single unindexed file
    expected = cfg['profile'].get('expected_shards')
    if not shards or (expected is not None and len(shards) != expected):
        raise ValueError('Unexpected checkpoint shard count')
    required = [model / 'config.json', model / 'tokenizer_config.json']
    docker = runtime_kind(cfg['profile']) == 'docker'
    if not docker:
        # Docker runtimes have no SIF and no overlay: the image is pulled on the GPU node.
        required.append(Path(cfg['sif']))
        for name in catalog()['runtimes'][cfg['profile']['runtime']]['overlay_required']:
            required.append(Path(cfg['overlay']) / name)
        for item in catalog()['runtimes'][cfg['profile']['runtime']].get('overlay_files', []):
            path = Path(cfg['overlay']) / item['target']
            if path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != item['sha256']:
                raise ValueError('Runtime patch checksum mismatch: ' + item['target'])
    for name in shards:
        if Path(name).is_absolute() or '..' in Path(name).parts:
            raise ValueError('Unsafe checkpoint shard path')
        required.append(model / name)
    for path in required:
        with path.open('rb') as handle:
            if not handle.read(1):
                raise ValueError('Empty required asset: ' + str(path))
    if not docker:
        runtime = load(Path(cfg['runtime_path']) / 'runtime.json')
        if runtime['id'] != cfg['profile']['runtime'] or Path(cfg['sif']).stat().st_size != runtime['sif_size']:
            raise ValueError('Runtime registration mismatch')
        if not shutil.which(cfg['container_command']):
            raise ValueError('Container runner must be available on this host')
    if not shutil.which('sbatch'):
        raise ValueError('Slurm must be available on this host')
    cache = Path(cfg['cache_path'])
    private_dir(cache.parent.parent)  # $BASE/users/<login>: holds keys, so never shared
    private_dir(cache.parent)
    private_dir(cache)
    for subdir in ('cache', 'tmp'):
        private_dir(cache / subdir)
    for subdir in ('hf', 'vllm', 'triton', 'inductor', 'xdg', 'flashinfer-root', 'tilelang'):
        private_dir(cache / 'cache' / subdir)
    private_dir(cache / 'tmp/runtime')
    identity = cfg.get('image') or load(Path(cfg['runtime_path']) / 'runtime.json')['id']
    print('PREFLIGHT_OK: {} readable shards; runtime {}; private cache {}.'.format(
        len(shards), identity, cache), file=sys.stderr)


def site_base(value):
    raw = value or os.environ.get('VLLM_TEAM_BASE')
    if not raw:
        raise ValueError('Specify --base or VLLM_TEAM_BASE')
    # The admin CLI passes '$WORK/<team>' unexpanded so it resolves on Leonardo.
    return Path(os.path.expandvars(raw)).expanduser().resolve()


def load_site(base):
    if not (base / 'inference-site.json').exists():
        raise ValueError('{} is not a team folder yet (no inference-site.json); create it with: cli.py init'.format(base))
    site = load(base / 'inference-site.json')
    if site['schema_version'] != 1:
        raise ValueError('Unsupported site schema')
    return site


def user_dir(base):
    """$BASE/users/<login>: this account's runs, caches, router and logs. Mode 700."""
    return base / 'users' / identifier(getpass.getuser())


def runtime_dir(base, site, runtime_id):
    # Fixed layout unless the site file maps a runtime somewhere else (older installations).
    return base / site.get('runtime_paths', {}).get(runtime_id, 'runtimes/' + runtime_id)


def checkpoint_dir(base, site, name):
    return base / site.get('checkpoints', {}).get(name, 'checkpoints/' + name)


def prepare_run(model, base_arg):
    manifest = verify_release()
    name = identifier(model)
    if not (ROOT / 'profiles' / (name + '.json')).exists():
        raise ValueError('Unknown profile {}; available: {}'.format(
            name, ', '.join(sorted(p.stem for p in (ROOT / 'profiles').glob('*.json')))))
    profile = profile_check(load(ROOT / 'profiles' / (name + '.json')))
    base = site_base(base_arg)
    site = load_site(base)
    kind = runtime_kind(profile)
    if kind == 'docker' and profile.get('hooks'):
        raise ValueError('The docker launcher runs no pre-start hooks')
    runtime_path = None
    if kind != 'docker':
        runtime_path = runtime_dir(base, site, profile['runtime']).resolve()
        if not (runtime_path / 'runtime.json').exists():
            raise ValueError('Runtime {} is not installed; run: cli.py install {}'.format(profile['runtime'], name))
        runtime = load(runtime_path / 'runtime.json')
        if runtime['id'] != profile['runtime']:
            raise ValueError('Wrong runtime registration')
    model_path = checkpoint_dir(base, site, profile['checkpoint']).resolve()
    if not (model_path / 'config.json').exists():
        raise ValueError('Checkpoint {} is not installed; run: cli.py install {}'.format(profile['checkpoint'], name))
    if not (base / 'users').is_dir():
        raise ValueError('{}/users is missing; create the team folder with: cli.py init'.format(base))
    mine = user_dir(base)
    cfg = {'schema_version': 1, 'profile': profile, 'site': site, 'base': str(base),
           'source_release': str(ROOT), 'release': manifest['release'],
           'model_path': str(model_path),
           'cache_path': str(mine / 'cache' / (name + '--' + profile['runtime'])),
           'container_command': 'docker' if kind == 'docker' else
                               (shutil.which('singularity') or shutil.which('apptainer') or 'singularity')}
    if kind == 'docker':
        # No local runtime install: the image is pulled on the GPU node at serve time.
        cfg['image'] = catalog()['runtimes'][profile['runtime']]['image']
    else:
        cfg['runtime_path'] = str(runtime_path)
        cfg['sif'] = str(runtime_path / runtime['sif'])
        cfg['overlay'] = str(runtime_path / runtime['overlay'])
    preflight(cfg)
    run = Path(tempfile.mkdtemp(prefix=name + '-', dir=str(private_dir(mine / 'runs'))))
    run.chmod(0o700)
    for filename in manifest['files']:
        destination = run / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(ROOT / filename), str(destination))
        destination.chmod(0o600)
    shutil.copyfile(str(ROOT / 'RELEASE_MANIFEST.json'), str(run / 'RELEASE_MANIFEST.json'))
    private_dir(run / '.runtime')
    private_dir(run / 'evidence')
    key_path = run / '.runtime/api_key'
    fd = os.open(str(key_path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(secrets.token_urlsafe(32) + '\n')
    save(run / 'RUN_CONFIG.json', cfg)
    return run


def prepare(args):
    run = prepare_run(args.model, args.base)
    # stdout is the run path only, allowing RUN_DIR=$(...). Never print the key.
    print(run)


def assert_private_run():
    if Path.cwd().resolve() != ROOT:
        raise ValueError('cd into the private run directory first')
    for path in (ROOT, ROOT / '.runtime', ROOT / '.runtime/api_key'):
        mode = path.lstat()
        if stat.S_ISLNK(mode.st_mode) or mode.st_uid != os.getuid() or mode.st_mode & 0o077:
            raise ValueError('Run directory and key must be private and owned by this account')


def time_limit(value):
    if not re.fullmatch(r'(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]', value):
        raise ValueError('Invalid time limit: ' + value)
    return value


def submit_command(cfg, reservation=None, qos=None, time=None):
    p, site = cfg['profile'], cfg['site']
    resources = p['resources']
    account = os.environ.get('SBATCH_ACCOUNT') or site.get('account')
    if account == 'YOUR_PROJECT':
        raise ValueError('Set a valid project allocation in the site configuration')
    qos = qos or site.get('qos')
    for value in (account, site['partition'], qos):
        if value:
            identifier(value)
    command = ['sbatch', '--parsable', '--job-name=' + p['id'] + '-common',
               '--partition=' + identifier(site['partition']),
               '--nodes=' + str(resources['nodes']), '--ntasks-per-node=1',
               '--gres=gpu:' + str(resources['gpus_per_node']),
               '--cpus-per-task=' + str(resources['cpus_per_task']), '--mem=' + resources['memory'],
               '--time=' + time_limit(time or resources['time']), '--signal=B:TERM@90', '--output=%x-%j.out',
               '--chdir=' + str(ROOT)]
    if account:
        command.append('--account=' + identifier(account))
    if qos:
        command.append('--qos=' + identifier(qos))
    if reservation:
        command.append('--reservation=' + identifier(reservation))
    return command + [str(ROOT / 'serve_model.sbatch')]


def sbatch(command, marker, cwd=None):
    """Run sbatch once; `marker` guards against a second submission when the outcome is unknown."""
    fd = os.open(str(marker), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    os.close(fd)
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    if result.returncode:
        marker.unlink()
        raise RuntimeError(result.stderr.decode(errors='replace'))
    job = result.stdout.decode().strip().split(';')[0]
    if not re.fullmatch(r'[0-9]+', job):
        raise RuntimeError('Submission outcome uncertain. Keep {} and check squeue before retrying.'.format(marker))
    marker.unlink()
    return job


def submit(args):
    assert_private_run()
    cfg = config()
    preflight(cfg)
    command = submit_command(cfg, args.reservation, args.qos, args.time)
    if args.dry_run:
        # The exact command, ready to run from this directory by hand.
        print(' '.join(shlex.quote(word) for word in command))
        return
    if (ROOT / '.runtime/job_id').exists():
        raise ValueError('This run has already been submitted; prepare a new run')
    job = sbatch(command, ROOT / '.runtime/submitting')
    (ROOT / '.runtime/job_id').write_text(job + '\n')
    print('Submitted job ' + job)
    print('Wait for INFERENCE_READY in {}-common-{}.out, then read .runtime/endpoint.json.'.format(cfg['profile']['id'], job))


def cpu_check():
    cfg = config()
    preflight(cfg)
    if runtime_kind(cfg['profile']) == 'docker':
        from node import docker_import_check
        docker_import_check(cfg)
        return
    from node import container_command
    modules = catalog()['runtimes'][cfg['profile']['runtime']]['imports']
    code = 'import {0}; print("RUNTIME_IMPORT_OK", {1})'.format(
        ','.join(modules), ','.join('{0}.__version__ if hasattr({0}, "__version__") else {0}.__file__'.format(m) for m in modules))
    command, environment = container_command(cfg, ['python3', '-c', code], gpu=False)
    subprocess.check_call(command, env=environment)




def main():
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__ + ' Team-folder commands (up, status, install, ...) are in team.py.')
    sub = parser.add_subparsers(dest='command')
    p = sub.add_parser('prepare', help='check assets and create a private run directory; prints its path')
    p.add_argument('model')
    p.add_argument('--base')
    p = sub.add_parser('submit', help='submit the prepared run (run from its directory)')
    p.add_argument('--reservation')
    p.add_argument('--qos', help='override the site QoS, e.g. boost_qos_dbg')
    p.add_argument('--time', help='override the profile time limit, e.g. 00:30:00')
    p.add_argument('--dry-run', action='store_true')
    sub.add_parser('preflight', help='re-check the run directory and assets')
    sub.add_parser('cpu-check', help='import the runtime modules in the container, no GPU')
    sub.add_parser('info', help='print the resolved run configuration')
    args = parser.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'submit':
        submit(args)
    elif args.command == 'preflight':
        assert_private_run()
        preflight(config())
    elif args.command == 'cpu-check':
        assert_private_run()
        cpu_check()
    elif args.command == 'info':
        cfg = config()
        print(json.dumps({k: cfg[k] for k in ('release', 'source_release', 'profile', 'runtime_path', 'model_path', 'cache_path')}, indent=2))
    else:
        parser.print_help()


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(str(error))
