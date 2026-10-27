"""One long reservation: in-node docker lifecycle + embedded router (the "stage").

Reservation restrictions make it impractical to ask for one short GPU job per model.
The stage shrinks the serving to a SINGLE allocation of 1 node / 8x B300 for as long as
possible; inside it a supervisor (this file, `stage.py serve`, run by stage_model.sbatch)
owns the rootless docker and loads/unloads whatever models are wanted, plus runs the
existing router so every loaded model is reached through one token.

  stage reserve            submit the single long 8-GPU job (serve/stage_model.sbatch)
  stage load   <profile>   inside the job: prepare + verify the image + docker run (its own 4-GPU slice)
  stage unload <profile>   stop that container and free its GPUs
  stage status             loaded models, router address, reservation state

Everything reuses the diagnostics/B300 recipe: node.ensure_docker (rootless-docker module),
node.verify_image (every flag and parser verified against `vllm serve --help=all` in the image
BEFORE the cold load of 704 GiB), node.docker_command (shm 16g, unlocked memlock, cache binds,
NCCL_ALGO=Ring) and inference.prepare_run (private run + fresh API key + RUN_CONFIG). The
reused router (router.py serve) discovers stage models through their .runtime/endpoint.json
branded "stage": true, never through the shared registry.

Control is file-based so no extra daemon or port is needed: commands are dropped as JSON files
in $BASE/users/<login>/stage/cmd/ and the supervisor moves them to done/ after acting.
"""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time
import urllib.error
import urllib.request

import inference
from inference import ROOT, load, save  # noqa: F401 (ROOT used by node)
import node
import router

STAGE_PORT = 18500     # embedded router listens here inside the node
API_PORT_BASE = 30200  # model API ports are allocated upward from here
GPU_PER_MODEL = 4      # each stage profile runs TP = gpus_per_node = 4
POLL = 3               # supervisor command/rescan interval, seconds


def stage_dir(base):
    return inference.user_dir(Path(base)) / 'stage'


def ensure_dirs(base):
    d = stage_dir(base)
    d.mkdir(parents=True, exist_ok=True)
    for sub in ('cmd', 'done'):
        (d / sub).mkdir(exist_ok=True)
    return d


def load_models(base):
    p = stage_dir(base) / 'models.json'
    return load(p) if p.exists() else {}


def save_models(base, records):
    save(stage_dir(base) / 'models.json', records)


def free_gpu_slice(records):
    """First 4-GPU block not claimed by a loaded model, or None (stage is full)."""
    used = set()
    for rec in records.values():
        used |= set(int(g) for g in rec['slice'].split(','))
    total = max(int(os.environ.get('STAGE_GPUS', '8')) // GPU_PER_MODEL, 1)
    for i in range(total):
        block = list(range(i * GPU_PER_MODEL, (i + 1) * GPU_PER_MODEL))
        if not any(g in used for g in block):
            return ','.join(map(str, block))
    return None


def free_api_port(records):
    used = {rec['api_port'] for rec in records.values()}
    port = API_PORT_BASE
    while port in used:
        port += 10
    return port


def request_healthy(port, timeout=5):
    try:
        with urllib.request.urlopen('http://127.0.0.1:{}/health'.format(port), timeout=timeout) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def load_model(base, profile_name, records):
    """prepare a run, verify the image, and start the container pinned to a free 4-GPU slice."""
    base = Path(base)
    ensure_dirs(base)
    run = inference.prepare_run(profile_name, base)
    cfg = load(run / 'RUN_CONFIG.json')
    p = cfg['profile']
    name = 'stage-{}'.format(p['id'])
    subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    slice_ = free_gpu_slice(records)
    if slice_ is None:
        raise RuntimeError('no free 4-GPU slice (stage already hosts {} model(s))'.format(len(records)))
    api_port = free_api_port(records)
    # node._ROOT must be the run directory: that is where the fresh API key lives.
    saved_root, node.ROOT = node.ROOT, run
    try:
        # Verify every flag and parser against the image BEFORE the cold load (diagnostic rule).
        extra = node.verify_image(cfg)
        engine = node.engine_command(cfg, 0, None, None, api_port) + extra
        saved = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = slice_
        try:
            argv = node.docker_command(cfg, engine, api_port=api_port, name=name)
        finally:
            if saved is None:
                os.environ.pop('CUDA_VISIBLE_DEVICES', None)
            else:
                os.environ['CUDA_VISIBLE_DEVICES'] = saved
    finally:
        node.ROOT = saved_root
    print('STAGE_LOAD {} slice {} port {} :: {}'.format(
        p['id'], slice_, api_port, ' '.join(shlex.quote(a) for a in argv)), flush=True)
    child = subprocess.Popen(argv)
    start = time.monotonic()
    while time.monotonic() - start < p['startup_timeout_seconds']:
        if child.poll() is not None:
            raise RuntimeError('{} container exited before readiness (rc {})'.format(p['id'], child.returncode))
        if request_healthy(api_port):
            break
        time.sleep(5)
    else:
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        raise RuntimeError('{} API not healthy within {} s'.format(p['id'], p['startup_timeout_seconds']))
    endpoint = {'job_id': name, 'stage': True, 'host': os.uname().nodename.split('.')[0],
                'port': api_port, 'model': p['served_model'], 'profile': p['id'],
                'runtime': p['runtime'], 'release': cfg['release'], 'gpu_slice': slice_,
                'ready_at': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                'note': 'stage-managed within the single reservation'}
    save(run / '.runtime/endpoint.json', endpoint)
    records[p['id']] = {'name': name, 'api_port': api_port, 'slice': slice_,
                        'served_model': p['served_model'], 'run': str(run)}
    save_models(base, records)
    print('STAGE_READY_MODEL {} {} slice {} port {}'.format(p['id'], p['served_model'], slice_, api_port), flush=True)
    return records


def unload_model(base, profile_name, records):
    rec = records.get(profile_name)
    if rec is None:
        return records
    base = Path(base)
    ensure_dirs(base)
    subprocess.run(['docker', 'rm', '-f', rec['name']], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ep = Path(rec['run']) / '.runtime/endpoint.json'
    if ep.exists():
        ep.unlink()
    records.pop(profile_name, None)
    save_models(base, records)
    print('STAGE_UNLOAD {}'.format(profile_name), flush=True)
    return records


def sweep(base, records, log):
    """Stop and withdraw any loaded model whose container or /health is gone."""
    for profile, rec in list(records.items()):
        try:
            alive = subprocess.run(['docker', 'inspect', '-f', '{{.State.Running}}', rec['name']],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL).stdout.decode().strip() == 'true'
        except OSError:
            alive = True
        if alive and request_healthy(rec['api_port']):
            continue
        unload_model(base, profile, records)
        log.write('withdrew {} (unhealthy or stopped)\n'.format(profile))
        log.flush()
    return records


def start_router(base, site, log):
    """Prepare the router directory and spawn router.py serve inside the reservation."""
    try:
        cfg = router.prepare(base, site, hours=24.0, restart=True, jobs=router.active_jobs())
        directory = Path(cfg['directory'])
    except ValueError as error:
        log.write('router.prepare: {}\n'.format(error))
        log.flush()
        fallback = stage_dir(base) / 'router'
        cfg = load(fallback / 'ROUTER.json') if (fallback / 'ROUTER.json').exists() else {'directory': str(fallback)}
        directory = Path(cfg.get('directory', str(fallback)))
    port = int(os.environ.get('ROUTER_PORT') or STAGE_PORT)
    env = dict(os.environ, ROUTER_PORT=str(port))
    child = subprocess.Popen([sys.executable, str(directory / 'router.py'), 'serve'],
                             cwd=str(directory), env=env)
    log.write('router pid {} on 0.0.0.0:{}\n'.format(child.pid, port))
    log.flush()
    return port, child


def serve(args):
    base = inference.site_base(args.base)
    site = inference.load_site(base)
    node.ensure_docker()
    d = ensure_dirs(base)
    log = (d / 'supervisor.out').open('a')
    records = load_models(base)
    port, router_child = start_router(base, site, log)
    save(stage_dir(base) / 'status.json',
         {'updated': time.time(), 'supervisor': 'running', 'router_port': port,
          'node': os.uname().nodename, 'models': {k: v['served_model'] for k, v in records.items()}})
    print('STAGE_SERVE ready; router 0.0.0.0:{} ; commands in {}'.format(port, d / 'cmd'), flush=True)
    cmd_dir, done_dir = d / 'cmd', d / 'done'
    while True:
        try:
            for cmd_path in sorted(cmd_dir.glob('*.cmd')):
                target = done_dir / cmd_path.name
                try:
                    action = json.loads(cmd_path.read_text())
                    act, profile = action.get('action'), action.get('profile')
                    if act == 'load':
                        try:
                            records = load_model(base, profile, records)
                            result = {'ok': True, 'action': 'load', 'profile': profile}
                        except Exception as error:  # noqa: BLE001 - report and stay up
                            result = {'ok': False, 'action': 'load', 'profile': profile, 'error': str(error)}
                            print('STAGE_LOAD_FAILED {} :: {}'.format(profile, error), flush=True)
                    elif act == 'unload':
                        records = unload_model(base, profile, records)
                        result = {'ok': True, 'action': 'unload', 'profile': profile}
                    else:
                        result = {'ok': False, 'action': act or '?', 'error': 'unknown action'}
                    target.write_text(json.dumps(result) + '\n')
                    cmd_path.unlink()
                    save(stage_dir(base) / 'status.json',
                         {'updated': time.time(), 'supervisor': 'running', 'router_port': port,
                          'node': os.uname().nodename, 'models': {k: v['served_model'] for k, v in records.items()}})
                except (ValueError, OSError) as error:
                    target.write_text(json.dumps({'ok': False, 'error': str(error)}) + '\n')
                    cmd_path.unlink()
            records = sweep(base, records, log)
        except KeyboardInterrupt:
            break
        time.sleep(POLL)
    if router_child.poll() is None:
        router_child.terminate()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['serve'])
    parser.add_argument('--base', required=True, help="team folder, e.g. '$WORK/team_vllm'")
    args = parser.parse_args()
    serve(args)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(str(error))
