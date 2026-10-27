"""One model-neutral container launcher per allocated node."""
import json
import os
from pathlib import Path
import shlex
import signal
import socket
import subprocess
import sys

import inference
from inference import ROOT, config, profile_check


def engine_command(cfg, rank, master, master_port, api_port):
    p = profile_check(cfg['profile'])
    parallel = p['parallelism']
    command = ['vllm', 'serve', cfg['model_path'], '--served-model-name', p['served_model'],
               '--tensor-parallel-size', str(parallel['tensor']), '--pipeline-parallel-size', str(parallel['pipeline']),
               '--data-parallel-size', str(parallel['data'])]
    if p['resources']['nodes'] > 1:
        command += ['--nnodes', str(p['resources']['nodes']), '--node-rank', str(rank),
                    '--master-addr', master, '--master-port', str(master_port)]
    command += p['engine_args']
    return command + (['--host', '0.0.0.0', '--port', str(api_port)] if rank == 0 else ['--headless'])


def container_command(cfg, command, gpu=True):
    cache = Path(cfg['cache_path'])
    variables = {'VLLM_ENABLE_CUDA_COMPATIBILITY': '1', 'VLLM_NO_USAGE_STATS': '1',
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                 'PYTHONPATH': cfg['overlay'], 'TMPDIR': '/tmp/inference-tmp',
                 'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple', 'NCCL_DEBUG': 'INFO', 'NCCL_DEBUG_SUBSYS': 'INIT,NET'}
    variables.update(cfg['profile'].get('environment', {}))
    directories = {'HF_HOME': 'hf', 'VLLM_CACHE_ROOT': 'vllm', 'TRITON_CACHE_DIR': 'triton',
                   'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'XDG_CACHE_HOME': 'xdg',
                   'FLASHINFER_WORKSPACE_BASE': 'flashinfer-root', 'TILELANG_CACHE_DIR': 'tilelang'}
    variables.update({key: str(cache / 'cache' / value) for key, value in directories.items()})
    if gpu:
        variables.update({'CUDA_VISIBLE_DEVICES': os.environ['CUDA_VISIBLE_DEVICES'],
                          'VLLM_HOST_IP': socket.gethostbyname(socket.gethostname()),
                          'VLLM_API_KEY': (ROOT / '.runtime/api_key').read_text().strip()})
    environment = {k: v for k, v in os.environ.items() if not k.startswith(('SINGULARITYENV_', 'APPTAINERENV_'))}
    prefix = 'APPTAINERENV_' if Path(cfg['container_command']).name == 'apptainer' else 'SINGULARITYENV_'
    environment.update({prefix + k: v for k, v in variables.items()})
    runner = [cfg['container_command'], 'exec'] + (['--nv'] if gpu else [])
    runner += ['--cleanenv', '--no-home', '--contain']
    # Bind each canonical asset explicitly. Legacy GLM-only mounts cannot expose
    # a runtime moved outside that model directory.
    for source, target, readonly in ((cfg['runtime_path'], cfg['runtime_path'], True),
                                     (cfg['model_path'], cfg['model_path'], True),
                                     (str(cache), str(cache), False), (str(ROOT), str(ROOT), False),
                                     (str(cache / 'tmp/runtime'), '/tmp/inference-tmp', False)):
        runner += ['--bind', source + ':' + target + (':ro' if readonly else '')]
    if gpu:
        runner += ['--bind', '/dev/infiniband']
    return runner + [cfg['sif']] + command, environment


# ---- docker launcher -----------------------------------------------------------------
# Everything below comes from diagnostics/glm53_b300_first_run.sh, the first-run script that put
# GLM-5.3 on one curiosity B300 node: no docker in the PATH (module rootless-docker/1.75, Bright,
# Slurm 25.05), GPU mode probed between --gpus all, --runtime=nvidia and CDI, caches bound from
# the team folder, --shm-size=16g and unlocked memlock, and every engine flag verified against
# `vllm serve --help=all` in the image BEFORE the 20-40 min cold load of 704 GiB.


def docker_image(cfg):
    return inference.catalog()['runtimes'][cfg['profile']['runtime']]['image']


def is_docker(cfg):
    return inference.catalog()['runtimes'][cfg['profile']['runtime']].get('kind') == 'docker'


def _run_quiet(argv):
    try:
        return subprocess.run(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
    except OSError:
        return 1


def _docker_works():
    return _run_quiet(['docker', 'info']) == 0


def ensure_docker():
    """curiosity has no docker in the PATH: the rootless-docker module provides it."""
    if _docker_works():
        return
    for init in ('/usr/share/lmd/lmod/init/bash', '/etc/profile.d/modules.sh', '/usr/share/Modules/init/bash'):
        if not Path(init).is_file():
            continue
        script = ('. {0} >/dev/null 2>&1; module load rootless-docker 2>/dev/null || '
                  'module load rootless-docker/1.75 2>/dev/null || true; printf %s "$PATH"').format(shlex.quote(init))
        result = subprocess.run(['bash', '-c', script], stdout=subprocess.PIPE)
        path = result.stdout.decode().strip()
        if path:
            os.environ['PATH'] = path
        if _docker_works():
            return
    raise RuntimeError('docker introuvable : module load rootless-docker, puis relancer '
                       '(cf. glm53_b300_first_run.sh preflight)')


def docker_user_args():
    """A rootless daemon already maps our uid; a rootful one needs --user to keep our files ours."""
    try:
        info = subprocess.run(['docker', 'info', '--format', '{{json .SecurityOptions}}'],
                              stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError:
        return []
    if b'rootless' in info.stdout.lower():
        return []
    return ['--user', '{}:{}'.format(os.getuid(), os.get_gid())]


_DOCKER_GPU_ARGS = None


def docker_gpu_args(cfg):
    """--gpus all (toolkit), --runtime=nvidia, or CDI: probed once with nvidia-smi in the image."""
    global _DOCKER_GPU_ARGS
    if _DOCKER_GPU_ARGS is not None:
        return _DOCKER_GPU_ARGS
    ensure_docker()
    image = docker_image(cfg)
    for candidate in (['--gpus', 'all'], ['--runtime=nvidia'], ['--device', 'nvidia.com/gpu=all']):
        probe = ['docker', 'run', '--rm'] + docker_user_args() + candidate + \
                ['--entrypoint', 'nvidia-smi', image, '-L']
        if _run_quiet(probe) == 0:
            _DOCKER_GPU_ARGS = candidate
            print('GPU_OK mode conteneur :', ' '.join(candidate), flush=True)
            return candidate
    raise RuntimeError('aucune methode GPU ne fonctionne (--gpus all / --runtime=nvidia / CDI) ; '
                       'cf. gpu_args() dans glm53_b300_first_run.sh')


def docker_environment(cfg, gpu):
    cache = Path(cfg['cache_path'])
    variables = {'HOME': '/tmp', 'VLLM_NO_USAGE_STATS': '1',
                 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
                 'NCCL_ALGO': 'Ring', 'NCCL_PROTO': 'Simple'}
    directories = {'HF_HOME': 'hf', 'VLLM_CACHE_ROOT': 'vllm', 'TRITON_CACHE_DIR': 'triton',
                   'TORCHINDUCTOR_CACHE_DIR': 'inductor', 'XDG_CACHE_HOME': 'xdg',
                   'FLASHINFER_WORKSPACE_BASE': 'flashinfer-root', 'TILELANG_CACHE_DIR': 'tilelang'}
    variables.update({key: str(cache / 'cache' / value) for key, value in directories.items()})
    variables.update(cfg['profile'].get('environment', {}))
    if gpu:
        # Slurm already sets CUDA_VISIBLE_DEVICES for the task: pass it through untouched.
        if os.environ.get('CUDA_VISIBLE_DEVICES'):
            variables['CUDA_VISIBLE_DEVICES'] = os.environ['CUDA_VISIBLE_DEVICES']
        variables['VLLM_API_KEY'] = (ROOT / '.runtime/api_key').read_text().strip()
    return variables


def docker_command(cfg, command, gpu=True, api_port=None, name=None):
    """The docker run line: same-path mounts, cache env, 16g shm, unlocked memlock (diagnostics recipe)."""
    ensure_docker()
    argv = ['docker', 'run', '--rm']
    if name:
        argv += ['--name', name]
    argv += docker_user_args()
    if gpu:
        argv += docker_gpu_args(cfg)
    cache = Path(cfg['cache_path'])
    argv += ['--volume', '{}:{}:ro'.format(cfg['model_path'], cfg['model_path']),
             '--volume', '{}:{}'.format(cache, cache),
             '--shm-size=16g', '--ulimit', 'memlock=-1', '--ulimit', 'stack=67108864']
    if api_port:
        argv += ['--publish', '{}:{}'.format(api_port, api_port)]
    for key, value in sorted(docker_environment(cfg, gpu).items()):
        argv += ['--env', '{}={}'.format(key, value)]
    return argv + ['--entrypoint', command[0], docker_image(cfg)] + list(command[1:])


# The registry probe from the diagnostics script: canonical list_registered(), falling back to
# the manager dicts, so a missing method on one parser manager cannot void the whole registry.
PARSER_PROBE = """import json
from vllm.tool_parsers import ToolParserManager as T
from vllm.reasoning import ReasoningParserManager as R
def list_reg(M, *attrs):
    try:
        v = M.list_registered()
        if v:
            return sorted(v)
    except Exception:
        pass
    out = set()
    for a in attrs:
        d = getattr(M, a, None)
        if isinstance(d, dict):
            out |= set(d.keys())
    return sorted(out)
print(json.dumps({"tool": list_reg(T, "tool_parsers", "lazy_parsers"),
                  "reasoning": list_reg(R, "reasoning_parsers", "tool_parsers", "lazy_parsers")}))"""


def verify_image(cfg):
    """The diagnostics' guard: verify every flag and parser against the image BEFORE the cold load.

    An unknown flag only kills vLLM after the weights are loaded (a 20-40 min loss on GLM-5.3),
    so `vllm serve --help=all` and the parser registry are dumped into the run's cache once
    and every profile flag is checked against them before anything heavy starts.
    """
    ensure_docker()
    cache = Path(cfg['cache_path'])
    help_path = cache / 'vllm-serve-help.txt'
    parsers_path = cache / 'vllm-parsers.json'
    subprocess.run(['docker', 'pull', docker_image(cfg)], check=True)
    if not help_path.is_file() or not help_path.stat().st_size:
        with help_path.open('w') as handle:
            result = subprocess.run(docker_command(cfg, ['sh', '-c', 'vllm serve --help=all']), stdout=handle)
        if result.returncode or not help_path.is_file() or not help_path.stat().st_size:
            if help_path.exists():
                help_path.unlink()
            raise RuntimeError('vllm serve --help=all a echoue dans l image')
    if not parsers_path.is_file():
        result = subprocess.run(docker_command(cfg, ['python3', '-c', PARSER_PROBE]), stdout=subprocess.PIPE)
        if result.returncode:
            raise RuntimeError('registre des parseurs illisible dans l image')
        parsers_path.write_bytes(result.stdout)
    try:
        registry = json.loads(parsers_path.read_text())
    except (OSError, ValueError):
        registry = {}
    help_text = help_path.read_text()
    extra = []
    for kind, flag in (('tool', '--tool-call-parser'), ('reasoning', '--reasoning-parser')):
        names = cfg['profile'].get('parser_candidates', {}).get(kind)
        if not names:
            continue
        chosen = next((n for n in names if n in (registry.get(kind) or [])), None)
        if chosen is None:
            chosen = next((n for n in names if n in help_text), None)
        if chosen is None:
            raise RuntimeError('aucun parseur {} dans cette image ; candidats : {} ; registre : {}'.format(
                kind, ', '.join(names), ', '.join(registry.get(kind) or []) or '(vide)'))
        print('{}-parser: {}'.format(kind, chosen), flush=True)
        extra += [flag, chosen]
    flags = {value.split('=', 1)[0] for value in cfg['profile']['engine_args'] if value.startswith('--')}
    flags |= {value for value in extra if value.startswith('--')}
    missing = sorted(flag for flag in flags if flag not in help_text)
    if missing:
        raise RuntimeError('FLAGS ABSENTS de cette image : {} (voir {} ; ajustez engine_args ou '
                           'parser_candidates du profil)'.format(', '.join(missing), help_path))
    return extra


def docker_import_check(cfg):
    """`inference.py cpu-check` for docker runtimes: import the runtime's modules in the image."""
    modules = ','.join(inference.catalog()['runtimes'][cfg['profile']['runtime']]['imports'])
    code = ('import importlib\n'
            'for name in "' + modules + '".split(","):\n'
            '    importlib.import_module(name)\n'
            '    print("RUNTIME_IMPORT_OK", name)')
    subprocess.check_call(docker_command(cfg, ['python3', '-c', code]))


def docker_serve(cfg, api_port):
    """Run vLLM in the foreground so Slurm owns its lifetime; TERM removes the container by name."""
    p = profile_check(cfg['profile'])
    if p['resources']['nodes'] != 1:
        raise ValueError('Le lanceur docker est valide sur un seul noeud (recette B300 du script de diagnostic)')
    extra = verify_image(cfg)
    command = engine_command(cfg, 0, None, None, api_port) + extra
    name = 'serve-{}-{}'.format(p['id'], os.environ.get('SLURM_JOB_ID', 'local'))
    argv = docker_command(cfg, command, api_port=api_port, name=name)
    shown = [value if not value.startswith('VLLM_API_KEY=') else 'VLLM_API_KEY=[REDACTED]' for value in argv]
    print('DOCKER_RUN', ' '.join(shlex.quote(value) for value in shown), flush=True)
    child = subprocess.Popen(argv)

    def stop(signum, frame):
        subprocess.run(['docker', 'rm', '-f', name], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            child.kill()
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    return child.wait()


def main():
    os.umask(0o077)
    cfg = config()
    p = cfg['profile']
    rank = int(os.environ['SLURM_NODEID'])
    resources = p['resources']
    if int(os.environ['SLURM_JOB_NUM_NODES']) != resources['nodes'] or not 0 <= rank < resources['nodes']:
        raise ValueError('Allocation differs from selected profile')
    if is_docker(cfg):
        # The B300/docker recipe: single node, image pulled and every flag verified on this node.
        sys.exit(docker_serve(cfg, int(os.environ['INFERENCE_API_PORT'])))
    master = os.environ['INFERENCE_MASTER_ADDR']
    master_port = int(os.environ['INFERENCE_MASTER_PORT'])
    if 'nccl_ep' in p.get('hooks', []):
        command = ['python3', '-c', 'import os,sys,vllm; os.execv(sys.executable,[sys.executable,*sys.argv[1:]])',
                   '-m', 'torch.distributed.run', '--nnodes=' + str(resources['nodes']),
                   '--nproc-per-node=' + str(resources['gpus_per_node']), '--max-restarts=0',
                   '--node-rank=' + str(rank), '--master-addr=' + master,
                   '--master-port=' + str(master_port + 1), str(ROOT / 'hooks/nccl_ep.py')]
        argv, environment = container_command(cfg, command)
        subprocess.run(argv, env=environment, check=True, timeout=240)
    if 'warm_engram' in p.get('hooks', []):
        with (ROOT / 'evidence/engram-prewarm-{}-{}.log'.format(os.environ['SLURM_JOB_ID'], rank)).open('w') as handle:
            subprocess.check_call([sys.executable, str(ROOT / 'hooks/warm_engram.py'), cfg['model_path'],
                                   str(rank), str(resources['nodes'])], stdout=handle, stderr=subprocess.STDOUT)
    argv, environment = container_command(cfg, engine_command(cfg, rank, master, master_port, int(os.environ['INFERENCE_API_PORT'])))
    os.execvpe(argv[0], argv, environment)


if __name__ == '__main__':
    main()
