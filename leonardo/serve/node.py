"""One model-neutral container launcher per allocated node."""
import os
from pathlib import Path
import socket
import subprocess
import sys

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


def main():
    os.umask(0o077)
    cfg = config()
    p = cfg['profile']
    rank = int(os.environ['SLURM_NODEID'])
    resources = p['resources']
    if int(os.environ['SLURM_JOB_NUM_NODES']) != resources['nodes'] or not 0 <= rank < resources['nodes']:
        raise ValueError('Allocation differs from selected profile')
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
