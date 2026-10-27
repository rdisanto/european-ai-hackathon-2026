"""Eight-rank English-only correctness preflight, without model weights."""
import datetime
import json
import os
import socket
import time

import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
local = int(os.environ['LOCAL_RANK'])
world = int(os.environ['WORLD_SIZE'])
nodes = int(os.environ["GROUP_WORLD_SIZE"])
local_world = int(os.environ["LOCAL_WORLD_SIZE"])
assert world == nodes * local_world
start = time.monotonic()
torch.cuda.set_device(local)
dist.init_process_group('nccl', timeout=datetime.timedelta(seconds=90))
for count in (1024, 1024 * 1024):
    tensor = torch.full((count,), float(rank + 1), device='cuda', dtype=torch.float32)
    dist.all_reduce(tensor)
    torch.cuda.synchronize()
    assert torch.all(tensor == world * (world + 1) // 2).item()
print('NCCL_SMOKE_OK', json.dumps({'rank': rank, 'world_size': world,
      'host': socket.gethostname(), 'device': torch.cuda.get_device_name(local),
      'torch': torch.__version__, 'cuda': torch.version.cuda,
      'seconds': round(time.monotonic() - start, 3)}), flush=True)
gathered = torch.empty(world * 1024, device='cuda', dtype=torch.float32)
source = torch.full((1024,), float(rank + 1), device='cuda', dtype=torch.float32)
dist.all_gather_into_tensor(gathered, source)
expected = torch.arange(1, world + 1, device='cuda', dtype=torch.float32).repeat_interleave(1024)
assert torch.equal(gathered, expected)
reduced = torch.empty(1024, device='cuda', dtype=torch.float32)
dist.reduce_scatter_tensor(reduced, gathered)
assert torch.all(reduced == float(world * (rank + 1))).item()
print('EP_COLLECTIVES_OK', rank, flush=True)
dist.barrier()
print('VLLM_NCCL_SMOKE_START' , rank, flush=True)
from vllm.distributed.parallel_state import (
    init_distributed_environment, initialize_model_parallel,
    destroy_model_parallel, set_custom_all_reduce,
)
from vllm.distributed.communication_op import tensor_model_parallel_all_reduce
from vllm.config import VllmConfig, ParallelConfig, set_current_vllm_config
config = VllmConfig(parallel_config=ParallelConfig(
    tensor_parallel_size=world, pipeline_parallel_size=1,
    nnodes=nodes, node_rank=rank // local_world, distributed_executor_backend='mp',
    enable_expert_parallel=True, all2all_backend='allgather_reducescatter'))
with set_current_vllm_config(config):
    set_custom_all_reduce(False)
    init_distributed_environment(world_size=world, rank=rank, local_rank=local,
                                 timeout=datetime.timedelta(seconds=90))
    initialize_model_parallel(tensor_model_parallel_size=world,
                              pipeline_model_parallel_size=1)
    tensor = torch.full((1024,), float(rank + 1), device='cuda', dtype=torch.float32)
    result = tensor_model_parallel_all_reduce(tensor)
    torch.cuda.synchronize()
    assert torch.all(result == world * (world + 1) // 2).item()
    print('VLLM_NCCL_SMOKE_OK', rank, flush=True)
    destroy_model_parallel()
dist.barrier()
dist.destroy_process_group()
