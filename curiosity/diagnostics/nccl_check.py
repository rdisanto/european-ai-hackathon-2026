"""All-reduce bandwidth check across Slurm tasks (one task per GPU).

Measures inside each node (NVLink) and across all nodes (InfiniBand or, if
something is misconfigured, TCP sockets). Run it with NCCL_DEBUG=INFO and look
for 'NET/IB' vs 'NET/Socket' in the log.
"""
import os
import socket
import time

# vLLM initializes CUDA forward-compatibility paths before torch loads CUDA.
# The standalone check must do this too on Leonardo's older host driver.
import vllm
import torch
import torch.distributed as dist


def bench(nbytes, group, iters=20, warmup=5):
    x = torch.ones(nbytes // 2, dtype=torch.bfloat16, device="cuda")
    for _ in range(warmup):
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    dist.barrier(group=group)
    t0 = time.perf_counter()
    for _ in range(iters):
        dist.all_reduce(x, group=group)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / iters
    k = dist.get_world_size(group)
    algbw = nbytes / dt / 1e9
    return dt * 1e3, algbw * 2 * (k - 1) / k


def main():
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local)
    dist.init_process_group("nccl", rank=rank, world_size=world)

    me = (rank, socket.gethostname(), local)
    everyone = [None] * world
    dist.all_gather_object(everyone, me)
    hosts = sorted({h for _, h, _ in everyone})

    # every rank must create every group, in the same order
    node_groups = {h: dist.new_group([r for r, hh, _ in everyone if hh == h]) for h in hosts}
    my_node_group = node_groups[me[1]]

    # correctness: sum of ones over all ranks == world
    y = torch.ones(1024, device="cuda")
    dist.all_reduce(y)
    ok = bool(torch.all(y == world).item())
    if not ok:
        raise RuntimeError("NCCL all-reduce correctness check failed")

    if rank == 0:
        print(f"vLLM {vllm.__version__}, torch {torch.__version__}, NCCL {torch.cuda.nccl.version()}, "
              f"{world} ranks on {len(hosts)} node(s): {', '.join(hosts)}")
        print(f"correctness check: {'OK' if ok else 'FAILED'}")
        print(f"{'scope':<28}{'size':>8}{'time ms':>10}{'busbw GB/s':>12}")

    for nbytes in (256 << 20, 1 << 30):
        ms, bw = bench(nbytes, my_node_group)
        if rank == 0:
            print(f"{'intra-node (NVLink)':<28}{nbytes >> 20:>6}MB{ms:>10.2f}{bw:>12.1f}")
        if len(hosts) > 1:
            ms, bw = bench(nbytes, None)
            if rank == 0:
                print(f"{'all nodes (network)':<28}{nbytes >> 20:>6}MB{ms:>10.2f}{bw:>12.1f}")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
