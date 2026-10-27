"""Sequentially warm this node's approximate TP-owned Engram rows.

Read-only page-cache optimization. Safetensors headers give exact tensor byte
ranges; a row margin covers the small imbalance between Engram hash heads.
"""
import json
import os
from pathlib import Path
import struct
import time
import sys

base = Path(sys.argv[1])
node, nodes = map(int, sys.argv[2:])
assert 0 <= node < nodes
buffer = bytearray(16 * 1024 * 1024)
for number in (47, 48):
    path = base / ('model-%05d-of-00048.safetensors' % number)
    with path.open('rb', buffering=0) as handle:
        length = struct.unpack('<Q', handle.read(8))[0]
        header = json.loads(handle.read(length))
        for name, tensor in sorted(header.items()):
            if '.engram.embed.' not in name:
                continue
            rows = tensor['shape'][0]
            first, last = tensor['data_offsets']
            row_bytes = (last - first) // rows
            assert row_bytes * rows == last - first
            low = max(0, rows * node // nodes - 131072)
            high = min(rows, rows * (node + 1) // nodes + 131072)
            remaining = (high - low) * row_bytes
            handle.seek(8 + length + first + low * row_bytes)
            start = time.monotonic()
            print('WARM_START', os.uname().nodename, path.name, name, remaining, flush=True)
            total = remaining
            progress = 0
            while remaining:
                count = handle.readinto(memoryview(buffer)[:min(len(buffer), remaining)])
                if not count:
                    raise RuntimeError('Unexpected EOF')
                remaining -= count
                if total - remaining >= progress + 16 * 1024**3:
                    progress = total - remaining
                    print('WARM_PROGRESS', os.uname().nodename, name, progress, flush=True)
            print('WARM_COMPLETE', os.uname().nodename, path.name, name, total,
                  round(time.monotonic() - start, 3), 'seconds', flush=True)
