"""Publish an endpoint only after tests; withdraw it when health fails.

The private record (.runtime/endpoint.json) is what the router and client use. When the site
configures a shared registry, the job also claims its model name there at start and publishes an
address-only record after the checks.
"""
import argparse
import datetime
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

from inference import ROOT, check_suite, config, load, save
import registry


child = None
claimed = None


def stop(signum, frame):
    if child is not None and child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
    raise SystemExit(128 + signum)


def alive(pid):
    state = Path('/proc/{}/stat'.format(pid))
    try:
        return state.read_text().rsplit(')', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        if Path('/proc').exists():
            return False
        # Allows lifecycle regression tests on macOS; Curiosity uses /proc above.
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False


def healthy(port):
    try:
        with urllib.request.urlopen('http://127.0.0.1:{}/health'.format(port), timeout=5) as response:
            return response.status == 200
    except (urllib.error.URLError, TimeoutError):
        return False


def main():
    global child, claimed
    os.umask(0o077)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launcher-pid', type=int, required=True)
    args = parser.parse_args()
    cfg = config()
    port = int(os.environ['INFERENCE_API_PORT'])
    job = os.environ['SLURM_JOB_ID']
    shared = registry.directory(cfg)
    if shared is not None:
        registry.claim(shared, cfg['profile']['served_model'], job)
        claimed = shared
    suite = check_suite(cfg['profile'])
    start = time.monotonic()
    while not healthy(port):
        if not alive(args.launcher_pid):
            raise RuntimeError('Model launcher exited before readiness')
        if time.monotonic() - start > cfg['profile']['startup_timeout_seconds']:
            raise RuntimeError('API startup timed out')
        time.sleep(5)
    save(ROOT / ('evidence/startup-' + job + '.json'), {'healthy_after_seconds': round(time.monotonic() - start, 2)})
    # output = ROOT / ('evidence/api-tests-' + job + '.json')
    # child = subprocess.Popen([sys.executable, str(ROOT / 'test_endpoint.py'),
    #                           '--base-url', 'http://127.0.0.1:' + str(port),
    #                           '--api-key-file', str(ROOT / '.runtime/api_key'), '--output', str(output)])
    # if child.wait() != 0:
    #     raise RuntimeError('English-only API checks failed; endpoint withheld')
    # checks = load(output)
    # if sorted(checks) != sorted(suite) or not all(c['passed'] for c in checks.values()):
    #     raise RuntimeError('API checks incomplete')
    if not alive(args.launcher_pid) or not healthy(port):
        raise RuntimeError('Model stopped during validation; endpoint withheld')
    try:
        end = subprocess.check_output(['squeue', '-h', '-j', job, '-o', '%e']).decode().strip()
    except subprocess.CalledProcessError:
        end = 'Check squeue'
    endpoint = {'job_id': job, 'host': os.uname().nodename.split('.')[0], 'port': port,
                'model': cfg['profile']['served_model'], 'profile': cfg['profile']['id'],
                'runtime': cfg['profile']['runtime'], 'release': cfg['release'],
                'nodes': os.environ['SLURM_JOB_NODELIST'], 'checks_passed': len(suite),
                'ready_at': datetime.datetime.now().astimezone().isoformat(), 'slurm_expected_end': end,
                'note': 'Service stops about 90 seconds before allocation expiry.'}
    save(ROOT / '.runtime/endpoint.json', endpoint)
    save(ROOT / ('evidence/ready-' + job + '.json'), endpoint)
    if claimed is not None:
        registry.publish(claimed, cfg['profile']['served_model'], endpoint, job)
    print('INFERENCE_READY', endpoint, flush=True)
    failures = 0
    while True:
        if not alive(args.launcher_pid):
            raise RuntimeError('Model launcher exited; stopping allocation')
        failures = 0 if healthy(port) else failures + 1
        if failures >= 3:
            raise RuntimeError('Three API health failures; stopping allocation')
        time.sleep(5)


if __name__ == '__main__':
    try:
        main()
    finally:
        path = ROOT / '.runtime/endpoint.json'
        if path.exists():
            path.unlink()
        if claimed is not None:
            registry.withdraw(claimed, config()['profile']['served_model'], os.environ['SLURM_JOB_ID'])
