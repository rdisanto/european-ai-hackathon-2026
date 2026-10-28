"""The router: one authenticated entry point for all of your model servers, run as its own Slurm job.

It finds your servers through their private run directories in $BASE/users/<login>/runs (never
through the shared registry, which teammates can write), routes each request by its "model" field
and swaps the caller's router token for that server's own key. Callers never see a server key.

Optionally (site setting cluster_link.enabled, off by default) it also opens the cluster link: a
chisel client that dials out to the chisel server in Kubernetes and offers this router there, so
agentgateway reaches it without anyone logging in to Curiosity from the cluster.

    python3 router.py prepare --base $WORK/<team> [--hours 12]   # then run the sbatch line it prints
    python3 router.py serve   # run by router.sbatch inside the job
    python3 router.py next    # run by router.sbatch: queue the successor job
team.py router up does prepare and submit in one step; team.py router down|token stop it and print its token.
"""
import argparse
import datetime
import getpass
import hashlib
import hmac
import http.client
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import secrets
import shlex
import shutil
import signal
from socketserver import ThreadingMixIn
import subprocess
import sys
import tempfile
import threading
import time

import inference
from inference import ROOT, load, save

JOB_NAME = 'curiosity-router'
ROUTER_DEFAULTS = {'qos': None, 'time': '04:00:00', 'cpus': 2, 'memory': '8000M'}
LINK_DEFAULTS = {'enabled': False, 'server': None, 'fingerprint': None, 'proxy': None, 'remote_port': 9000}
MAX_BODY = 16 * 1024 * 1024
# GLM-5.3 at 1M tokens: the first byte of a reply follows a prefill that can run for many minutes.
UPSTREAM_TIMEOUT = 3600
HOP_BY_HOP = {'connection', 'keep-alive', 'proxy-authenticate', 'proxy-authorization', 'te', 'trailer',
              'transfer-encoding', 'upgrade', 'server', 'date'}


def paths(base):
    """Where this account's router lives: code copy and job state, token, cluster link password."""
    mine = inference.user_dir(base)
    return {'directory': mine / 'router', 'token': mine / 'router_token', 'link_auth': mine / 'link_auth'}


def write_private(path, text):
    inference.private_dir(path.parent)
    temporary = path.with_name('.' + path.name + '.tmp')
    fd = os.open(str(temporary), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, 'w') as handle:
        handle.write(text)
    os.replace(str(temporary), str(path))


def read_private(path):
    mode = path.lstat()
    if mode.st_uid != os.getuid() or mode.st_mode & 0o077 or not path.is_file():
        raise ValueError('{} must be a private file owned by this account'.format(path))
    return path.read_text().strip()


def token(base, action):
    """ensure: create if missing; set: replace from stdin; show: print."""
    path = paths(base)['token']
    inference.private_dir(path.parent)
    if action == 'set':
        value = sys.stdin.readline().strip()
        if len(value) < 32:
            raise ValueError('Refusing a router token shorter than 32 characters')
        write_private(path, value + '\n')
        print('Router token replaced. A running router keeps the old one until restarted.', file=sys.stderr)
    elif action == 'ensure':
        if not path.exists():
            write_private(path, secrets.token_urlsafe(32) + '\n')
            print('Created a new router token in ' + str(path), file=sys.stderr)
    else:
        print(read_private(path))


def set_link_auth(base):
    value = sys.stdin.readline().strip()
    if ':' not in value:
        raise ValueError('Expected user:password for the cluster link')
    write_private(paths(base)['link_auth'], value + '\n')
    print('Cluster link password stored.', file=sys.stderr)


def settings(site):
    router, link = dict(ROUTER_DEFAULTS), dict(LINK_DEFAULTS)
    router.update(site.get('router', {}))
    link.update(site.get('cluster_link', {}))
    return router, link


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def job_command(cfg):
    router, site = cfg['router'], cfg['site']
    directory = Path(cfg['directory'])
    account = os.environ.get('SBATCH_ACCOUNT') or site['account']
    if account == 'YOUR_PROJECT':
        raise ValueError('Set a valid project allocation in the site configuration')
    command = ['sbatch', '--parsable', '--job-name=' + JOB_NAME, '--dependency=singleton',
               '--partition=' + inference.identifier(site.get('cpu_partition') or site.get('partition', 'lrd_all_serial')),
               '--nodes=1', '--ntasks=1', '--cpus-per-task=' + str(int(router['cpus'])),
               '--mem=' + router['memory'], '--time=' + inference.time_limit(router['time']),
               '--signal=B:TERM@60', '--output=%x-%j.out', '--chdir=' + str(directory)]
    # curiosity's hackathon partition has no project account; only add --account when one is set.
    if account:
        command.append('--account=' + inference.identifier(account))
    if router['qos']:
        command.append('--qos=' + inference.identifier(router['qos']))
    return command + [str(directory / 'router.sbatch')]


def seconds(limit):
    days, _, clock = limit.rpartition('-')
    h, m, s = map(int, clock.split(':'))
    return int(days or 0) * 86400 + h * 3600 + m * 60 + s


def prepare(base, site, hours, restart, jobs):
    """Create users/<login>/router/ with a code copy, ROUTER.json and the token; return its config."""
    manifest = inference.verify_release()
    router, link = settings(site)
    where = paths(base)
    current = state(base, jobs)
    if current and current['jobs']:
        if not restart:
            raise ValueError('A router is already running ({}); use --restart to replace it'.format(current['summary']))
        down()
    if link['enabled']:
        if not (link['server'] and link['fingerprint']):
            raise ValueError('cluster_link.enabled needs cluster_link.server and cluster_link.fingerprint in the site file')
        read_private(where['link_auth'])
    token(base, 'ensure')
    staging = Path(tempfile.mkdtemp(prefix='.router-', dir=str(where['directory'].parent)))
    for filename in manifest['files']:
        destination = staging / filename
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(str(ROOT / filename), str(destination))
    shutil.copyfile(str(ROOT / 'RELEASE_MANIFEST.json'), str(staging / 'RELEASE_MANIFEST.json'))
    inference.private_dir(staging / '.runtime')
    if link['enabled']:
        # Private, verified copy: the shared tools folder could be changed after this check.
        tool = inference.catalog()['tools']['chisel']
        source = base / 'tools' / ('chisel-' + tool['version'])
        if not source.is_file() or sha256(source) != tool['sha256']:
            shutil.rmtree(str(staging))
            raise ValueError('{} is missing or does not match catalog.json; run: cli.py install --link-client'.format(source))
        shutil.copyfile(str(source), str(staging / 'chisel'))
        (staging / 'chisel').chmod(0o700)
    cfg = {'schema_version': 1, 'base': str(base), 'site': site, 'router': router, 'cluster_link': link,
           'directory': str(where['directory']), 'release': manifest['release'], 'until': time.time() + hours * 3600}
    save(staging / 'ROUTER.json', cfg)
    if where['directory'].exists():
        shutil.rmtree(str(where['directory']))
    os.rename(str(staging), str(where['directory']))
    return cfg


def up(base, site, hours, restart, jobs):
    cfg = prepare(base, site, hours, restart, jobs)
    link = cfg['cluster_link']
    directory = Path(cfg['directory'])
    # SLURM_SUBMIT_DIR must be the router directory: router.sbatch starts with cd into it.
    job = inference.sbatch(job_command(cfg), directory / '.runtime/submitting', cwd=str(directory))
    print('Submitted router job {}; it resubmits itself until {}.'.format(
        job, datetime.datetime.fromtimestamp(cfg['until']).isoformat(timespec='minutes')))
    print('Cluster link: ' + ('ON to ' + link['server'] if link['enabled'] else 'off (site setting cluster_link.enabled)'))


def next_job():
    """Called at the start of each router job: queue the successor, which starts when this one ends."""
    cfg = load(Path('ROUTER.json'))
    job = os.environ['SLURM_JOB_ID']
    if time.time() + seconds(cfg['router']['time']) >= cfg['until']:
        print('Router chain ends with this job (until reached).')
        return
    directory = Path(cfg['directory'])
    successor = inference.sbatch(job_command(cfg), directory / ('.runtime/next-' + job), cwd=str(directory))
    print('Queued successor router job ' + successor)


def down():
    inference.ensure_slurm()
    subprocess.call(['scancel', '--name=' + JOB_NAME, '--user=' + getpass.getuser()])
    print('Cancelled all {} jobs of {}.'.format(JOB_NAME, getpass.getuser()))


def state(base, jobs):
    """Your router jobs and listening address, or None when there is no trace of one."""
    mine = [j for j in (jobs or {}).values() if j['name'] == JOB_NAME]
    endpoint = None
    path = paths(base)['directory'] / '.runtime/endpoint.json'
    if path.exists():
        endpoint = load(path)
        if jobs is not None and endpoint['job_id'] not in jobs:
            endpoint = None
    if not mine and not endpoint:
        return None
    parts = ['{} {}'.format(j['job_id'], j['state']) for j in mine]
    if endpoint:
        parts.append('listening on {}:{}, cluster link {}'.format(endpoint['host'], endpoint['port'],
                                                                 'on' if endpoint['cluster_link'] else 'off'))
    return {'jobs': mine, 'endpoint': endpoint, 'summary': '; '.join(parts)}


def own_runs(base):
    """This account's run directories, newest first, with job id and endpoint when known."""
    runs = []
    for path in (inference.user_dir(base) / 'runs').glob('*'):
        try:
            mode = path.lstat()
            if not path.is_dir() or path.is_symlink() or mode.st_uid != os.getuid() or mode.st_mode & 0o077:
                continue
            cfg = load(path / 'RUN_CONFIG.json')
            job = (path / '.runtime/job_id').read_text().strip() if (path / '.runtime/job_id').exists() else None
            endpoint = load(path / '.runtime/endpoint.json') if (path / '.runtime/endpoint.json').exists() else None
        except (OSError, ValueError, KeyError):
            continue
        runs.append({'run': path, 'profile': cfg['profile'], 'job': job, 'endpoint': endpoint, 'mtime': mode.st_mtime})
    return sorted(runs, key=lambda r: -r['mtime'])


def active_jobs():
    """Own Slurm jobs by id, or None if squeue is unavailable."""
    try:
        inference.ensure_slurm()
        output = subprocess.check_output(['squeue', '-h', '-u', getpass.getuser(), '-o', '%i|%j|%T|%M|%l|%N'],
                                         stderr=subprocess.DEVNULL, timeout=30).decode()
    except (OSError, subprocess.SubprocessError):
        return None
    jobs = {}
    for line in output.splitlines():
        fields = line.strip().split('|')
        if len(fields) == 6:
            jobs[fields[0]] = dict(zip(('job_id', 'name', 'state', 'elapsed', 'limit', 'nodes'), fields))
    return jobs


class Routes:
    """Model name -> your newest ready server, rescanned at most every 10 seconds."""

    def __init__(self, base):
        self.base = base
        self.lock = threading.Lock()
        self.table, self.scanned, self.jobs, self.jobs_at = {}, 0, None, 0

    def current(self):
        with self.lock:
            now = time.monotonic()
            if now - self.jobs_at > 30:
                self.jobs, self.jobs_at = active_jobs(), now
            if now - self.scanned > 10:
                table = {}
                for run in reversed(own_runs(self.base)):  # oldest first, so newer runs win
                    endpoint = run['endpoint']
                    if not endpoint:
                        continue
                    if not endpoint.get('stage') and (self.jobs is not None and endpoint['job_id'] not in self.jobs):
                        continue  # a Slurm server no longer on squeue; but stage runs stay until their supervisor withdraws them
                    try:
                        key = read_private(run['run'] / '.runtime/api_key')
                    except (OSError, ValueError):
                        continue
                    table[endpoint['model']] = {'host': endpoint['host'], 'port': int(endpoint['port']),
                                                'key': key, 'job_id': endpoint['job_id']}
                self.table, self.scanned = table, now
            return self.table


class Server(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def handler(routes, router_token):
    expected = router_token.encode()

    class Handler(BaseHTTPRequestHandler):
        server_version = 'curiosity-router'

        def log_message(self, *args):
            pass

        def log(self, status, model=None, started=None):
            print('{} {} {} {} {}'.format(self.command, self.path, model or '-', status,
                                          '{:.1f}s'.format(time.monotonic() - started) if started else ''), flush=True)

        def reply(self, status, obj):
            body = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            self.log(status)

        def error(self, status, message, code):
            self.reply(status, {'error': {'message': message, 'type': 'invalid_request_error', 'code': code}})

        def authorized(self):
            header = self.headers.get('Authorization', '')
            if header.startswith('Bearer ') and hmac.compare_digest(header[7:].strip().encode(), expected):
                return True
            self.error(401, 'Missing or wrong router token', 'unauthorized')
            return False

        def do_GET(self):
            if self.path == '/health':
                return self.reply(200, {'status': 'ok'})
            if not self.authorized():
                return
            if self.path in ('/v1/models', '/models'):
                models = sorted(routes.current())
                return self.reply(200, {'object': 'list', 'data': [
                    {'id': m, 'object': 'model', 'owned_by': 'curiosity'} for m in models]})
            self.error(404, 'Unknown path', 'not_found')

        def body(self):
            if self.headers.get('Transfer-Encoding', '').lower() == 'chunked':
                parts, total = [], 0
                while True:
                    size = int(self.rfile.readline().split(b';')[0], 16)
                    if size == 0:
                        while self.rfile.readline() not in (b'\r\n', b'\n', b''):
                            pass
                        return b''.join(parts)
                    total += size
                    if total > MAX_BODY:
                        return None
                    parts.append(self.rfile.read(size))
                    self.rfile.readline()
            length = int(self.headers.get('Content-Length') or 0)
            return None if length > MAX_BODY else self.rfile.read(length)

        def do_POST(self):
            if not self.authorized():
                return
            if not self.path.startswith('/') or '..' in self.path:
                return self.error(400, 'Invalid path', 'invalid_path')
            data = self.body()
            if data is None:
                return self.error(413, 'Request body too large', 'too_large')
            try:
                model = json.loads(data.decode()).get('model')
            except (ValueError, AttributeError):
                return self.error(400, 'Body must be a JSON object with a "model" field', 'invalid_json')
            route = routes.current().get(model)
            if route is None:
                return self.error(404, 'Model {} is not running on Curiosity right now'.format(model), 'model_not_found')
            self.forward(route, model, data)

        def forward(self, route, model, data):
            started = time.monotonic()
            connection = http.client.HTTPConnection(route['host'], route['port'], timeout=UPSTREAM_TIMEOUT)
            headers = {'Content-Type': self.headers.get('Content-Type', 'application/json'),
                       'Accept': self.headers.get('Accept', '*/*'), 'Authorization': 'Bearer ' + route['key']}
            try:
                connection.request('POST', self.path, data, headers)
                response = connection.getresponse()
            except (OSError, http.client.HTTPException) as error:
                connection.close()
                return self.error(502, 'Curiosity server for {} unreachable: {}'.format(model, type(error).__name__), 'upstream_unreachable')
            try:
                self.send_response(response.status)
                for name, value in response.getheaders():
                    if name.lower() not in HOP_BY_HOP:
                        self.send_header(name, value)
                self.send_header('Connection', 'close')
                self.end_headers()
                # read1 returns as soon as data arrives, so server-sent events stream through.
                while True:
                    chunk = response.read1(65536)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                    self.wfile.flush()
            except (OSError, http.client.HTTPException):
                pass  # the caller or the server went away mid-stream
            finally:
                connection.close()
                self.log(response.status, model, started)

    return Handler


def link_command(cfg, port):
    link = cfg['cluster_link']
    command = [str(Path(cfg['directory']) / 'chisel'), 'client', '--fingerprint', link['fingerprint'], '--keepalive', '25s']
    if link['proxy']:
        command += ['--proxy', link['proxy']]
    return command + [link['server'], 'R:0.0.0.0:{}:127.0.0.1:{}'.format(int(link['remote_port']), port)]


def serve():
    os.umask(0o077)
    cfg = load(Path('ROUTER.json'))
    base = Path(cfg['base'])
    port = int(os.environ['ROUTER_PORT'])
    server = Server(('0.0.0.0', port), handler(Routes(base), read_private(paths(base)['token'])))
    link = None
    if cfg['cluster_link']['enabled']:
        # chisel reads AUTH from the environment, which keeps the password out of the process list.
        environment = dict(os.environ, AUTH=read_private(paths(base)['link_auth']))
        link = subprocess.Popen(link_command(cfg, port), env=environment)
    endpoint = {'job_id': os.environ.get('SLURM_JOB_ID', 'local'), 'host': os.uname().nodename.split('.')[0],
                'port': port, 'cluster_link': link is not None, 'release': cfg['release'],
                'started': datetime.datetime.now().astimezone().isoformat()}
    save(Path('.runtime/endpoint.json'), endpoint)
    print('ROUTER_READY', json.dumps(endpoint), flush=True)

    def stop(signum, frame):
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
    finally:
        server.server_close()
        if link is not None and link.poll() is None:
            link.terminate()
            link.wait()
        path = Path('.runtime/endpoint.json')
        if path.exists() and load(path)['job_id'] == endpoint['job_id']:
            path.unlink()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=['prepare', 'serve', 'next'])
    parser.add_argument('--base', help='prepare: the team folder')
    parser.add_argument('--hours', type=float, default=12, help='prepare: keep resubmitting for this long')
    parser.add_argument('--restart', action='store_true', help='prepare: cancel a running router first')
    args = parser.parse_args()
    if args.command == 'serve':
        serve()
    elif args.command == 'next':
        next_job()
    else:
        os.umask(0o077)
        base = inference.site_base(args.base)
        cfg = prepare(base, inference.load_site(base), args.hours, args.restart, active_jobs())
        print('Router prepared in {}. Submit it from there:'.format(cfg['directory']))
        print('cd {} && {}'.format(shlex.quote(cfg['directory']), ' '.join(shlex.quote(w) for w in job_command(cfg))))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as error:
        raise SystemExit(str(error))
