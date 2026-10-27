"""Router routing, authentication, streaming and job chaining, with fake servers and Slurm."""
import getpass
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

REPO = Path(__file__).resolve().parents[1]
RELEASE = REPO / 'serve'
TOKEN = 'router-token-for-tests-0123456789abcdef'


def free_port():
    with socket.socket() as probe:
        probe.bind(('127.0.0.1', 0))
        return probe.getsockname()[1]


class Backend(ThreadingMixIn, HTTPServer):
    """Stands in for one vLLM server: echoes what reached it, streams slowly when asked."""
    daemon_threads = True

    def __init__(self, name):
        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if body.get('stream'):
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.end_headers()
                    self.wfile.write(b'data: {"first": true}\n\n')
                    self.wfile.flush()
                    time.sleep(0.8)
                    self.wfile.write(b'data: [DONE]\n\n')
                    return
                reply = json.dumps({'backend': name, 'auth': self.headers.get('Authorization'),
                                    'path': self.path, 'model': body['model']}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(reply)))
                self.end_headers()
                self.wfile.write(reply)

        super().__init__(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()


class Router(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.home, self.bin, self.base = root / 'home', root / 'bin', root / 'team'
        for path in (self.home, self.bin):
            path.mkdir()
        self.mine = self.base / 'users' / getpass.getuser()
        self.directory = self.mine / 'router'
        (self.directory / '.runtime').mkdir(parents=True)
        self.mine.chmod(0o700)
        (self.mine / 'router_token').write_text(TOKEN + '\n')
        (self.mine / 'router_token').chmod(0o600)
        self.env = dict(os.environ, HOME=str(self.home), PATH=str(self.bin) + ':' + os.environ['PATH'])
        self.command('squeue', '#!/bin/bash\nprintf "101|a|RUNNING|1:00|1:00:00|n1\\n102|b|RUNNING|1:00|1:00:00|n2\\n"\n')
        self.backends = {name: Backend(name) for name in ('model-a', 'model-b', 'model-c', 'model-d')}
        for job, name in enumerate(sorted(self.backends), start=101):
            run = self.mine / 'runs' / (name + '-x')
            (run / '.runtime').mkdir(parents=True)
            run.chmod(0o755 if name == 'model-d' else 0o700)  # model-d: not private, never routed
            (run / 'RUN_CONFIG.json').write_text(json.dumps({'profile': {'id': name, 'served_model': name}}))
            (run / '.runtime/job_id').write_text(str(job))
            (run / '.runtime/api_key').write_text('key-' + name + '\n')
            (run / '.runtime/api_key').chmod(0o600)
            (run / '.runtime/endpoint.json').write_text(json.dumps({
                'job_id': str(job), 'host': '127.0.0.1', 'port': self.backends[name].server_port, 'model': name}))
        # model-c (job 103) is not in squeue: a dead job's endpoint must not be routed.
        (self.directory / 'ROUTER.json').write_text(json.dumps(
            {'base': str(self.base), 'cluster_link': {'enabled': False}, 'release': 'test', 'directory': str(self.directory)}))
        self.port = free_port()
        self.process = subprocess.Popen([sys.executable, str(RELEASE / 'router.py'), 'serve'], cwd=str(self.directory),
                                        env=dict(self.env, ROUTER_PORT=str(self.port), SLURM_JOB_ID='900'),
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        until = time.monotonic() + 10
        while not (self.directory / '.runtime/endpoint.json').exists():
            self.assertLess(time.monotonic(), until, self.process.stdout.read1(10000) if self.process.poll() is not None else 'no endpoint')
            time.sleep(0.05)

    def tearDown(self):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.process.stdout.close()
        for backend in self.backends.values():
            backend.shutdown()
            backend.server_close()
        self.temp.cleanup()

    def command(self, name, source):
        path = self.bin / name
        path.write_text(source)
        path.chmod(0o700)

    def call(self, method, path, body=None, token=TOKEN, **kwargs):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        headers = {'Content-Type': 'application/json'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        connection.request(method, path, None if body is None else json.dumps(body).encode(), headers, **kwargs)
        response = connection.getresponse()
        return response, connection

    def json_call(self, method, path, body=None, token=TOKEN):
        response, connection = self.call(method, path, body, token)
        try:
            return response.status, json.loads(response.read())
        finally:
            connection.close()

    def test_authentication_and_catalog(self):
        self.assertEqual(self.json_call('GET', '/health', token=None), (200, {'status': 'ok'}))
        self.assertEqual(self.json_call('GET', '/v1/models', token=None)[0], 401)
        self.assertEqual(self.json_call('GET', '/v1/models', token='wrong')[0], 401)
        self.assertEqual(self.json_call('POST', '/v1/chat/completions', {'model': 'model-a'}, token='wrong')[0], 401)
        status, catalog = self.json_call('GET', '/v1/models')
        self.assertEqual(status, 200)
        self.assertEqual([m['id'] for m in catalog['data']], ['model-a', 'model-b'])

    def test_routes_by_model_and_swaps_the_key(self):
        for model in ('model-a', 'model-b'):
            for path in ('/v1/chat/completions', '/classify'):
                status, reply = self.json_call('POST', path, {'model': model, 'messages': []})
                self.assertEqual(status, 200)
                self.assertEqual(reply, {'backend': model, 'auth': 'Bearer key-' + model, 'path': path, 'model': model})
        for model in ('model-c', 'model-d', 'unknown'):
            status, reply = self.json_call('POST', '/v1/chat/completions', {'model': model})
            self.assertEqual((status, reply['error']['code']), (404, 'model_not_found'))
        self.assertEqual(self.json_call('POST', '/v1/chat/completions', ['not', 'an', 'object'])[0], 400)

    def test_chunked_request_body(self):
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        body = iter([b'{"model": ', b'"model-b"}'])
        connection.request('POST', '/v1/completions', body, {'Authorization': 'Bearer ' + TOKEN,
                                                             'Content-Type': 'application/json'}, encode_chunked=True)
        self.assertEqual(json.loads(connection.getresponse().read())['backend'], 'model-b')
        connection.close()

    def test_streams_without_buffering(self):
        start = time.monotonic()
        response, connection = self.call('POST', '/v1/chat/completions', {'model': 'model-a', 'stream': True})
        self.assertEqual(response.headers['Content-Type'], 'text/event-stream')
        first = response.read1(1000)
        first_at = time.monotonic() - start
        rest = response.read()
        connection.close()
        self.assertIn(b'"first"', first)
        self.assertIn(b'[DONE]', rest)
        self.assertLess(first_at, 0.6, 'first event was held back until the stream ended')
        self.assertGreater(time.monotonic() - start, 0.75)

    def test_unreachable_server_is_a_502(self):
        self.backends['model-b'].shutdown()
        self.backends['model-b'].server_close()
        status, reply = self.json_call('POST', '/v1/chat/completions', {'model': 'model-b'})
        self.assertEqual((status, reply['error']['code']), (502, 'upstream_unreachable'))
        self.assertNotIn('key-model-b', json.dumps(reply))

    def test_endpoint_withdrawn_on_stop(self):
        self.process.terminate()
        self.process.wait(timeout=5)
        self.assertFalse((self.directory / '.runtime/endpoint.json').exists())


class RouterJobs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.home, self.bin, self.base = root / 'home', root / 'bin', root / 'team'
        for path in (self.home, self.bin, self.base):
            path.mkdir()
        self.env = dict(os.environ, HOME=str(self.home), PATH=str(self.bin) + ':' + os.environ['PATH'])
        self.team('init', '--account', 'TEST_PROJECT')
        self.site = json.loads((self.base / 'inference-site.json').read_text())
        self.mine = self.base / 'users' / getpass.getuser()
        self.command('squeue', '#!/bin/bash\nexit 0\n')
        self.command('sbatch', '#!/bin/bash\nprintf "%s\\n" "$PWD" "$@" >> "$HOME/sbatch-calls"\nprintf "777\\n"\n')

    def tearDown(self):
        self.temp.cleanup()

    def command(self, name, source):
        path = self.bin / name
        path.write_text(source)
        path.chmod(0o700)

    def team(self, *args):
        return subprocess.run([sys.executable, str(RELEASE / 'team.py')] + list(args) + ['--base', str(self.base)],
                              env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def next_job(self, job):
        directory = self.mine / 'router'
        return subprocess.run([sys.executable, str(directory / 'router.py'), 'next'], cwd=str(directory),
                              env=dict(self.env, SLURM_JOB_ID=job), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_up_chains_and_link_switch(self):
        result = self.team('router', 'up', '--hours', '9')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'Cluster link: off', result.stdout)
        directory = self.mine / 'router'
        calls = (self.home / 'sbatch-calls').read_text().split('\n')
        self.assertEqual(calls[0], str(directory))
        for flag in ('--job-name=curiosity-router', '--dependency=singleton', '--partition=hackathon',
                     '--time=04:00:00', '--chdir=' + str(directory)):
            self.assertIn(flag, calls)
        self.assertFalse(any(c.startswith('--qos') for c in calls))
        self.assertEqual((self.mine / 'router_token').stat().st_mode & 0o077, 0)
        self.assertFalse((directory / 'chisel').exists())

        # Inside the job: 9 h until > 4 h time limit, so a successor is queued exactly once.
        self.assertEqual(self.next_job('777').returncode, 0)
        self.assertEqual((self.home / 'sbatch-calls').read_text().count('--dependency=singleton'), 2)
        cfg = json.loads((directory / 'ROUTER.json').read_text())
        cfg['until'] = time.time() + 3600
        (directory / 'ROUTER.json').write_text(json.dumps(cfg))
        self.assertIn(b'chain ends', self.next_job('778').stdout)
        self.assertEqual((self.home / 'sbatch-calls').read_text().count('--dependency=singleton'), 2)

        # A running router is not replaced silently.
        self.command('squeue', '#!/bin/bash\nprintf "777|curiosity-router|RUNNING|1:00|4:00:00|n1\\n"\n')
        result = self.team('router', 'up')
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'--restart', result.stderr)

    def test_cluster_link_needs_settings_password_and_verified_chisel(self):
        self.site['cluster_link'].update(enabled=True, server='https://example.invalid/curiosity-link', fingerprint='abc')
        (self.base / 'inference-site.json').write_text(json.dumps(self.site))
        self.assertIn(b'link_auth', self.team('router', 'up').stderr)
        result = subprocess.run([sys.executable, str(RELEASE / 'team.py'), 'router', 'link-auth', '--base', str(self.base)],
                                input=b'curiosity:secret\n', env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual((self.mine / 'link_auth').stat().st_mode & 0o077, 0)
        (self.base / 'tools/chisel-1.13.0').write_bytes(b'not the pinned binary')
        self.assertIn(b'does not match catalog.json', self.team('router', 'up').stderr)
        self.assertFalse((self.home / 'sbatch-calls').exists())


if __name__ == '__main__':
    unittest.main()
