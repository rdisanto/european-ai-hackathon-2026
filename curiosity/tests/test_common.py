"""Model-independent preparation, submission and HTTP lifecycle regressions."""
import copy
import getpass
import hashlib
import importlib.util
import json
import os
import re
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
RELEASE = Path(os.environ.get('VLLM_TEST_RELEASE', str(REPO / 'serve'))).resolve()
MODELS = tuple(sorted(p.stem for p in (RELEASE / 'profiles').glob('*.json')))
sys.path.insert(0, str(RELEASE))
import inference
import node
import registry


def allocated_ports():
    source = (RELEASE / 'serve_model.sbatch').read_text()
    assignments = '\n'.join(re.findall(r'^export INFERENCE_(?:MASTER|API)_PORT=.*$', source, re.M))
    assert len(assignments.splitlines()) == 2
    script = 'for SLURM_JOB_ID in {0..9999}; do\n' + assignments + '\nprintf "%s %s\\n" "$INFERENCE_MASTER_PORT" "$INFERENCE_API_PORT"\ndone'
    return [tuple(map(int, line.split())) for line in subprocess.check_output(['bash', '-c', script]).decode().splitlines()]


class Framework(unittest.TestCase):
    def test_port_ranges_do_not_overlap_for_any_job_suffix(self):
        ports = allocated_ports()
        masters = {master for master, _ in ports}
        probes = {master + 1 for master in masters}
        apis = {api for _, api in ports}
        self.assertTrue(apis.isdisjoint(masters | probes))
        self.assertEqual(len(apis), 10000)
        self.assertTrue(all(1 <= p <= 65535 for p in masters | probes | apis))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.base = self.root / 'team'
        self.base.mkdir()
        (self.base / 'users').mkdir()
        self.bin = self.root / 'bin'
        self.bin.mkdir()
        self.home = self.root / 'home'
        self.home.mkdir()
        self.env = dict(os.environ, HOME=str(self.home), PATH=str(self.bin) + ':' + os.environ['PATH'])
        # cli.py runs remote commands through `bash -lc`, whose login profile resets PATH:
        # restore the stub directory there so Slurm and container stubs stay reachable.
        (self.home / '.bash_profile').write_text('export PATH=' + str(self.bin) + ':$PATH\n')
        self.command('singularity', '#!/bin/bash\nexit 0\n')
        self.command('sbatch', '#!/bin/bash\nprintf "12345\\n"\n')
        self.command('scontrol', '#!/bin/bash\nprintf "127.0.0.1\\n"\n')
        self.command('getent', '#!/bin/bash\nprintf "127.0.0.1 localhost\\n"\n')
        self.command('squeue', '#!/bin/bash\nprintf "2030-01-01T00:00:00\\n"\n')
        # The team folder as `cli.py init` creates it, then fake installs at the fixed paths.
        self.team('init', '--account', 'TEST_PROJECT')
        for runtime_id in inference.catalog()['runtimes']:
            runtime = self.base / 'runtimes' / runtime_id
            (runtime / 'python-overlay').mkdir(parents=True)
            for name in inference.catalog()['runtimes'][runtime_id]['overlay_required']:
                (runtime / 'python-overlay' / name).parent.mkdir(parents=True, exist_ok=True)
                (runtime / 'python-overlay' / name).write_text('# test fixture\n')
            for item in inference.catalog()['runtimes'][runtime_id].get('overlay_files', []):
                shutil.copyfile(str(RELEASE / item['source']), str(runtime / 'python-overlay' / item['target']))
            (runtime / 'image.sif').write_bytes(b'fake-sif')
            inference.save(runtime / 'runtime.json', {'id': runtime_id, 'sif': 'image.sif', 'sif_size': 8, 'overlay': 'python-overlay'})
        for model in MODELS:
            path = self.base / 'checkpoints' / inference.load(RELEASE / 'profiles' / (model + '.json'))['checkpoint']
            path.mkdir(parents=True, exist_ok=True)  # profiles can share a checkpoint
            for name in ('config.json', 'tokenizer_config.json'):
                (path / name).write_text('{}')
            profile = inference.load(RELEASE / 'profiles' / (model + '.json'))
            if profile.get('expected_shards') == 1:
                (path / 'model.safetensors').write_bytes(b'x')
                continue
            shards = ['shard-{}.safetensors'.format(i) for i in range(profile.get('expected_shards', 2))]
            for name in shards:
                (path / name).write_bytes(b'x')
            inference.save(path / 'model.safetensors.index.json', {'weight_map': {str(i): s for i, s in enumerate(shards)}})
        self.runs = self.base / 'users' / getpass.getuser() / 'runs'

    def tearDown(self):
        self.temp.cleanup()

    def team(self, *args, check=True):
        result = subprocess.run([sys.executable, str(RELEASE / 'team.py')] + list(args) + ['--base', str(self.base)],
                                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if check and result.returncode:
            raise AssertionError(result.stderr.decode())
        return result

    def command(self, name, source):
        path = self.bin / name
        path.write_text(source)
        path.chmod(0o700)

    def prepare(self, model):
        output = subprocess.check_output([sys.executable, str(RELEASE / 'inference.py'), 'prepare', model, '--base', str(self.base)], env=self.env, stderr=subprocess.PIPE)
        return Path(output.decode().strip())

    def test_private_pinned_runs_and_runtime_reuse(self):
        runs = [self.prepare(name) for name in MODELS]
        configs = [inference.load(p / 'RUN_CONFIG.json') for p in runs]
        backport = [c for c in configs if c['profile']['runtime'] == 'vllm-backport-sm80-0.13.1']
        self.assertEqual(backport[0]['sif'], backport[1]['sif'])
        self.assertEqual(backport[0]['overlay'], backport[1]['overlay'])
        self.assertNotEqual(backport[0]['cache_path'], backport[1]['cache_path'])
        keys = [(p / '.runtime/api_key').read_text() for p in runs]
        self.assertNotEqual(keys[0], keys[1])
        self.assertEqual(len(set(keys)), len(MODELS))
        self.assertEqual(len({cfg['cache_path'] for cfg in configs}), len(MODELS))
        for run in runs:
            for path in (run, run / '.runtime', run / '.runtime/api_key'):
                self.assertEqual(path.stat().st_mode & 0o077, 0)
            for name, digest in inference.load(run / 'RELEASE_MANIFEST.json')['files'].items():
                self.assertEqual(hashlib.sha256((run / name).read_bytes()).hexdigest(), digest)

    def test_deepseek16_preparation_checks_patch_and_keeps_the_old_identity(self):
        old = inference.load(self.prepare('deepseek41') / 'RUN_CONFIG.json')
        new = inference.load(self.prepare('deepseek41-16gpu') / 'RUN_CONFIG.json')
        self.assertEqual(old['model_path'], new['model_path'])
        self.assertNotEqual(old['runtime_path'], new['runtime_path'])
        self.assertNotEqual(old['cache_path'], new['cache_path'])
        self.assertNotEqual(old['profile']['served_model'], new['profile']['served_model'])
        command = inference.submit_command(new)
        self.assertIn('--nodes=4', command)
        self.assertIn('--gres=gpu:4', command)
        engine = node.engine_command(new, 0, 'head', 29001, 31001)
        self.assertEqual(engine[engine.index('--tensor-parallel-size') + 1], '8')
        self.assertEqual(engine[engine.index('--pipeline-parallel-size') + 1], '2')
        self.assertEqual(engine[engine.index('--engram-config') + 1], '{"cpu_offload":false}')
        (Path(new['overlay']) / 'pp_kv_guard.py').write_text('# tampered patch\n')
        result = subprocess.run([sys.executable, str(RELEASE / 'inference.py'), 'prepare', 'deepseek41-16gpu', '--base', str(self.base)],
                                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'Runtime patch checksum mismatch', result.stderr)
    def test_reject_bad_profiles_and_missing_assets(self):
        p = inference.load(RELEASE / 'profiles/glm53.json')
        for edit in ({'id': '../escape'}, {'engine_args': ['--api-key=secret']}, {'hooks': ['arbitrary_script']}):
            obj = copy.deepcopy(p)
            obj.update(edit)
            with self.assertRaises(ValueError):
                inference.profile_check(obj)
        # resources are now optional (stage/B300 profiles like glm53 omit them), so the
        # TP-mismatch check must run against a profile that still pins nodes x gpus_per_node.
        q = inference.load(RELEASE / 'profiles/deepseek41.json')
        q['parallelism']['tensor'] = 16  # exceeds deepseek41's 2 nodes x 4 GPUs
        with self.assertRaises(ValueError):
            inference.profile_check(q)
        (self.base / 'checkpoints/glm53/shard-0.safetensors').unlink()
        result = subprocess.run([sys.executable, str(RELEASE / 'inference.py'), 'prepare', 'glm53', '--base', str(self.base)], env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse(self.runs.exists() and list(self.runs.iterdir()))

    def test_submission_is_explicit_and_cannot_repeat(self):
        run = self.prepare('deepseek41')
        args = [sys.executable, str(run / 'inference.py'), 'submit']
        dry = subprocess.check_output(args + ['--dry-run'], cwd=str(run), env=self.env, stderr=subprocess.PIPE).decode()
        command = shlex.split(dry)
        for flag in ('--nodes=2', '--gres=gpu:4', '--mem=480GB', '--time=02:00:00', '--signal=B:TERM@90'):
            self.assertIn(flag, command)
        self.assertFalse((run / '.runtime/job_id').exists())
        subprocess.check_output(args, cwd=str(run), env=self.env, stderr=subprocess.PIPE)
        self.assertEqual((run / '.runtime/job_id').read_text().strip(), '12345')
        second = subprocess.run(args, cwd=str(run), env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(second.returncode, 0)

    def test_model_settings_survive_refactor(self):
        settings = [('glm53', '4', '1', []), ('deepseek41', '8', '1', ['deepseek_v41'])]
        if 'qwen235b' in MODELS:
            settings.append(('qwen235b', '4', '2', ['qwen3', 'hermes']))
        for model, tp, pp, parsers in settings:
            cfg = inference.load(self.prepare(model) / 'RUN_CONFIG.json')
            command = node.engine_command(cfg, 0, '10.0.0.1', 29500, 21001)
            self.assertEqual(command[command.index('--tensor-parallel-size') + 1], tp)
            self.assertEqual(command[command.index('--pipeline-parallel-size') + 1], pp)
            for value in parsers:
                self.assertIn(value, command)
            self.assertIn('--headless', node.engine_command(cfg, 1, '10.0.0.1', 29500, 21001))
            if model == 'glm53':
                # B300 docker profile: the glm parsers are resolved from the image's registry at
                # load time via parser_candidates (stock vLLM 0.32 renames them), not baked into
                # engine_args, so none may appear in the static command above.
                candidates = cfg['profile'].get('parser_candidates', {})
                self.assertTrue({'glm47', 'glm45'} & set(candidates.get('tool') or []))
                self.assertTrue({'glm47', 'glm45'} & set(candidates.get('reasoning') or []))
            if model == 'deepseek41':
                self.assertIn('{"cpu_offload":true}', command)
                self.assertIn('--enable-expert-parallel', command)
                self.assertNotIn('--enable-ep-weight-filter', command)

    def test_single_node_profile_and_runtime_without_overlay(self):
        cfg = inference.load(self.prepare('qwen06b') / 'RUN_CONFIG.json')
        command = node.engine_command(cfg, 0, '10.0.0.1', 29500, 31001)
        for flag in ('--nnodes', '--node-rank', '--master-addr', '--master-port', '--headless'):
            self.assertNotIn(flag, command)
        self.assertEqual(command[-4:], ['--host', '0.0.0.0', '--port', '31001'])
        # The backport runtime still insists on its TileLang overlay.
        (self.base / 'runtimes/vllm-backport-sm80-0.13.1/python-overlay/tilelang/__init__.py').unlink()
        with self.assertRaises(subprocess.CalledProcessError):
            self.prepare('deepseek41')

    def test_docker_launcher_builds_the_b300_recipe(self):
        run = self.prepare('glm53')
        cfg = inference.load(run / 'RUN_CONFIG.json')
        # A docker stub: daemon healthy, rootless, GPU probe fine -- enough to build the command.
        self.command('docker', '#!/bin/bash\n'
                     'if [[ "$*" == *"--format"* ]]; then printf \'{"SecurityOptions":["name=rootless"]}\'; fi\n'
                     'exit 0\n')
        engine = node.engine_command(cfg, 0, None, None, 21001)
        self.assertEqual(engine[engine.index('--tensor-parallel-size') + 1], '4')
        self.assertEqual(engine[engine.index('--max-model-len') + 1], '1048576')
        self.assertEqual(engine[engine.index('--max-num-seqs') + 1], '3')
        self.assertIn('{"method":"mtp","num_speculative_tokens":5}', engine)
        self.assertEqual(engine[-4:], ['--host', '0.0.0.0', '--port', '21001'])
        saved, node.ROOT = node.ROOT, run  # the api key lives in the run directory
        saved_path = os.environ['PATH']
        os.environ['PATH'] = str(self.bin) + ':' + saved_path  # node.docker_command runs in-process: expose the stub
        try:
            argv = node.docker_command(cfg, engine, api_port=21001, name='serve-test')
        finally:
            node.ROOT = saved
            os.environ['PATH'] = saved_path
        self.assertEqual(argv[:5], ['docker', 'run', '--rm', '--name', 'serve-test'])
        self.assertEqual(argv[argv.index('--gpus') + 1], 'all')
        self.assertIn(['--ulimit', 'memlock=-1'], [argv[i:i + 2] for i in range(len(argv) - 1)])
        self.assertIn('--shm-size=16g', argv)
        self.assertEqual(argv[argv.index('--publish') + 1], '21001:21001')
        self.assertEqual(argv[argv.index('--entrypoint') + 1], 'vllm')
        self.assertEqual(argv[argv.index('--entrypoint') + 2], cfg['image'])
        joined = ' '.join(argv)
        self.assertIn('--volume {}:{}:ro'.format(cfg['model_path'], cfg['model_path']), joined)
        self.assertIn('NCCL_ALGO=Ring', joined)
        self.assertIn('NCCL_PROTO=Simple', joined)
        self.assertIn('HF_HUB_OFFLINE=1', joined)
        key = (run / '.runtime/api_key').read_text().strip()
        self.assertIn('VLLM_API_KEY=' + key, joined)

    def test_docker_install_downloads_checkpoints_on_the_gpu_partition(self):
        self.command('sbatch', '#!/bin/bash\necho "$*" >> "$HOME/sbatch-calls"\nprintf "702\\n"\n')
        self.command('squeue', '#!/bin/bash\nexit 0\n')
        output = self.team('install', 'glm53', 'dsv4flashablit').stdout.decode()
        calls = (self.home / 'sbatch-calls').read_text().splitlines()
        self.assertEqual(len(calls), 2, output)
        for call in calls:
            self.assertIn('--job-name=install-checkpoint-', call)
            self.assertIn('--partition=hackathon', call)
            self.assertNotIn('install/runtime.sbatch', call)
            self.assertNotIn('--dependency', call)
        self.assertTrue(calls[0].endswith('install/docker_checkpoint.sbatch glm53'))
        self.assertTrue(calls[1].endswith('install/docker_checkpoint.sbatch dsv4flashablit'))
        (self.base / 'checkpoints/glm53/download_manifest.json').write_text('{}')
        self.assertIn('checkpoint glm53: installed', self.team('install', 'glm53').stdout.decode())

    def test_number_checks_accept_phrasing_but_not_wrong_answers(self):
        import test_endpoint
        for answer in ('42', '42.', 'The answer is 42.', '17 + 25 = 42', '42.0'):
            self.assertEqual(test_endpoint.final_number(answer), 42, answer)
        for answer in ('43', '', 'forty-two', '42 or 43'):
            self.assertNotEqual(test_endpoint.final_number(answer), 42, answer)

    def test_check_suites_and_runtime_catalog_are_enforced(self):
        p = inference.load(RELEASE / 'profiles/qwen06b.json')
        for edit in ({'checks': 'nonsense'}, {'runtime': 'not-in-catalog'}):
            obj = copy.deepcopy(p)
            obj.update(edit)
            with self.assertRaises(ValueError):
                inference.profile_check(obj)
        self.assertEqual(len(inference.check_suite(p)), 7)
        self.assertNotIn('forced_function_call', inference.CHECK_SUITES['chat'])

    def test_registry_claims_one_server_per_name(self):
        root = self.base / 'endpoints'  # created by init
        old_path = os.environ['PATH']
        os.environ['PATH'] = self.env['PATH']
        try:
            registry.claim(root, 'model-a', '100')
            registry.claim(root, 'model-a', '100')  # idempotent for the owner
            with self.assertRaises(ValueError):
                registry.claim(root, 'model-a', '200')  # squeue reports job 100 as active
            with self.assertRaises(ValueError):
                registry.publish(root, 'model-a', {'job_id': '200'}, '200')
            registry.publish(root, 'model-a', {'job_id': '100', 'host': 'n1', 'port': 1}, '100')
            self.assertEqual(registry.records(root)[0]['owner'], __import__('getpass').getuser())
            registry.withdraw(root, 'model-a', '200')  # not the owner: no effect
            self.assertTrue((root / 'model-a.json').exists())
            self.command('squeue', '#!/bin/bash\necho "slurm_load_jobs error: Invalid job id specified" >&2\nexit 1\n')
            registry.claim(root, 'model-a', '200')  # job 100 is gone: take over
            self.assertFalse((root / 'model-a.json').exists())
            self.command('squeue', '#!/bin/bash\necho "Socket timed out" >&2\nexit 1\n')
            with self.assertRaises(ValueError):
                registry.claim(root, 'model-a', '300')  # unknown state: never take over
            for bad in ('../x', '.hidden', 'a/b', ''):
                with self.assertRaises(ValueError):
                    registry.claim(root, bad, '1')
        finally:
            os.environ['PATH'] = old_path

    def test_up_status_logs_down(self):
        self.command('scancel', '#!/bin/bash\necho "$@" > "$HOME/scancel-args"\n')
        self.command('squeue', '#!/bin/bash\nprintf "12345|qwen06b-common|RUNNING|1:00|1:00:00|lrdn0001\\n"\n')
        output = self.team('up', 'qwen06b', '--qos', 'boost_qos_dbg', '--time', '00:30:00').stdout.decode()
        self.assertIn('Submitted job 12345', output)
        run = Path(output.split('Run directory: ')[1].split()[0])
        self.assertEqual(run.parent, self.runs)  # runs live in $BASE/users/<login>/runs, mode 700
        self.assertEqual(run.parent.parent.stat().st_mode & 0o077, 0)
        report = json.loads(self.team('status', '--json').stdout.decode())
        self.assertEqual([(s['profile'], s['job_id'], s['ready']) for s in report['servers']], [('qwen06b', '12345', False)])
        self.assertEqual(report['registry'], [])
        self.assertEqual(report['installed']['runtimes'], sorted(inference.catalog()['runtimes']))
        (run / 'qwen06b-common-12345.out').write_text('line 1\nINFERENCE_READY\n')
        for target in ('12345', 'qwen06b', 'qwen3-0.6b-curiosity'):
            self.assertIn('INFERENCE_READY', self.team('logs', target).stdout.decode())
        self.team('down', 'qwen3-0.6b-curiosity')
        self.assertEqual((self.home / 'scancel-args').read_text().split(), ['12345'])
        self.assertNotEqual(self.team('down', 'glm53', check=False).returncode, 0)

    def test_init_status_and_share(self):
        fresh = self.root / 'fresh'
        status = subprocess.run([sys.executable, str(RELEASE / 'team.py'), 'status', '--base', str(fresh)],
                                env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(status.returncode, 0)
        self.assertIn(b'cli.py init', status.stdout)
        self.assertIn(b'--force', self.team('init', '--account', 'OTHER', check=False).stderr)
        self.assertEqual(inference.load(self.base / 'inference-site.json')['account'], 'TEST_PROJECT')
        (self.base / 'users/someone-else').mkdir()
        (self.base / 'users/someone-else/secret').write_text('x')
        # A running install: a temporary root file system with links to system paths.
        (self.base / 'staging/vllm-0.31.0/tmp/rootfs').mkdir(parents=True)
        (self.base / 'staging/vllm-0.31.0/tmp/rootfs/lib').symlink_to('/usr/lib')
        (self.base / 'logs/linked.out').symlink_to('/etc/hostname')
        self.command('setfacl', '#!/bin/bash\necho "$*" >> "$HOME/setfacl-calls"\n')
        user = getpass.getuser()
        self.team('share', user)
        calls = (self.home / 'setfacl-calls').read_text().splitlines()
        access = [c for c in calls if c.startswith('-m u:{}:rwX '.format(user))]
        defaults = [c for c in calls if c.startswith('-d -m ')]
        self.assertTrue(access and defaults)
        everything = ' '.join(access + defaults)
        self.assertIn(str(self.base / 'checkpoints/qwen06b/config.json'), everything)
        self.assertIn(str(self.base / 'users') + ' ', ' '.join(access) + ' ')
        self.assertNotIn(str(self.base / 'users') + '/', everything)  # personal folders stay private
        self.assertNotIn('staging/vllm-0.31.0', everything)
        self.assertNotIn('linked.out', everything)
        self.assertIn(str(self.base / 'staging') + ' ', ' '.join(defaults) + ' ')
        self.assertNotIn(' ' + str(self.base / 'users') + ' ', ' ' + ' '.join(defaults) + ' ')
        self.assertIn(b'No such user', self.team('share', 'no-such-login-xyz', check=False).stderr)

    def test_install_submits_only_what_is_missing(self):
        self.command('sbatch', '#!/bin/bash\necho "$*" >> "$HOME/sbatch-calls"\nn=$(wc -l < "$HOME/sbatch-calls"); printf "%s\\n" "$((700 + n))"\n')
        self.command('squeue', '#!/bin/bash\nexit 0\n')
        shutil.rmtree(str(self.base / 'runtimes/vllm-0.31.0'))
        output = self.team('install', 'qwen06b', 'qwen35').stdout.decode()
        calls = (self.home / 'sbatch-calls').read_text().splitlines()
        # qwen235b's runtime is there; no checkpoint has a download manifest in the fixture.
        self.assertEqual(len(calls), 3, output)
        self.assertIn('--job-name=install-runtime-vllm-0.31.0', calls[0])
        self.assertIn('--partition=hackathon', calls[0])  # curiosity's cpu_partition (site.example.json)
        self.assertTrue(calls[0].endswith('install/runtime.sbatch vllm-0.31.0'))
        self.assertIn('--dependency=afterok:701', calls[1])  # qwen06b weights wait for its runtime
        self.assertNotIn('--dependency', calls[2])
        self.assertIn('CURIOSITY_RELEASE=' + str(REPO), calls[0])
        (self.base / 'checkpoints/qwen06b/download_manifest.json').write_text('{}')
        self.assertIn('checkpoint qwen06b: installed', self.team('install', 'qwen06b').stdout.decode())

    def test_diagnose_uses_the_right_queue(self):
        self.command('sbatch', '#!/bin/bash\necho "$*" >> "$HOME/sbatch-calls"\nprintf "800\\n"\n')
        self.team('diagnose', 'reach', '--target', 'lrdn0001:31234')
        self.team('diagnose', 'probe')
        reach, probe = (self.home / 'sbatch-calls').read_text().splitlines()
        self.assertIn('--partition=hackathon', reach)
        self.assertIn('TARGET=lrdn0001:31234', reach)
        self.assertIn('--partition=hackathon', probe)
        self.assertTrue(probe.endswith('diagnostics/probe.sbatch'))

    def test_admin_cli_over_ssh(self):
        # A stand-in ssh that runs the remote command locally, as the same (temporary) HOME.
        self.command('ssh', '#!/bin/bash\nwhile [[ "$1" == -* ]]; do [[ "$1" == -[oL] ]] && shift; shift; done\nshift\nexec bash -c "$*"\n')
        self.command('squeue', '#!/bin/bash\nprintf "12345|qwen06b-common|PENDING|0:00|1:00:00|\\n"\n')
        self.command('scancel', '#!/bin/bash\necho "$@" > "$HOME/scancel-args"\n')
        cli = [sys.executable, str(REPO / 'cli.py'), '--user', 'tester', '--base', str(self.base)]
        output = subprocess.check_output(cli + ['up', 'qwen06b', '--qos', 'boost_qos_dbg'], env=self.env).decode()
        self.assertIn('Submitted job 12345', output)
        releases = list((self.home / '.curiosity-inference/releases').iterdir())
        self.assertEqual(len(releases), 1)
        self.assertEqual(releases[0].stat().st_mode & 0o077, 0)
        for path in ('serve/team.py', 'install/runtime.sbatch', 'diagnostics/probe.sbatch'):
            self.assertTrue((releases[0] / path).is_file(), path)
        output = subprocess.check_output(cli + ['status'], env=self.env).decode()
        self.assertIn('qwen06b', output)
        self.assertIn('starting', output)
        subprocess.check_call(cli + ['down', 'qwen06b'], env=self.env, stdout=subprocess.DEVNULL)
        self.assertEqual((self.home / 'scancel-args').read_text().split(), ['12345'])
        token = subprocess.check_output(cli + ['router', 'token'], env=self.env, stderr=subprocess.DEVNULL).decode().strip()
        self.assertGreaterEqual(len(token), 32)
        self.assertEqual(subprocess.check_output(cli + ['router', 'token'], env=self.env).decode().strip(), token)
        self.assertEqual(len(list((self.home / '.curiosity-inference/releases').iterdir())), 1)
        self.assertEqual(subprocess.run(cli[:2], env=self.env, stdout=subprocess.PIPE).returncode, 0)  # bare call: help

    def test_forward_adds_and_removes_its_port_forward(self):
        # Control commands (-O) are logged; anything else runs locally as the remote command.
        self.command('ssh', '#!/bin/bash\nif [[ " $* " == *" -O "* ]]; then echo "$*" >> "$HOME/ssh-control"\n'
                     '  [[ " $* " == *" -O forward "* ]] && exit "${FORWARD_RC:-0}"\n'
                     '  [[ " $* " == *" -O check "* ]] && exit 1\n  exit 0; fi\n'
                     'while [[ "$1" == -* ]]; do [[ "$1" == -[oL] ]] && shift; shift; done\nshift\nexec bash -c "$*"\n')
        self.command('squeue', '#!/bin/bash\nprintf "900|curiosity-router|RUNNING|0:10|4:00:00|login08\\n"\n')
        endpoint = self.base / 'users' / getpass.getuser() / 'router/.runtime/endpoint.json'
        endpoint.parent.mkdir(parents=True)
        inference.save(endpoint, {'job_id': '900', 'host': 'login08', 'port': 41900, 'cluster_link': False})
        cli = [sys.executable, str(REPO / 'cli.py'), '--user', 'tester', '--base', str(self.base), 'forward', '--local-port', '18123']
        result = subprocess.run(cli, env=self.env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        calls = [line.split(' -O ')[1].split()[0] for line in (self.home / 'ssh-control').read_text().splitlines()]
        self.assertEqual(calls, ['cancel', 'forward', 'check', 'cancel'])
        self.assertIn('-L 127.0.0.1:18123:login08:41900', (self.home / 'ssh-control').read_text())
        busy = subprocess.run(cli, env=dict(self.env, FORWARD_RC='255'), stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
        self.assertNotEqual(busy.returncode, 0)
        self.assertIn(b'Could not listen on 127.0.0.1:18123', busy.stderr)

    def test_jobs_run_with_plain_sbatch_from_the_repository_root(self):
        # As Slurm would run them: no CURIOSITY_RELEASE, the submit directory is the repository root.
        env = dict(self.env, BASE=str(self.base), SLURM_JOB_ID='1', SLURM_SUBMIT_DIR=str(REPO.parent))
        env.pop('CURIOSITY_RELEASE', None)
        result = subprocess.run(['bash', str(REPO / 'install/runtime.sbatch'), 'vllm-0.31.0'], env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(b'already installed', result.stdout)
        result = subprocess.run(['bash', str(REPO / 'install/runtime.sbatch'), 'vllm-0.31.0'],
                                env=dict(env, SLURM_SUBMIT_DIR=str(self.root)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(b'repository root', result.stderr)
        # Each script names its job, queue and log, so `sbatch -A <account> <script>` is enough.
        for script in list((REPO / 'install').glob('*.sbatch')) + list((REPO / 'diagnostics').glob('*.sbatch')):
            header = script.read_text()
            for option in ('--job-name=', '--partition=', '--output='):
                self.assertIn('#SBATCH ' + option, header, script.name)

    def test_router_prepare_prints_the_sbatch_command(self):
        self.command('squeue', '#!/bin/bash\nexit 0\n')
        output = subprocess.check_output([sys.executable, str(RELEASE / 'router.py'), 'prepare', '--base', str(self.base),
                                          '--hours', '2'], env=self.env, stderr=subprocess.PIPE).decode()
        directory = self.base / 'users' / getpass.getuser() / 'router'
        self.assertTrue((directory / 'ROUTER.json').is_file() and (directory / 'router.sbatch').is_file())
        line = output.splitlines()[-1]
        self.assertTrue(line.startswith('cd {} && sbatch '.format(directory)), line)
        self.assertIn('--job-name=curiosity-router', shlex.split(line))
        self.assertFalse((self.home / 'sbatch-calls').exists())

    def test_failed_launcher_never_publishes_endpoint(self):
        self.command('srun', '#!/bin/bash\nexit 9\n')
        run = self.prepare('glm53')
        env = dict(self.env, SLURM_SUBMIT_DIR=str(run), SLURM_JOB_ID='12999', SLURM_JOB_NODELIST='node[1-2]', SLURM_CPUS_PER_TASK='32')
        result = subprocess.run(['bash', str(run / 'serve_model.sbatch')], cwd=str(run), env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
        self.assertNotEqual(result.returncode, 0)
        self.assertFalse((run / '.runtime/endpoint.json').exists())
        self.assertIn(b'launcher exited before readiness', result.stderr)

    def test_batch_readiness_english_checks_and_term_cleanup(self):
        fixture = REPO / 'tests/fake_api.py'
        self.command('srun', '#!/bin/bash\nexec ' + shutil.which('python3') + ' ' + str(fixture) + '\n')
        for model in MODELS:
            with self.subTest(model=model):
                run = self.prepare(model)
                # Pick an unused API port in the launcher's deterministic range.
                for job in range(12300, 12900):
                    probe = socket.socket()
                    try:
                        probe.bind(('127.0.0.1', allocated_ports()[job % 10000][1]))
                        break
                    except OSError:
                        pass
                    finally:
                        probe.close()
                env = dict(self.env, SLURM_SUBMIT_DIR=str(run), SLURM_JOB_ID=str(job), SLURM_JOB_NODELIST='node[1-2]', SLURM_CPUS_PER_TASK='32')
                with (run / 'batch-test.log').open('w') as log:
                    process = subprocess.Popen(['bash', str(run / 'serve_model.sbatch')], cwd=str(run), env=env, stdout=log, stderr=log)
                    try:
                        until = time.monotonic() + 25
                        while not (run / '.runtime/endpoint.json').exists() and process.poll() is None and time.monotonic() < until:
                            time.sleep(0.1)
                        self.assertTrue((run / '.runtime/endpoint.json').exists(), (run / 'batch-test.log').read_text())
                        profile = inference.load(run / 'RUN_CONFIG.json')['profile']
                        # This branch's lifecycle publishes on health alone (the API-check block
                        # in lifecycle.py is disabled); startup evidence must still be recorded.
                        startup = inference.load(run / ('evidence/startup-' + str(job) + '.json'))
                        self.assertIn('healthy_after_seconds', startup)
                        record = self.base / 'endpoints' / (profile['served_model'] + '.json')
                        self.assertEqual(inference.load(record)['job_id'], str(job))
                        self.assertNotIn((run / '.runtime/api_key').read_text().strip(), record.read_text())
                        process.send_signal(signal.SIGTERM)
                        self.assertEqual(process.wait(timeout=8), 143)
                        self.assertFalse((run / '.runtime/endpoint.json').exists())
                        self.assertEqual(list((self.base / 'endpoints').iterdir()), [])
                        pid = int((run / '.runtime/fake-server-pid').read_text())
                        with self.assertRaises(ProcessLookupError):
                            os.kill(pid, 0)
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            process.wait(timeout=8)


if __name__ == '__main__':
    unittest.main()
