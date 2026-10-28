"""Stage controller: one reservation, in-node docker lifecycle + endpoointing (no real GPUs)."""
import json
import os
import sys
import unittest
import unittest.mock as mock
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # import test_common in any invocation mode
import test_common
from test_common import Framework
RELEASE = test_common.RELEASE
sys.path.insert(0, str(RELEASE))
import inference
import node
import stage


class Stage(Framework):
    def docker_stub(self):
        # Fake rootless docker: probing commands return instantly, a "run" stays alive,
        # and a nvidia-smi capability probe is a success.
        self.command('docker', '#!/bin/bash\n'
                     'if [[ "${1:-}" == info ]]; then\n'
                     '  if [[ "$*" == *--format* ]]; then printf \'{"SecurityOptions":["name=rootless"]}\'; fi\n'
                     '  exit 0\n'
                     'fi\n'
                     'if [[ "${1:-}" == rm || "$*" == *nvidia-smi* ]]; then exit 0; fi\n'
                     'exec sleep 1000\n')

    def test_load_assigns_slice_and_port_and_unload_frees_them(self):
        self.docker_stub()
        saved_path = os.environ['PATH']
        os.environ['PATH'] = str(self.bin) + ':' + saved_path  # in-process: expose docker/sbatch stubs
        self.addCleanup(lambda: os.environ.__setitem__('PATH', saved_path))
        base = self.base
        try:
            with mock.patch.object(node, 'verify_image', return_value=[]), \
                    mock.patch.object(stage, 'request_healthy', return_value=True):
                records = stage.load_model(str(base), 'glm53', {})
            self.assertIn('glm53', records)
            rec = records['glm53']
            self.assertEqual(rec['slice'], '0,1,2,3')           # first free 4-GPU block
            self.assertEqual(rec['api_port'], stage.API_PORT_BASE)
            ep = inference.load(Path(rec['run']) / '.runtime/endpoint.json')
            self.assertTrue(ep['stage'])
            self.assertEqual(ep['model'], 'glm-5.3')
            self.assertEqual(ep['gpu_slice'], '0,1,2,3')
            # A second, differently sized model must take the other free 4-GPU slice.
            with mock.patch.object(node, 'verify_image', return_value=[]), \
                    mock.patch.object(stage, 'request_healthy', return_value=True):
                records = stage.load_model(str(base), 'dsv4flashablit', records)
            self.assertEqual(records['glm53']['slice'], '0,1,2,3')
            self.assertEqual(records['dsv4flashablit']['slice'], '4,5,6,7')
            self.assertEqual(records['dsv4flashablit']['api_port'], stage.API_PORT_BASE + 10)
            # A third model has no free 4-GPU slice on an 8-GPU node.
            self.assertIsNone(stage.free_gpu_slice(records))
            # Unload removes the container record and the endpoint, freeing the slice.
            stage.unload_model(str(base), 'glm53', records)
            self.assertNotIn('glm53', stage.load_models(str(base)))
            self.assertIsNotNone(stage.free_gpu_slice(records))
        finally:
            pass

    def test_status_json_exposes_router_endpoint(self):
        base = self.base
        d = stage.ensure_dirs(base)
        inference.save(d / 'status.json', {'supervisor': 'running', 'router_port': stage.STAGE_PORT, 'node': 'r3c1'})
        import subprocess as sp
        out = sp.check_output([sys.executable, str(RELEASE / 'team.py'), 'stage', 'status', '--json', '--base',
                               str(base)], env=self.env, stderr=sp.PIPE)
        data = json.loads(out)
        self.assertEqual(data['router'], {'host': 'r3c1', 'port': stage.STAGE_PORT})
        # without a supervisor report there is no routable endpoint (forward would refuse).
        d2 = stage.stage_dir(base)
        (d2 / 'status.json').unlink()
        data2 = json.loads(sp.check_output([sys.executable, str(RELEASE / 'team.py'), 'stage', 'status', '--json',
                                            '--base', str(base)], env=self.env))
        self.assertIsNone(data2['router'])

    def test_reserve_reattaches_when_a_stage_already_runs(self):
        base = self.base
        import subprocess as sp
        self.command('sbatch', '#!/bin/bash\necho "$*" >> "$HOME/sbatch-calls"\nprintf "700\\n"\n')
        self.command('squeue', '#!/bin/bash\nprintf "600|stage|RUNNING|0:10|11:00:00|r3c1\\n"\n')
        self.command('scancel', '#!/bin/bash\necho "$@" >> "$HOME/scancel-args"\n')
        # With a stage already running, reserve reattaches: no second submission, nothing cancelled.
        reattach = sp.run([sys.executable, str(RELEASE / 'team.py'), 'stage', 'reserve', '--base', str(base)],
                          env=self.env, capture_output=True)
        self.assertEqual(reattach.returncode, 0, reattach.stderr)
        self.assertIn(b'already reserved', reattach.stdout)
        self.assertFalse((self.home / 'sbatch-calls').exists())
        self.assertFalse((self.home / 'scancel-args').exists())
        # --restart replaces it on purpose: cancels the existing stage and submits a fresh one.
        replaced = sp.run([sys.executable, str(RELEASE / 'team.py'), 'stage', 'reserve', '--restart', '--base',
                           str(base)], env=self.env, capture_output=True)
        self.assertEqual(replaced.returncode, 0, replaced.stderr)
        calls = (self.home / 'sbatch-calls').read_text().splitlines()
        self.assertEqual(len(calls), 1)
        self.assertIn('--gres=gpu:6', calls[0])
        sc = (self.home / 'scancel-args').read_text()
        self.assertIn('--name=stage', sc)   # --restart cancels by job name (as router down does)
        self.assertIn('--user=', sc)

    def test_command_files_are_written_for_the_supervisor(self):
        base = self.base
        d = stage.ensure_dirs(base)
        # stage_request (team.py side) drops a JSON command file the supervisor polls.
        import subprocess as sp
        out = sp.check_output([sys.executable, str(RELEASE / 'team.py'), 'stage', 'load', 'glm53',
                               '--base', str(base)], env=self.env, stderr=sp.PIPE)
        cmds = list((d / 'cmd').glob('*.cmd'))
        self.assertEqual(len(cmds), 1, out)
        self.assertEqual(json.loads(cmds[0].read_text()), {'action': 'load', 'profile': 'glm53'})
        self.assertIn(b'Queued load glm53', out)

    def test_stage_is_team_shared_status_reserve_and_ownership(self):
        # The stage lives in $BASE/stage (never users/<login>/stage): a teammate must see
        # a stage someone else reserved, reattach instead of submitting a second one, and
        # only the owner can take it down (Slurm grants no cross-account scancel).
        base = self.base
        import subprocess as sp
        d = stage.ensure_dirs(base)
        stage.write_status(base, stage.STAGE_PORT, {})
        status = inference.load(d / 'status.json')
        status['owner'] = 'alice'  # another team member reserved this stage
        status['node'] = 'r3c1'
        inference.save(d / 'status.json', status)
        self.command('squeue', '#!/bin/bash\nprintf "600|stage|RUNNING|0:10|11:00:00|r3c1\n"\n')
        self.command('sbatch', '#!/bin/bash\necho "$*" >> "$HOME/sbatch-calls"\nprintf "700\n"\n')
        self.command('scancel', '#!/bin/bash\nexit 0\n')
        # Any member sees the stage and its owner ...
        data = json.loads(sp.check_output([sys.executable, str(RELEASE / 'team.py'), 'stage', 'status',
                                           '--json', '--base', str(base)], env=self.env, stderr=sp.PIPE))
        self.assertEqual(data['supervisor'], 'running')
        self.assertEqual(data['owner'], 'alice')
        self.assertEqual(data['router'], {'host': 'r3c1', 'port': stage.STAGE_PORT})
        # ... and reserve reattaches to it instead of submitting a second stage ...
        text = sp.check_output([sys.executable, str(RELEASE / 'team.py'), 'stage', 'reserve',
                                '--base', str(base)], env=self.env, stderr=sp.PIPE)
        self.assertIn(b'already reserved', text)
        self.assertFalse((self.home / 'sbatch-calls').exists())
        # ... while down refuses a non-owner.
        down = sp.run([sys.executable, str(RELEASE / 'team.py'), 'stage', 'down',
                       '--base', str(base)], env=self.env, capture_output=True)
        self.assertNotEqual(down.returncode, 0)
        self.assertIn(b'alice', down.stderr)


if __name__ == '__main__':
    unittest.main()
