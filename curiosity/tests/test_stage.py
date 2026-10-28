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
            # A second, differently sized model takes a slice of its own size, no overlap.
            with mock.patch.object(node, 'verify_image', return_value=[]), \
                    mock.patch.object(stage, 'request_healthy', return_value=True):
                records = stage.load_model(str(base), 'qwen38flash', records)
            self.assertEqual(records['glm53']['slice'], '0,1,2,3')
            self.assertEqual(records['qwen38flash']['slice'], '4')
            self.assertEqual(records['qwen38flash']['api_port'], stage.API_PORT_BASE + 10)
            # With 0-4 busy no aligned 4-GPU block is free on the 8-GPU node.
            self.assertIsNone(stage.free_gpu_slice(records))
            # Unload removes the container record and the endpoint, freeing the slice.
            stage.unload_model(str(base), 'glm53', records)
            self.assertNotIn('glm53', stage.load_models(str(base)))
            self.assertIsNotNone(stage.free_gpu_slice(records))
        finally:
            pass

    def test_status_json_references_the_gpu_slices_so_loads_cannot_overlap(self):
        # status.json must reference every slice the stage owns: the load in progress
        # claims its slice BEFORE the container starts (models.json only gains the
        # record once /health passes) and loaded models keep theirs. Without the
        # reference a second load could be handed GPUs a container already holds.
        self.docker_stub()
        saved_path = os.environ['PATH']
        os.environ['PATH'] = str(self.bin) + ':' + saved_path
        self.addCleanup(lambda: os.environ.__setitem__('PATH', saved_path))
        base = self.base
        claimed = {}

        def probe(cfg):  # runs during preflight: the claim must already be on disk
            claimed.update(inference.load(stage.stage_dir(base) / 'status.json'))
            return []

        with mock.patch.object(node, 'verify_image', side_effect=probe), \
                mock.patch.object(stage, 'request_healthy', return_value=True):
            stage.load_model(str(base), 'glm53', {})
        # 1. the pending claim pinned its slice and container name before docker run
        self.assertEqual(claimed['loading']['slice'], '0,1,2,3')
        self.assertEqual(claimed['loading']['name'], 'stage-glm53')
        # 2. the loaded model keeps its slice + port referenced in status.json
        status = inference.load(stage.stage_dir(base) / 'status.json')
        self.assertEqual(status['models']['glm53']['slice'], '0,1,2,3')
        self.assertEqual(status['models']['glm53']['api_port'], stage.API_PORT_BASE)
        # 3. a pending claim reserves its slice: free_gpu_slice must not hand it out twice
        self.assertEqual(stage.free_gpu_slice({}, {'slice': '0,1,2,3'}), '4,5,6,7')
        # 4. a stale claim (supervisor died mid-load, record never reached models.json):
        #    the next load must skip that slice instead of overlapping it.
        inference.save(stage.stage_dir(base) / 'models.json', {})
        inference.save(stage.stage_dir(base) / 'status.json',
                       dict(status, models={}, loading={'profile': 'glm53', 'name': 'stage-glm53',
                                                         'slice': '0,1,2,3'}))
        with mock.patch.object(node, 'verify_image', return_value=[]), \
                mock.patch.object(stage, 'request_healthy', return_value=True):
            records = stage.load_model(str(base), 'glm53', {})
        self.assertEqual(records['glm53']['slice'], '4,5,6,7')

    def test_stage_dirs_and_files_stay_group_usable_across_members(self):
        # A stage created by one member must be fully usable by the others. Two classic
        # breakages: directories without setgid (new files inherit the CREATOR's primary
        # group, not the team group) and shared files at 0644 (teammates can read but
        # never rewrite status/models). And a chmod on another member's path raises
        # EPERM -- commands must survive that, not crash on it.
        import stat as stat_mod
        d = stage.ensure_dirs(str(self.base))
        for sub in ('', 'cmd', 'done'):
            self.assertTrue(os.stat(d / sub).st_mode & stat_mod.S_ISGID,
                            'setgid missing on {}'.format(sub))
        stage.write_status(str(self.base), 18500, {}, None)
        self.assertEqual(os.stat(d / 'status.json').st_mode & 0o777, 0o664)
        stage.save_models(str(self.base), {})
        self.assertEqual(os.stat(d / 'models.json').st_mode & 0o777, 0o664)
        with mock.patch.object(Path, 'chmod', side_effect=PermissionError):
            # simulate a stage owned by ANOTHER member: same commands must not die
            stage.ensure_dirs(str(self.base))
            stage.write_status(str(self.base), 18500, {}, None)

    def test_profile_gpu_affinity_sizes_the_slice(self):
        # Each profile declares its GPU affinity through parallelism (TP x PP x DP):
        # glm53 = 4 GPUs gets a 4-GPU slice, gemma4 = 1 GPU gets ONE gpu, and blocks
        # stay aligned so a freed 4-GPU block remains usable by the next 4-GPU profile.
        self.docker_stub()
        saved_path = os.environ['PATH']
        os.environ['PATH'] = str(self.bin) + ':' + saved_path
        self.addCleanup(lambda: os.environ.__setitem__('PATH', saved_path))
        base = self.base
        with mock.patch.object(node, 'verify_image', return_value=[]), \
                mock.patch.object(stage, 'request_healthy', return_value=True):
            records = stage.load_model(str(base), 'glm53', {})
            self.assertEqual(records['glm53']['slice'], '0,1,2,3')
            self.assertEqual(records['glm53']['gpus'], 4)
            records = stage.load_model(str(base), 'gemma4', records)
        self.assertEqual(records['gemma4']['slice'], '4')  # its own 1-GPU block, no overlap
        self.assertEqual(records['gemma4']['gpus'], 1)
        self.assertEqual(stage.profile_gpus(inference.load(RELEASE / 'profiles' / 'qwen38flash.json')), 1)
        self.assertEqual(stage.profile_gpus(inference.load(RELEASE / 'profiles' / 'glm53.json')), 4)
        # status.json references the affinity + slice of every loaded model, and the
        # finished claim is cleared (no stale pinning once the record exists).
        status = inference.load(stage.stage_dir(base) / 'status.json')
        self.assertEqual(status['models']['glm53']['gpus'], 4)
        self.assertEqual(status['models']['glm53']['slice'], '0,1,2,3')
        self.assertEqual(status['models']['gemma4']['gpus'], 1)
        self.assertIsNone(status['loading'])
        # Mixed affinities stay coherent: with 0-4 busy, a 2-GPU profile gets 6-7 and a
        # 1-GPU one the first free gpu, never an overlapping block.
        self.assertEqual(stage.free_gpu_slice(records, None, 2), '6,7')
        self.assertEqual(stage.free_gpu_slice(records, None, 1), '5')
        # the last free gpus can be taken by a pending claim: then the stage is full
        self.assertEqual(stage.free_gpu_slice(records, {'slice': '5,6'}, 1), '7')
        self.assertIsNone(stage.free_gpu_slice(records, {'slice': '5,6,7'}, 1))
        # After an unload, the freed 4-GPU block stays whole for the next big profile.
        stage.unload_model(str(base), 'glm53', records)
        self.assertEqual(stage.free_gpu_slice(stage.load_models(str(base)), None, 4), '0,1,2,3')

    def test_reclaim_claims_frees_the_slice_of_a_dead_load(self):
        # A container that outlived the supervisor still owns its GPUs: stage startup
        # must docker rm -f the claimed container so the slice can be handed out again.
        import subprocess
        base = self.base
        d = stage.ensure_dirs(base)
        inference.save(d / 'status.json', {'supervisor': 'running', 'router_port': stage.STAGE_PORT,
                                           'loading': {'profile': 'glm53', 'name': 'stage-glm53',
                                                       'slice': '0,1,2,3'}})
        calls = []

        def fake_run(argv, **kwargs):
            calls.append(list(argv))
            return subprocess.CompletedProcess(argv, 0)
        with mock.patch.object(stage.subprocess, 'run', side_effect=fake_run):
            stage.reclaim_claims(base, {}, (d / 'test.log').open('a'))
        self.assertEqual(calls, [['docker', 'rm', '-f', 'stage-glm53']])
        # A claim matching an already-recorded model is an unclean shutdown, not a dead
        # load: sweep() judges that one by /health, reclaim must not touch it.
        calls[:] = []
        with mock.patch.object(stage.subprocess, 'run', side_effect=fake_run):
            stage.reclaim_claims(base, {'glm53': {'name': 'stage-glm53', 'slice': '0,1,2,3'}},
                                 (d / 'test.log').open('a'))
        self.assertEqual(calls, [])

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
        self.assertIn('--gres=gpu:8', calls[0])
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
        cmd_mode = cmds[0].stat().st_mode & 0o777
        self.assertEqual(cmd_mode, 0o664, oct(cmd_mode))  # CLI umask 077: the shared supervisor (and any teammate) must read AND rewrite it
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
        self.assertEqual((d / 'status.json').stat().st_mode & 0o777, 0o644, 'status.json must be team-readable')
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

    def test_stage_router_token_is_shared_with_the_team(self):
        base = self.base
        import subprocess as sp
        d = stage.ensure_dirs(base)
        # First: a shared token alone (no per-account file) -- `router token` prints it
        # and does not fabricate a per-account token for the caller.
        shared = d / 'router_token'
        shared.write_text('team-secret-token\n')
        os.chmod(str(shared), 0o644)
        out = sp.check_output([sys.executable, str(RELEASE / 'team.py'), 'router', 'token',
                               '--base', str(base)], env=self.env, stderr=sp.PIPE)
        self.assertEqual(out.decode().strip(), 'team-secret-token')
        self.assertFalse((inference.user_dir(base) / 'router_token').exists(),
                         'ensure must not create a per-account token while the stage shares one')
        # Then: the supervisor copies the owner's token into the shared folder (0644).
        shared.unlink()
        owner_token = inference.user_dir(base) / 'router_token'
        owner_token.parent.mkdir(parents=True, exist_ok=True)
        owner_token.write_text('owner-secret-token\n')
        self.assertTrue(stage.share_router_token(base))
        self.assertEqual(shared.read_text().strip(), 'owner-secret-token')
        self.assertEqual(shared.stat().st_mode & 0o777, 0o664)


if __name__ == '__main__':
    unittest.main()
