"""Historical DeepSeek settings, isolated runtime installation and PP allocator regressions."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

LEONARDO = Path(__file__).resolve().parents[1]
SERVE = LEONARDO / 'serve'
EVIDENCE = LEONARDO / 'validation/runs/2026-10-20-deepseek41-16gpu'
PATCH = SERVE / 'runtime_patches/pp_kv_empty_groups'
sys.path.insert(0, str(LEONARDO / 'install'))
import overlay_files
from register_runtime import register

spec = importlib.util.spec_from_file_location('pp_guard_test', str(PATCH / 'pp_kv_guard.py'))
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)

SOURCE = '''from collections import defaultdict
def get_kv_cache_config_from_groups(kv_cache_groups):
    output=[]
    for group in kv_cache_groups:
        group_spec = group.kv_cache_spec
        layers_by_spec = defaultdict(list)
        if hasattr(group_spec, 'kv_cache_specs'):
            for layer_name, spec in group_spec.kv_cache_specs.items():
                layers_by_spec[spec].append(layer_name)
        elif group.layer_names:
            layers_by_spec[group_spec].extend(group.layer_names)
        output.extend(name for names in layers_by_spec.values() for name in names)
    return output
'''


class AllocatorGuard(unittest.TestCase):
    def run_source(self, source, groups):
        namespace = {}
        exec(compile(source, 'allocator-fixture', 'exec'), namespace)
        return namespace['get_kv_cache_config_from_groups'](groups)

    def patched(self, source=SOURCE):
        return guard.patch_source(source, hashlib.sha256(source.encode()).hexdigest())

    def test_projected_groups_keep_indices_and_allocate_only_worker_local_layers(self):
        groups = [SimpleNamespace(layer_names=[], kv_cache_spec=SimpleNamespace(kv_cache_specs={'remote': 1})),
                  SimpleNamespace(layer_names=['local'], kv_cache_spec=SimpleNamespace(kv_cache_specs={'local': 1, 'other': 1})),
                  SimpleNamespace(layer_names=['ordinary'], kv_cache_spec=2)]
        self.assertEqual(self.run_source(SOURCE, groups), ['remote', 'local', 'other', 'ordinary'])
        self.assertEqual(self.run_source(self.patched(), groups), ['local', 'ordinary'])
        self.assertEqual([g.layer_names for g in groups], [[], ['local'], ['ordinary']])

    def test_ordinary_local_groups_are_unchanged(self):
        groups = [SimpleNamespace(layer_names=['local'], kv_cache_spec=SimpleNamespace(kv_cache_specs={'local': 1}))]
        self.assertEqual(self.run_source(SOURCE, groups), self.run_source(self.patched(), groups))

    def test_unknown_source_and_changed_loop_are_rejected(self):
        with self.assertRaises(ImportError):
            guard.patch_source(SOURCE)
        with self.assertRaises(ImportError):
            self.patched(SOURCE.replace('layers_by_spec[spec].append(layer_name)', 'pass'))

    def test_guard_ignores_existing_bytecode(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'fixture.py'
            path.write_text(SOURCE)
            import py_compile
            py_compile.compile(str(path), doraise=True)
            loader = guard.GuardedLoader('fixture', str(path))
            from unittest.mock import patch
            original = guard.patch_source
            with patch.object(guard, 'patch_source', lambda text: original(text, hashlib.sha256(SOURCE.encode()).hexdigest())):
                namespace = {}
                exec(loader.get_code('fixture'), namespace)
            group = SimpleNamespace(layer_names=[], kv_cache_spec=SimpleNamespace(kv_cache_specs={'remote': 1}))
            self.assertEqual(namespace['get_kv_cache_config_from_groups']([group]), [])

    def test_sitecustomize_aborts_if_guard_installation_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            shutil.copyfile(str(PATCH / 'sitecustomize.py'), str(path / 'sitecustomize.py'))
            (path / 'pp_kv_guard.py').write_text('raise ImportError("broken guard")\n')
            result = subprocess.run([sys.executable, '-c', 'print("must not run")'], cwd=directory,
                                    env=dict(os.environ, PYTHONPATH=directory), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 78)
            self.assertNotIn(b'must not run', result.stdout)

    def test_import_hook_rejects_unexpected_vllm_source(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('vllm', 'vllm/v1', 'vllm/v1/core'):
                (root / name).mkdir(exist_ok=True)
                (root / name / '__init__.py').write_text('')
            (root / 'vllm/v1/core/kv_cache_utils.py').write_text(SOURCE)
            result = subprocess.run([sys.executable, '-c', 'import vllm.v1.core.kv_cache_utils'], cwd=directory,
                                    env=dict(os.environ, PYTHONPATH=os.pathsep.join((str(PATCH), directory))),
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn(b'unexpected base module SHA-256', result.stderr)


class RuntimeOverlay(unittest.TestCase):
    def test_reuse_and_patch_do_not_modify_installed_base(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            (base / 'inference-site.json').write_text('{}')
            original = base / 'original'
            original.mkdir()
            (original / 'image.sif').write_bytes(b'fake-sif')
            (original / 'overlay/tilelang').mkdir(parents=True)
            (original / 'overlay/tilelang/__init__.py').write_text('# base tilelang\n')
            parent_id = 'vllm-backport-sm80-0.13.1'
            runtime_id = parent_id + '-ppkv1'
            parent = base / 'runtimes' / parent_id
            register(parent, original / 'image.sif', original / 'overlay', parent_id)
            staging = base / 'staging'
            (staging / 'python-overlay').mkdir(parents=True)
            self.assertTrue(overlay_files.reuse_runtime(base, SERVE, runtime_id, staging))
            overlay_files.copy_overlay(SERVE, runtime_id, staging / 'python-overlay')
            self.assertFalse((parent / 'python-overlay/sitecustomize.py').exists())
            (staging / 'python-overlay/tilelang/__init__.py').write_text('# staging edit\n')
            self.assertEqual((parent / 'python-overlay/tilelang/__init__.py').read_text(), '# base tilelang\n')
            self.assertEqual((parent / 'image.sif').read_bytes(), b'fake-sif')
            register(base / 'runtimes' / runtime_id, staging / 'image.sif', staging / 'python-overlay', runtime_id)
            record = json.loads((base / 'runtimes' / runtime_id / 'runtime.json').read_text())
            self.assertIn('sitecustomize.py', record['overlay_files'])

            # A changed inherited package cannot silently seed another patched runtime.
            (parent / 'python-overlay/tilelang/__init__.py').write_text('# corrupt base\n')
            fresh = base / 'fresh-staging'
            (fresh / 'python-overlay').mkdir(parents=True)
            with self.assertRaises(ValueError):
                overlay_files.reuse_runtime(base, SERVE, runtime_id, fresh)

    def test_conflicting_sitecustomize_and_tampered_source_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shutil.copytree(str(SERVE / 'runtime_patches'), str(root / 'runtime_patches'))
            shutil.copyfile(str(SERVE / 'catalog.json'), str(root / 'catalog.json'))
            overlay = root / 'overlay'
            overlay.mkdir()
            (overlay / 'sitecustomize.py').write_text('# another customization\n')
            runtime_id = 'vllm-backport-sm80-0.13.1-ppkv1'
            with self.assertRaises(ValueError):
                overlay_files.copy_overlay(root, runtime_id, overlay)
            (overlay / 'sitecustomize.py').unlink()
            (root / 'runtime_patches/pp_kv_empty_groups/pp_kv_guard.py').write_text('# modified\n')
            with self.assertRaises(ValueError):
                overlay_files.copy_overlay(root, runtime_id, overlay)


class HistoricalEvidence(unittest.TestCase):
    def test_new_profile_preserves_measured_engine_settings_and_separate_identity(self):
        old = json.loads((EVIDENCE / 'model-profile.json').read_text())
        new = json.loads((SERVE / 'profiles/deepseek41-16gpu.json').read_text())
        # Renaming and a distinct direct-client port are the only engine-profile changes.
        for key in ('id', 'served_model', 'validation'):
            old.pop(key)
            new.pop(key)
        old['client'].pop('local_port')
        new['client'].pop('local_port')
        self.assertEqual(old, new)
        catalog = json.loads((SERVE / 'catalog.json').read_text())
        for item in catalog['runtimes'][new['runtime']]['overlay_files']:
            self.assertEqual(hashlib.sha256((SERVE / item['source']).read_bytes()).hexdigest(), item['sha256'])

    def test_historical_load_totals_and_startup_checks(self):
        data = json.loads((EVIDENCE / 'load-results.json').read_text())
        self.assertEqual(len(data['bursts']), 14)
        self.assertEqual(sum(b['successful'] for b in data['bursts']), 422)
        self.assertEqual(sum(b['failed'] for b in data['bursts']), 0)
        self.assertEqual({(b['concurrency'], b['repeat']) for b in data['bursts']},
                         {(level, repeat) for level in (1, 4, 8, 16, 32, 50, 100) for repeat in (1, 2)})
        checks = json.loads((EVIDENCE / 'startup-checks.json').read_text())
        self.assertEqual(len(checks), 7)
        self.assertTrue(all(c['passed'] for c in checks.values()))
        self.assertEqual(data['source_sha256'], hashlib.sha256((EVIDENCE / 'benchmark_serving.py').read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()
