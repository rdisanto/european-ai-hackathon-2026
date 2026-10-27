"""Narrow import-time fix for a pinned backport; never edits the SIF.

Equivalent allocation filter: https://github.com/vllm-project/vllm/pull/54793
Keep projected group indices intact, but allocate only worker-local layers.
"""
import ast
import hashlib
import importlib.abc
import importlib.machinery
import sys

TARGET = 'vllm.v1.core.kv_cache_utils'
EXPECTED_SHA256 = '3326435f857c903711939d4f6b2776395262729b464b3973949c1234a163ff3f'
VERSION = 'worker-local-uniform-kv-v1'


def patch_source(source, expected_sha256=EXPECTED_SHA256):
    if hashlib.sha256(source.encode()).hexdigest() != expected_sha256:
        raise ImportError('PP KV guard: unexpected base module SHA-256')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'get_kv_cache_config_from_groups']
    if len(functions) != 1:
        raise ImportError('PP KV guard: missing or ambiguous target function')
    function = functions[0]
    lines = source.splitlines(keepends=True)
    section = ''.join(lines[function.lineno - 1:function.end_lineno])
    needle = ('            for layer_name, spec in group_spec.kv_cache_specs.items():\n'
              '                layers_by_spec[spec].append(layer_name)\n')
    replacement = ('            for layer_name, spec in group_spec.kv_cache_specs.items():\n'
                   '                if layer_name in group.layer_names:\n'
                   '                    layers_by_spec[spec].append(layer_name)\n')
    if section.count(needle) != 1:
        raise ImportError('PP KV guard: unexpected allocation loop')
    section = section.replace(needle, replacement)
    return ''.join(lines[:function.lineno - 1]) + section + ''.join(lines[function.end_lineno:])


class GuardedLoader(importlib.machinery.SourceFileLoader):
    def get_code(self, fullname):
        # Always use checked source, so an existing unpatched pyc cannot win.
        return compile(patch_source(self.get_source(fullname)), self.path, 'exec', dont_inherit=True)

    def exec_module(self, module):
        super().exec_module(module)
        module.PP_KV_GUARD_VERSION = VERSION
        print('PP_KV_GUARD_APPLIED ' + VERSION, file=sys.stderr, flush=True)


class GuardFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None or not isinstance(spec.loader, importlib.machinery.SourceFileLoader):
            raise ImportError('PP KV guard: target must have Python source')
        spec.loader = GuardedLoader(fullname, spec.origin)
        return spec


def install():
    if TARGET in sys.modules:
        raise ImportError('PP KV guard installed after target import')
    if not any(isinstance(finder, GuardFinder) for finder in sys.meta_path):
        sys.meta_path.insert(0, GuardFinder())
