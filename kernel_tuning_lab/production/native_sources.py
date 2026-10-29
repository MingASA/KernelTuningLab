"""Verified upstream sources and narrowly extracted host rules.

No GPU package import is needed for discovery. Numerical kernels are never
translated: execution uses the actual backend or verified source extraction.
"""
from __future__ import annotations
import ast
import __future__
from functools import lru_cache
import hashlib
import json
from pathlib import Path

from .contracts import git_blob_sha

ROOT=Path(__file__).with_name('upstream')/'suite'


@lru_cache(maxsize=1)
def manifest():return json.loads((ROOT/'manifest.json').read_text())


def source(backend,relative):
    row=manifest()[backend]['files'][relative];path=ROOT/row['vendored']
    if git_blob_sha(path)!=row['git_blob_sha'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
        raise RuntimeError('vendored source identity changed: '+backend+'/'+relative)
    return path


def host_functions(backend,relative,names,namespace,*,remove_imports=()):
    """Compile original host-function bodies with explicit dependency injection.

    Removing a function-local import is permitted only by a named allowlist;
    the caller supplies that dependency and records this host-only extraction.
    No inferred translation of arbitrary callbacks or numerical kernels.
    """
    path=source(backend,relative);tree=ast.parse(path.read_text());chosen=[]
    class Imports(ast.NodeTransformer):
        def visit_ImportFrom(self,node):
            key='.'*node.level+(node.module or '')
            return None if key in remove_imports else node
    for node in tree.body:
        if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
            node.decorator_list=[];chosen.append(Imports().visit(node))
    if {node.name for node in chosen}!=set(names):raise ValueError('requested upstream host symbol is absent')
    exec(compile(ast.fix_missing_locations(ast.Module(body=chosen,type_ignores=[])),str(path),'exec',
        flags=__future__.annotations.compiler_flag),namespace)
    return {name:namespace[name] for name in names}


def native_component(root,backend,files=None):
    """Bind an installed/source checkout to exact upstream file contents."""
    root=Path(root);rows=manifest()[backend]['files'];names=files or rows.keys()
    for name in names:
        if git_blob_sha(root/name)!=rows[name]['git_blob_sha']:
            raise RuntimeError('installed '+backend+' file differs from pinned source: '+name)
    return {'repository':manifest()[backend]['repository'],'revision':manifest()[backend]['revision'],
        'source_git_blobs':{n:rows[n]['git_blob_sha'] for n in names}}


def pin(backend):
    data=manifest()[backend]
    return {'repository':data['repository'],'revision':data['revision'],
        'source_sha256':{k:v['sha256'] for k,v in data['files'].items()}}


def jit_module(backend,relative,namespace,*,name):
    """Original definitions/decorators with only dependency imports supplied.

    Used by extracted native Triton implementations. This does not rewrite any
    function body or launch a kernel; caller controls the admitted device.
    """
    import sys,types
    path=source(backend,relative);tree=ast.parse(path.read_text())
    nodes=[n for n in tree.body if not isinstance(n,(ast.Import,ast.ImportFrom))]
    module=types.ModuleType(name);module.__file__=str(path);module.__dict__.update(namespace);sys.modules[name]=module
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),module.__dict__)
    return module
