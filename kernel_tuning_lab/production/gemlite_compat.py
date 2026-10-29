"""Auditable optional-dtype registration patch for the pinned GemLite source.

Torch2.6 lacks E8M0. Low-bit A16 kernels do not use it. Preserve every Enum and
numeric function; condition only its three eager dictionary registrations.
No fake Torch dtype, Torch global monkeypatch, kernel rewrite or dtype cast.
"""
import hashlib
import ast
from pathlib import Path
from . import native_sources as sources

RAW_FILES=('gemlite/core.py','gemlite/bitpack.py','gemlite/quant_utils.py')


def raw_a16_source(relative):
    """Remove only optional Dynamo registrations/compile decorators. Every
    numerical/packing function body and Triton decorator remains byte-identical.
    This profile does not claim torch.compile/custom-op integration support.
    """
    if relative not in RAW_FILES:raise ValueError('not an audited raw-A16 host interop file')
    data=sources.source('gemlite',relative).read_text();tree=ast.parse(data);remove=set()
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef)):
            for decorator in node.decorator_list:
                function=decorator.func if isinstance(decorator,ast.Call) else decorator
                if ast.unparse(function) in ('torch.library.custom_op','torch.library.register_fake','torch.compile'):
                    remove.update(range(decorator.lineno-1,decorator.end_lineno))
    expected={'gemlite/core.py':2,'gemlite/bitpack.py':8,'gemlite/quant_utils.py':8}[relative]
    if len(remove)!=expected:raise ValueError('pinned optional interop decorator set changed')
    lines=data.splitlines(keepends=True)
    return ''.join(line for index,line in enumerate(lines) if index not in remove).encode()


def verify_runtime_source(root,relative):
    actual=(Path(root)/relative).read_bytes();original=sources.source('gemlite',relative).read_bytes()
    if actual==original:return {'profile':'unchanged upstream','actual_sha256':hashlib.sha256(actual).hexdigest()}
    if relative not in RAW_FILES or actual!=raw_a16_source(relative):raise RuntimeError('GemLite source differs from the exact audited raw-A16 interop profile: '+relative)
    return {'profile':'raw native A16; optional Dynamo/custom-op decorators omitted',
        'actual_sha256':hashlib.sha256(actual).hexdigest(),'original_sha256':hashlib.sha256(original).hexdigest(),
        'numerical_or_packing_bodies_modified':False,'unsupported':['torch.compile/custom-op integration','MX activation quantization integration']}


def optional_e8m0_source():
    data=sources.source('gemlite','gemlite/dtypes.py').read_text()
    pairs=(('    19: torch.float8_e8m0fnu,','    **({19: torch.float8_e8m0fnu} if hasattr(torch, "float8_e8m0fnu") else {}),'),
        ('    torch.float8_e8m0fnu: DType.E8M0,','    **({torch.float8_e8m0fnu: DType.E8M0} if hasattr(torch, "float8_e8m0fnu") else {}),'),
        ('    torch.float8_e8m0fnu: tl.uint8,','    **({torch.float8_e8m0fnu: tl.uint8} if hasattr(torch, "float8_e8m0fnu") else {}),'))
    for before,after in pairs:
        if data.count(before)!=1:raise ValueError('unexpected pinned GemLite dtype registration source')
        data=data.replace(before,after)
    return data.encode()


def verify_dtype_source(root):
    actual=(Path(root)/'gemlite/dtypes.py').read_bytes()
    original=sources.source('gemlite','gemlite/dtypes.py').read_bytes()
    if actual==original:return {'profile':'unchanged upstream','actual_sha256':hashlib.sha256(actual).hexdigest()}
    expected=optional_e8m0_source()
    if actual!=expected:raise RuntimeError('GemLite dtypes differs from both fixed upstream and exact optional-E8M0 registration patch')
    return {'profile':'conditional optional E8M0 registration; A16 math unchanged','actual_sha256':hashlib.sha256(actual).hexdigest(),
        'original_sha256':hashlib.sha256(original).hexdigest(),'unsupported_on_Torch26':['E8M0/MXFP scale conversion'],'numerical_functions_modified':False}
