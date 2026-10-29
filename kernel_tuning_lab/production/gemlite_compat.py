"""Auditable optional-dtype registration patch for the pinned GemLite source.

Torch2.6 lacks E8M0. Low-bit A16 kernels do not use it. Preserve every Enum and
numeric function; condition only its three eager dictionary registrations.
No fake Torch dtype, Torch global monkeypatch, kernel rewrite or dtype cast.
"""
import hashlib
from pathlib import Path
from . import native_sources as sources


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
