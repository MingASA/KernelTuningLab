"""Shared production identities and trace records; no runner or dependency loader.

Source availability, build/load, static feasibility and GPU acceptance are
independent facts. Unknown compiled sets are never replaced by declared sets.
"""
from __future__ import annotations
from dataclasses import dataclass,field
import hashlib
import json
from pathlib import Path
from typing import Mapping

from ..models import canonical_json,stable_id


@dataclass(frozen=True)
class NativeCandidate:
    native_id: str
    parameters: Mapping
    family: str
    packing_key: str | None=None
    preparation: Mapping=field(default_factory=dict)
    compiled: bool | None=None

    @property
    def id(self):return stable_id(self.to_dict())
    def to_dict(self):
        return {'native_id':self.native_id,'parameters':dict(self.parameters),'family':self.family,
            'packing_key':self.packing_key,'preparation':dict(self.preparation),'compiled':self.compiled}


@dataclass(frozen=True)
class ConfigurationMapping:
    requested: Mapping
    effective: Mapping
    reason: str
    authority: str
    deduplicated_native_ids: tuple[str,...]=()

    def to_dict(self):
        return {'requested':dict(self.requested),'effective':dict(self.effective),'reason':self.reason,
            'authority':self.authority,'deduplicated_native_ids':list(self.deduplicated_native_ids),
            'changed':canonical_json(self.requested)!=canonical_json(self.effective)}


def git_blob_sha(path):
    data=Path(path).read_bytes()
    return hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()


def verify_source_files(root,pins):
    """Check exact installed numerical/host sources, not a version label alone."""
    root=Path(root)
    for relative,expected in pins.items():
        path=root/relative
        if not path.is_file() or git_blob_sha(path)!=expected:
            raise RuntimeError('production source differs from frozen snapshot: '+relative)
    return {'verified_files':len(pins),'source_identity':stable_id(dict(pins))}


def candidate_sets(*,declared,compiled=None,static=None,accepted=()):
    """Report different coverage axes without inferring compilation or success."""
    return {'source_expressible':list(declared),'compiled':None if compiled is None else list(compiled),
        'static_feasible':None if static is None else list(static),'GPU_accepted':list(accepted),
        'notice':'Unknown and not-executed remain distinct from unsupported/failed.'}


def preparation_identity(backend,logical_input_id,candidate,packing_parameters):
    return stable_id([backend,logical_input_id,candidate.packing_key,dict(packing_parameters)])


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(canonical_json(value)+'\n');temp.replace(path)
