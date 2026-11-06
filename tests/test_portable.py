"""Data recovery contracts: exact bytes, SQLite integrity, traversal and drift."""
import hashlib
import io
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import tarfile

import pytest
from kernel_tuning_lab.portable import restore_vault, sha256, verify_vault


def vault(tmp_path, *, path='run/measurements.sqlite', bad_member=False):
    source = tmp_path/'source.sqlite'
    with sqlite3.connect(source) as db:
        db.execute('CREATE TABLE measurements(value TEXT)')
        db.execute("INSERT INTO measurements VALUES ('original measurement')")
    original = source.read_bytes();digest = sha256(source)
    root = tmp_path/'vault';(root/'catalogs').mkdir(parents=True);(root/'archives').mkdir()
    pack = root/'archives/evidence.tar.gz'
    with tarfile.open(pack, 'w:gz') as tar:
        data = b'corrupt' if bad_member else original
        member = tarfile.TarInfo('objects/'+digest);member.size = len(data);tar.addfile(member, io.BytesIO(data))
    (root/'catalogs/source.json').write_text(json.dumps({'files': [{'path': path, 'sha256': digest, 'bytes': len(original)}]}))
    index = {'format': 'ktlab-content-archive-v1', 'packs': [{'source': 'test', 'file': str(pack), 'sha256': sha256(pack)}],
             'object_owner': {digest: 'test'}, 'source_catalogs': {'test': 'catalogs/source.json'}}
    (root/'catalogs/OBJECT_INDEX.json').write_text(json.dumps(index))
    return root, original


def test_restore_preserves_measurement_bytes_and_checks_real_sqlite(tmp_path):
    root, data = vault(tmp_path)
    assert verify_vault(root)['objects'] == 1
    output = tmp_path/'restored';proof = restore_vault(root, 'test', output)
    assert proof['SQLite_checked'] == 1 and proof['experiments_executed'] == 0
    assert (output/'run/measurements.sqlite').read_bytes() == data
    assert restore_vault(root, 'test', output)['files'] == 1


def test_restore_refuses_different_existing_evidence(tmp_path):
    root, data = vault(tmp_path);output = tmp_path/'restored'
    (output/'run').mkdir(parents=True);(output/'run/measurements.sqlite').write_bytes(b'keep this evidence')
    with pytest.raises(ValueError, match='overwrite'):
        restore_vault(root, 'test', output)
    assert (output/'run/measurements.sqlite').read_bytes() == b'keep this evidence'


def test_restore_rejects_parent_traversal(tmp_path):
    root, data = vault(tmp_path, path='../outside.sqlite')
    with pytest.raises(ValueError, match='traverse'):
        restore_vault(root, 'test', tmp_path/'restored')
    assert not (tmp_path/'outside.sqlite').exists()


def test_member_corruption_cannot_hide_behind_an_updated_archive_hash(tmp_path):
    root, data = vault(tmp_path, bad_member=True)
    with pytest.raises(ValueError, match='member SHA256'):
        verify_vault(root)
    with pytest.raises(ValueError, match='content SHA256'):
        restore_vault(root, 'test', tmp_path/'restored')
    assert not (tmp_path/'restored/run/measurements.sqlite').exists()


def test_changed_archive_is_rejected_before_restore(tmp_path):
    root, data = vault(tmp_path);pack = root/'archives/evidence.tar.gz'
    pack.write_bytes(pack.read_bytes()+b'changed')
    with pytest.raises(ValueError, match='archive SHA256'):
        restore_vault(root, 'test', tmp_path/'restored')


def test_doctor_does_not_load_gpu_dependencies():
    code = 'import sys;from kernel_tuning_lab.portable import doctor;x=doctor();assert "torch" not in sys.modules and "triton" not in sys.modules;assert not x["CUDA_initialized_by_doctor"];assert "moe-cu124" in x["profiles"]'
    subprocess.run([sys.executable, '-c', code], check=True, timeout=20)
