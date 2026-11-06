"""Backup selection/integrity using tiny synthetic files, never real archives."""
import hashlib
import os
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def fixture(tmp_path):
    repo=tmp_path/'repo';(repo/'scripts').mkdir(parents=True);(repo/'artifacts').mkdir()
    script=repo/'scripts/backup_research_artifacts.sh';shutil.copyfile(ROOT/'scripts/backup_research_artifacts.sh',script)
    names=['5060ti-evidence-20261025.tar.gz','research-workflow-evidence-20261025.tar.gz',
        'research-phase1-evidence-20261025.tar.gz','research-phase2-evidence-20261026-v2.tar.gz',
        'research-phase2-evidence-20261026.tar.gz']
    for name in names:
        path=repo/'artifacts'/name;path.write_bytes(('synthetic backup fixture '+name).encode())
        Path(str(path)+'.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+name+'\n')
    return repo,script,names


def test_backup_prefers_final_phase2_and_preserves_all_sources(tmp_path):
    repo,script,names=fixture(tmp_path);destination=tmp_path/'backup'
    before={p.name:p.read_bytes() for p in (repo/'artifacts').iterdir()}
    result=subprocess.run(['bash',str(script)],env={**os.environ,'KT_BACKUP_DIR':str(destination)},capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
    assert all((destination/name).is_file() for name in names[:-1])
    assert not (destination/names[-1]).exists()
    assert {p.name:p.read_bytes() for p in (repo/'artifacts').iterdir()}==before


def test_existing_invalid_destination_is_reported_and_never_overwritten(tmp_path):
    repo,script,names=fixture(tmp_path);destination=tmp_path/'backup';destination.mkdir()
    target=destination/names[0];target.write_bytes(b'preexisting invalid backup')
    result=subprocess.run(['bash',str(script)],env={**os.environ,'KT_BACKUP_DIR':str(destination)},capture_output=True,text=True)
    assert result.returncode!=0
    assert target.read_bytes()==b'preexisting invalid backup'
