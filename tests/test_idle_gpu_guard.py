"""CPU-only resource fixtures; no CUDA load and no signals to real external jobs."""
import importlib.util
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

spec=importlib.util.spec_from_file_location('idle_gpu_guard',Path(__file__).parents[1]/'scripts/with_idle_gpu.py')
guard=importlib.util.module_from_spec(spec);spec.loader.exec_module(guard)


def device(index=0,**kwargs):
    return {'index':index,'name':'SYNTHETIC unit GPU','used_mib':3,'util_pct':0,'pids':[],**kwargs}


def test_gpu_admission_requires_repeated_empty_memory_util_and_processes():
    samples=[{'a':device(0,used_mib=1000),'b':device(1)},{'a':device(0),'b':device(1)}]
    assert guard.choose_idle(samples)=='b'
    for change in ({'used_mib':1000},{'util_pct':1},{'pids':[12345]}):
        with pytest.raises(guard.ResourceUnavailable):guard.choose_idle([{'a':device(**change)},{'a':device()}])
    with pytest.raises(guard.ResourceUnavailable):guard.choose_idle(samples,'a')


def test_pid_reuse_is_never_signalled_or_recognized_as_owned(monkeypatch):
    sent=[]
    monkeypatch.setattr(guard,'proc_identity',lambda pid:(1,200))
    monkeypatch.setattr(guard.os,'kill',lambda pid,sig:sent.append((pid,sig)))
    guard.signal_owned({4321:100},signal.SIGTERM)
    assert sent==[]
    assert not guard.descendant(4321,4321,100)
    known={4321:100};guard.collect_owned(4321,known)
    assert known=={4321:100}


def test_unknown_nvidia_process_state_denies_admission(monkeypatch):
    class Result:
        returncode=0
        stdout='0, unit-uuid, SYNTHETIC unit GPU, 1000, N/A, 0\n'
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**k:Result())
    with pytest.raises(guard.ResourceUnavailable,match='unknown'):guard.snapshot()


def test_invisible_and_reused_gpu_pids_are_foreign_even_with_same_login_account(monkeypatch):
    identities={10:(1,100),11:(10,110),12:None,13:(1,999)}
    monkeypatch.setattr(guard,'proc_identity',lambda pid:identities.get(pid))
    assert guard.foreign_pids([11,12,13],10,100,{10:100,13:100})==[12,13]


def test_xml_includes_graphics_tasks_in_busy_gpu_detection(monkeypatch):
    class Result:
        returncode=0
        stdout='<nvidia_smi_log><gpu><uuid>unit</uuid><product_name>SYNTHETIC</product_name><display_active>Disabled</display_active><fb_memory_usage><total>1000 MiB</total><used>3 MiB</used></fb_memory_usage><utilization><gpu_util>0 %</gpu_util></utilization><processes><process_info><pid>12345</pid><type>G</type></process_info></processes></gpu></nvidia_smi_log>'
    monkeypatch.setattr(guard.subprocess,'run',lambda *a,**k:Result())
    state=guard.snapshot();assert state['unit']['pids']==[12345]
    assert not guard.idle(state['unit'])


def test_busy_gpu_never_launches_a_command(tmp_path,monkeypatch):
    monkeypatch.setattr(guard,'snapshot',lambda:{'unit':device(pids=[12345])})
    monkeypatch.setattr(guard.time,'sleep',lambda seconds:None)
    def forbidden(*a,**k):raise AssertionError('must not launch on a busy GPU')
    monkeypatch.setattr(guard.subprocess,'Popen',forbidden)
    args=SimpleNamespace(receipt=str(tmp_path/'receipt.json'),timeout=10,ready_samples=2,interval=1,
        gpu='auto',expected_model='',select_only=False,command=['irrelevant'],cwd=None)
    assert guard.run(args)==75


def test_only_proven_terminated_own_pid_can_be_ignored_in_retiring_nvml_list(monkeypatch):
    monkeypatch.setattr(guard,'proc_identity',lambda pid:None)
    monkeypatch.setattr(guard,'proc_info',lambda pid:None)
    def absent(pid,signum):
        assert signum==0
        raise ProcessLookupError
    monkeypatch.setattr(guard.os,'kill',absent)
    assert guard.foreign_pids([123,456],10,100,{123:200})==[456]
    monkeypatch.setattr(guard.os,'kill',lambda pid,sig:None)
    assert guard.foreign_pids([123],10,100,{123:200})==[123]


def test_matching_owned_zombie_is_retired_but_reused_pid_is_not(monkeypatch):
    monkeypatch.setattr(guard,'proc_identity',lambda pid:None)
    monkeypatch.setattr(guard,'proc_info',lambda pid:(10,200,'Z'))
    assert guard.foreign_pids([123],10,100,{123:200})==[]
    assert guard.foreign_pids([123],10,100,{123:100})==[123]


@pytest.mark.skipif(not Path('/proc/self/stat').is_file(),reason='Linux owned process tracker')
def test_short_lived_owned_worker_retirement_does_not_look_like_foreign_gpu_job(tmp_path,monkeypatch):
    marker=tmp_path/'pid'
    child='import os,pathlib,time; pathlib.Path('+repr(str(marker))+').write_text(str(os.getpid())); time.sleep(.4)'
    parent='import subprocess,sys,time; subprocess.run([sys.executable,"-c",'+repr(child)+']); time.sleep(.3)'
    def fake_nvml(gpu_uuid=None):
        pids=[int(marker.read_text())] if marker.exists() else []
        return {'unit':device(pids=pids)}
    monkeypatch.setattr(guard,'snapshot',fake_nvml);monkeypatch.delenv('CUDA_VISIBLE_DEVICES',raising=False)
    receipt=tmp_path/'receipt.json'
    args=SimpleNamespace(receipt=str(receipt),timeout=10,ready_samples=2,interval=.03,
        gpu='auto',expected_model='',select_only=False,command=[sys.executable,'-c',parent],cwd=None)
    assert guard.run(args)==0
    import json
    result=json.loads(receipt.read_text())
    assert result['status']=='passed' and result['eligible']
    assert result['command']==args.command and result['cwd']==args.cwd
    assert result['external_processes_observed']==0 and result['retired_owned_nvml_entries']>0


@pytest.mark.skipif(not Path('/proc/self/stat').is_file(),reason='Linux sessions')
def test_cleanup_promptly_stops_owned_session_even_when_bash_ignores_sigint(tmp_path):
    marker=tmp_path/'child.pid';unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'])
    root=subprocess.Popen(['bash','-c','trap "" INT; sleep 30 & child=$!; echo "$child" > "$1"; wait',
        'unit-owned-session',str(marker)],start_new_session=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    known={root.pid:guard.proc_identity(root.pid)[1]}
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists();child=int(marker.read_text());guard.collect_owned(root.pid,known)
        start=time.monotonic();guard.stop_owned(root,known)
        assert time.monotonic()-start<3 and root.poll() is not None and guard.proc_identity(child) is None
        assert unrelated.poll() is None
    finally:
        guard.stop_owned(root,known);unrelated.terminate();unrelated.wait(timeout=5)


@pytest.mark.skipif(not Path('/proc/self/stat').is_file(),reason='Linux process resource metadata')
def test_cleanup_stops_only_its_own_cpu_tree_and_leaves_unrelated_fixture_alive(tmp_path):
    marker=tmp_path/'child.pid'
    code='import pathlib,subprocess,sys,time; p=subprocess.Popen([sys.executable,"-c","import time; time.sleep(30)"]); pathlib.Path(sys.argv[1]).write_text(str(p.pid)); time.sleep(30)'
    unrelated=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    root=subprocess.Popen([sys.executable,'-c',code,str(marker)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    known={root.pid:guard.proc_identity(root.pid)[1]}
    try:
        deadline=time.monotonic()+5
        while not marker.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert marker.exists();child=int(marker.read_text())
        guard.collect_owned(root.pid,known);assert child in known and unrelated.pid not in known
        guard.stop_owned(root,known)
        assert root.poll() is not None and unrelated.poll() is None
        assert guard.proc_identity(child) is None
    finally:
        guard.stop_owned(root,known)
        unrelated.terminate();unrelated.wait(timeout=5)
