"""CPU cooperation contracts; no CUDA timing/quiescence proof is simulated."""
import os
import threading
import time
import pytest
from kernel_tuning_lab.inference_resources import ProcessCoordinator,ResourceIsolationError,ResourceContextLimit
from kernel_tuning_lab.models import HardwareSpec


H=HardwareSpec('CPU phase fixture',memory_bytes=100,metadata={'device_uuid':'fixture-device'})


def test_resident_weights_can_remain_but_new_cooperative_submissions_block_during_measurement():
    gate=ProcessCoordinator();calls=[]
    token=gate.register('fixture-model',mode='single_card_resident',device_uuid='fixture-device',
        identity={'fixture':True},synchronize=lambda:calls.append('sync'),resident_bytes=80)
    started=threading.Event();entered=threading.Event()
    def background():
        started.set()
        with gate.inference_window(token):entered.set()
    with gate.measurement_window(H) as proof:
        thread=threading.Thread(target=background);thread.start();assert started.wait(1)
        assert not entered.wait(.05)
        assert proof['new_submissions_blocked'] and proof['resident_weights_may_remain']
    thread.join(1);assert entered.is_set() and calls==['sync','sync','sync']


def test_external_pid_and_false_separate_device_declarations_are_not_proof():
    gate=ProcessCoordinator()
    with pytest.raises(ResourceIsolationError,match='external'):
        gate.register('external',mode='single_card_resident',device_uuid='fixture-device',identity={},synchronize=lambda:None,owner_pid=os.getpid()+1000)
    gate.register('false-other-device',mode='separate_device',device_uuid='fixture-device',identity={},synchronize=lambda:None)
    with pytest.raises(ResourceIsolationError,match='measurement device'):
        with gate.measurement_window(H):pass


def test_same_thread_recursive_overlap_rejected_and_resource_budget_is_context_scoped():
    gate=ProcessCoordinator();exclusive=gate.resource_identity()
    token=gate.register('model',mode='single_card_resident',device_uuid='fixture-device',identity={'model_sha':'fixture'},synchronize=lambda:None,resident_bytes=80)
    assert gate.resource_identity()!=exclusive
    with gate.inference_window(token):
        with pytest.raises(ResourceIsolationError):
            with gate.measurement_window(H):pass
    with pytest.raises(ResourceContextLimit):gate.check_known_allocation(H,30)
    gate.unregister(token);gate.check_known_allocation(H,30)


def test_resident_parent_cannot_spawn_an_unregistered_measurement_worker(tmp_path,monkeypatch):
    from kernel_tuning_lab.inference_resources import COORDINATOR
    from kernel_tuning_lab.asset_runner import execute_asset_experiment
    token=COORDINATOR.register('fixture-resident-parent',mode='single_card_resident',device_uuid='fixture-device',
        identity={'fixture':True},synchronize=lambda:None,resident_bytes=4)
    try:
        monkeypatch.setattr('subprocess.Popen',lambda *a,**k:pytest.fail('must reject before starting a worker'))
        result=execute_asset_experiment(tmp_path/'unread-spec.json',tmp_path/'rejected')
        assert result['phase']=='resource_handoff' and result['exit_code']==2 and not result['physical_gpu_validated']
        # Same-process measurements remain permitted with persistent weights.
        with COORDINATOR.measurement_window(H) as proof:assert proof['resident_weights_may_remain']
    finally:COORDINATOR.unregister(token)
