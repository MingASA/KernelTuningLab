"""Run one bounded command on an idle GPU; abort only our descendants on conflict.

Linux, stdlib and nvidia-smi only. This is an admission/monitoring wrapper, not
a scheduler or a GPU reservation. It changes no global device state. Receipts
contain private device/PID information and belong outside public Git.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import xml.etree.ElementTree as ET


class ResourceUnavailable(RuntimeError):
    pass


def snapshot():
    result=subprocess.run(['nvidia-smi','-q','-x'],capture_output=True,text=True,timeout=8)
    if result.returncode:raise ResourceUnavailable('nvidia-smi inspection failed')
    devices={}
    try:
        for index,gpu in enumerate(ET.fromstring(result.stdout).findall('gpu')):
            uuid=gpu.findtext('uuid');processes=gpu.find('processes')
            if not uuid or processes is None or (processes.text or '').strip() not in ('','None'):
                raise ResourceUnavailable('unknown GPU process visibility')
            number=lambda path:int(gpu.findtext(path).split()[0])
            devices[uuid]={'uuid':uuid,'index':index,'name':gpu.findtext('product_name'),
                'memory_mib':number('fb_memory_usage/total'),'used_mib':number('fb_memory_usage/used'),
                'util_pct':number('utilization/gpu_util'),'display_active':gpu.findtext('display_active'),
                'pids':[int(p.findtext('pid')) for p in processes.findall('process_info')]}
    except (ET.ParseError,ValueError,TypeError,AttributeError) as exc:
        raise ResourceUnavailable('unknown device/process state; admission denied') from exc
    if not devices:raise ResourceUnavailable('no visible physical GPUs')
    return devices


def idle(device):
    return not device['pids'] and device['used_mib']<=64 and device['util_pct']==0 and device.get('display_active','Disabled')=='Disabled'


def choose_idle(samples,requested='auto',expected_model=''):
    allowed=None
    for devices in samples:
        current={uuid for uuid,device in devices.items() if idle(device)
            and (requested=='auto' or uuid==requested) and expected_model.lower() in device['name'].lower()}
        allowed=current if allowed is None else allowed&current
    if not allowed:raise ResourceUnavailable('no continuously idle requested GPU; no command launched')
    return min(allowed,key=lambda uuid:samples[-1][uuid]['index'])


def proc_identity(pid):
    """Only kernel process metadata, never another task's cmdline or files."""
    try:
        fields=Path('/proc',str(pid),'stat').read_text().rsplit(')',1)[1].split()
        if fields[0]=='Z':return None
        return int(fields[1]),int(fields[19])
    except (FileNotFoundError,ProcessLookupError,PermissionError,IndexError):return None


def descendant(pid,root_pid,root_start):
    seen=set()
    while pid>1 and pid not in seen:
        seen.add(pid);identity=proc_identity(pid)
        if identity is None:return False
        if pid==root_pid:return identity[1]==root_start
        pid=identity[0]
    return False


def collect_owned(root_pid,known):
    root=proc_identity(root_pid)
    if root_pid not in known or root is None or root[1]!=known[root_pid]:return
    pending=[root_pid];visited=set()
    while pending:
        pid=pending.pop()
        if pid in visited:continue
        visited.add(pid);identity=proc_identity(pid)
        if identity is None:continue
        if pid not in known or known[pid]!=identity[1]:continue
        for path in Path('/proc',str(pid),'task').glob('*/children'):
            try:
                for child in map(int,path.read_text().split()):
                    info=proc_identity(child)
                    if info is not None and info[0]==pid:
                        known[child]=info[1];pending.append(child)
            except (FileNotFoundError,ProcessLookupError,PermissionError):pass


def signal_owned(known,signum):
    for pid,started in known.items():
        identity=proc_identity(pid)
        if pid<=1 or identity is None or identity[1]!=started:continue
        try:os.kill(pid,signum)
        except ProcessLookupError:pass


def foreign_pids(pids,root_pid,root_start,known):
    foreign=[]
    for pid in pids:
        identity=proc_identity(pid)
        if identity is None or not (known.get(pid)==identity[1] or descendant(pid,root_pid,root_start)):
            foreign.append(pid)
    return foreign


def stop_owned(process,known):
    # Capture descendants before the parent can exit/reparent them. A reused PID
    # never inherits ownership. No pkill, global process-group or device reset.
    collect_owned(process.pid,known)
    if process.poll() is None:process.send_signal(signal.SIGINT)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:pass
    signal_owned(known,signal.SIGTERM)
    remaining=[(pid,proc_identity(pid)) for pid in known]
    if any(info is not None and info[1]==known[pid] for pid,info in remaining):
        time.sleep(1);signal_owned(known,signal.SIGKILL)
    try:process.wait(timeout=5)
    except subprocess.TimeoutExpired:pass


def run(args):
    receipt=Path(args.receipt)
    receipt.parent.mkdir(parents=True,exist_ok=True)
    # Never replace an earlier failed attempt's resource evidence.
    with receipt.open('x') as stream:stream.write('{}\n')
    result={'schema_version':1,'status':'checking','eligible':False,'guard_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'wall_cap_s':args.timeout,'conflict_checks':0,'external_processes_observed':0,'initial_samples':[]}
    started=time.monotonic();process=None;known={};cancel=[]
    def interrupt(signum,frame):cancel.append(signum)
    previous={sig:signal.signal(sig,interrupt) for sig in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    code=75
    try:
        for i in range(args.ready_samples):
            if i:time.sleep(args.interval)
            result['initial_samples'].append(snapshot())
        uuid=choose_idle(result['initial_samples'],args.gpu,args.expected_model)
        result['gpu']=result['initial_samples'][-1][uuid]
        existing=os.environ.get('CUDA_VISIBLE_DEVICES')
        if existing is not None:
            raise ResourceUnavailable('pre-existing CUDA_VISIBLE_DEVICES allocation: do not override it')
        if args.select_only:
            result.update(status='selected_idle_not_reserved',eligible=False);code=0
        else:
            command=args.command[1:] if args.command[:1]==['--'] else args.command
            if not command:raise ValueError('an explicit bounded command is required')
            # Fresh observation immediately before Popen closes the preparation
            # interval; monitoring remains necessary on a shared server.
            if not idle(snapshot()[uuid]):raise ResourceUnavailable('GPU became busy before launch')
            if cancel:raise ResourceUnavailable('cancelled before launch')
            process=subprocess.Popen(command,cwd=args.cwd,env={**os.environ,'CUDA_VISIBLE_DEVICES':uuid},start_new_session=True)
            identity=proc_identity(process.pid)
            if identity is not None:known[process.pid]=identity[1]
            result.update(status='running',command_pid=process.pid,process_start_ticks=known.get(process.pid))
            receipt.write_text(json.dumps(result,indent=2)+'\n')
            while process.poll() is None:
                collect_owned(process.pid,known)
                state=snapshot()[uuid]
                foreign=foreign_pids(state['pids'],process.pid,known.get(process.pid),known)
                result['conflict_checks']+=1
                if foreign:
                    result['external_processes_observed']+=len(foreign)
                    raise ResourceUnavailable('another task appeared; stopping only this wrapper\'s command')
                if cancel:raise ResourceUnavailable('wrapper interrupted; stopping our command')
                if time.monotonic()-started>=args.timeout:
                    result['status']='timeout';code=124;break
                time.sleep(args.interval)
            else:
                code=process.returncode
                result.update(status='passed' if code==0 else 'command_failed',eligible=code==0)
    except (ResourceUnavailable,subprocess.TimeoutExpired,OSError,ValueError,KeyError) as exc:
        result.update(status='resource_unavailable' if process is None else 'aborted',reason=str(exc));code=75
    finally:
        if process is not None:
            if code or process.poll() is None:stop_owned(process,known)
            else:
                # A normally exited supervisor must not leave our workers alive.
                signal_owned(known,signal.SIGTERM)
        result.update(exit_code=code,wall_s=time.monotonic()-started)
        receipt.write_text(json.dumps(result,indent=2)+'\n')
        for sig,handler in previous.items():signal.signal(sig,handler)
    print(json.dumps({'status':result['status'],'exit_code':code,'eligible':result['eligible']}))
    return code


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu',default='auto',help='physical UUID or auto; never a global reservation')
    parser.add_argument('--expected-model',default='')
    parser.add_argument('--receipt',required=True,help='new private JSON; existing attempts are not overwritten')
    parser.add_argument('--cwd');parser.add_argument('--timeout',type=float,required=True)
    parser.add_argument('--interval',type=float,default=2);parser.add_argument('--ready-samples',type=int,default=3)
    parser.add_argument('--select-only',action='store_true');parser.add_argument('command',nargs=argparse.REMAINDER)
    args=parser.parse_args()
    if args.timeout<=0 or args.interval<=0 or args.ready_samples<2:parser.error('positive deadline/interval and at least two idle samples required')
    raise SystemExit(run(args))
