"""Trusted-code execution hygiene and Linux filesystem restriction, not a VM sandbox."""
from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess

ENV_NAMES = {'LANG', 'LC_ALL', 'LD_LIBRARY_PATH', 'CUDA_HOME', 'CUDA_PATH', 'CXX', 'CC',
    'TORCH_CUDA_ARCH_LIST', 'MAX_JOBS', 'KT_FLUTE_BUILD_MANIFEST', 'KT_VLLM085_MOE_COMPONENTS',
    'VLLM_TUNED_CONFIG_FOLDER', 'VLLM_BATCH_INVARIANT', 'VLLM_TRITON_USE_TD', 'PYTHONPATH',
    'KT_GEMLITE_RAW_OVERLAY'}

SYSTEM_READ_PATHS = ('/usr', '/bin', '/sbin', '/lib', '/lib64', '/sys', '/etc/ld.so.cache', '/etc/ld.so.conf',
    '/etc/ld.so.conf.d', '/etc/passwd', '/etc/group', '/etc/nsswitch.conf', '/etc/localtime', '/etc/os-release',
    '/etc/machine-id', '/etc/hosts', '/etc/resolv.conf', '/etc/ssl/certs', '/etc/fonts', '/etc/alternatives')


def clean_environment(directory, environment, *, cpu_only=False):
    directory = Path(directory)
    for name in ('home', 'tmp', 'cache', 'cache/triton', 'cache/cuda', 'cache/inductor'):
        (directory / name).mkdir(parents=True, exist_ok=True, mode=0o700)
    env = {'PATH': str(Path(environment['python']).parent) + ':/usr/local/bin:/usr/bin:/bin',
           'HOME': str(directory / 'home'), 'TMPDIR': str(directory / 'tmp'),
           'XDG_CACHE_HOME': str(directory / 'cache'), 'TRITON_CACHE_DIR': str(directory / 'cache/triton'),
           'CUDA_CACHE_PATH': str(directory / 'cache/cuda'), 'TORCHINDUCTOR_CACHE_DIR': str(directory / 'cache/inductor'),
           'PYTHONNOUSERSITE': '1', 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONHASHSEED': '0',
           'OMP_NUM_THREADS': '1', 'OPENBLAS_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'NUMEXPR_NUM_THREADS': '1', 'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1',
           'LANG': 'C.UTF-8', 'CUDA_VISIBLE_DEVICES': ''}
    for key, value in environment.get('env', {}).items():
        if key not in ENV_NAMES or not isinstance(value, str):
            raise ValueError('environment variable is not explicitly supported: ' + key)
        env[key] = value
    if cpu_only:
        env['CUDA_VISIBLE_DEVICES'] = ''
    return env


def landlock_abi():
    if os.name != 'posix' or not Path('/proc').exists():
        return 0
    libc = ctypes.CDLL(None, use_errno=True)
    return max(0, libc.syscall(444, 0, 0, 1))


def protect_supervisor():
    """Keep a child of this service from reading its parent /proc environment."""
    if os.name == 'posix' and Path('/proc').exists():
        libc = ctypes.CDLL(None, use_errno=True)
        return libc.prctl(4, 0, 0, 0, 0) == 0  # PR_SET_DUMPABLE; process-local only
    return False


def namespace_ids():
    return {name: os.readlink('/proc/self/ns/' + name) for name in ('user', 'mnt', 'pid', 'ipc', 'uts', 'net')}


@lru_cache(maxsize=1)
def restriction_capabilities():
    abi = landlock_abi()
    binary = shutil.which('bwrap', path='/usr/bin:/bin')
    result = {'landlock_ABI': abi, 'bubblewrap': None, 'uploaded_code_available': bool(abi)}
    if binary:
        args = [binary, '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts', '--unshare-net',
                '--cap-drop', 'ALL']
        for path in ('/usr', '/bin', '/lib', '/lib64'):
            if Path(path).exists():
                args += ['--ro-bind', path, path]
        args += ['--proc', '/proc', '--dev', '/dev', '--', '/usr/bin/true']
        try:
            probe = subprocess.run(args, capture_output=True, text=True, timeout=5)
            if probe.returncode == 0:
                result['bubblewrap'] = {'binary': binary, 'sha256': hashlib.sha256(Path(binary).read_bytes()).hexdigest()}
                result['uploaded_code_available'] = True
            else:
                result['bubblewrap_error'] = probe.stderr[-500:]
        except (OSError, subprocess.TimeoutExpired) as exc:
            result['bubblewrap_error'] = str(exc)
    return result


def nvidia_device_minor(gpu, information_root=Path('/proc/driver/nvidia/gpus')):
    """Bind by the driver's physical UUID, never by a CUDA/NVML display ordinal."""
    matches = set()
    for file in Path(information_root).glob('*/information'):
        values = dict(line.split(':', 1) for line in file.read_text().splitlines() if ':' in line)
        values = {key.strip(): value.strip() for key, value in values.items()}
        if values.get('GPU UUID', '').lower() == gpu.lower():
            minor = values.get('Device Minor', '')
            if minor.isdigit():
                matches.add(int(minor))
    if len(matches) != 1:
        raise RuntimeError('cannot resolve one physical NVIDIA device minor from the selected UUID')
    return matches.pop()


def bubblewrap_command(python_command, directory, environment, *, source, output, runtime, gpu=None):
    capability = restriction_capabilities().get('bubblewrap')
    if not capability:
        raise RuntimeError('Bubblewrap user/mount/PID/network namespace probe failed')
    args = [capability['binary'], '--unshare-user', '--unshare-pid', '--unshare-ipc', '--unshare-uts',
            '--die-with-parent', '--new-session', '--cap-drop', 'ALL']
    if not environment.get('network_access', False):
        args.append('--unshare-net')
    # Only necessary system configuration is visible; SSH host keys and user homes are not mounted.
    read_paths = [*SYSTEM_READ_PATHS, runtime['prefix'], runtime['base_prefix'], source, *environment.get('read_paths', [])]
    for path in dict.fromkeys(str(Path(p)) for p in read_paths if Path(p).exists()):
        args += ['--ro-bind', path, path]
    args += ['--proc', '/proc', '--dev', '/dev', '--tmpfs', '/dev/shm']
    if gpu:
        minor = nvidia_device_minor(gpu)
        device = Path('/dev/nvidia' + str(minor))
        if not device.is_char_device():
            raise RuntimeError('selected physical GPU device node is not accessible for namespace binding')
        paths = [device, *[p for p in (Path('/dev/nvidiactl'), Path('/dev/nvidia-uvm'),
                 Path('/dev/nvidia-uvm-tools'), Path('/dev/nvidia-caps')) if p.exists()]]
        for path in paths:
            args += ['--dev-bind', str(path), str(path)]
    for path in (directory, output):
        args += ['--bind', str(path), str(path)]
    args += ['--chdir', str(directory), '--', *python_command]
    before = {'namespaces': namespace_ids(), 'bubblewrap': capability,
              'network_isolated': not environment.get('network_access', False)}
    Path(directory, 'namespace-before.json').write_text(json.dumps(before, indent=2))
    return args


def restrict_filesystem(read_paths, write_paths):
    abi = landlock_abi()
    if not abi:
        return {'mode': 'unavailable', 'ABI': 0}
    libc = ctypes.CDLL(None, use_errno=True)
    class Ruleset(ctypes.Structure):
        _fields_ = [('handled_access_fs', ctypes.c_uint64)]
    class Beneath(ctypes.Structure):
        _pack_ = 1
        _fields_ = [('allowed_access', ctypes.c_uint64), ('parent_fd', ctypes.c_int32)]
    handled = (1 << (15 if abi >= 3 else 14 if abi >= 2 else 13)) - 1
    attr = Ruleset(handled)
    fd = libc.syscall(444, ctypes.byref(attr), ctypes.sizeof(attr), 0)
    if fd < 0:
        raise OSError(ctypes.get_errno(), 'landlock_create_ruleset')
    admitted = []
    try:
        for paths, access in ((read_paths, 1 | 4 | 8), (write_paths, handled)):
            for path in dict.fromkeys(str(Path(p).resolve()) for p in paths if Path(p).exists()):
                parent = os.open(path, os.O_PATH | os.O_CLOEXEC)
                try:
                    allowed = access if Path(path).is_dir() else access & (1 | 2 | 4 | (1 << 14))
                    rule = Beneath(allowed, parent)
                    if libc.syscall(445, fd, 1, ctypes.byref(rule), 0) < 0:
                        raise OSError(ctypes.get_errno(), 'landlock_add_rule: ' + path)
                    admitted.append(path)
                finally:
                    os.close(parent)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.syscall(446, fd, 0) != 0:
            raise OSError(ctypes.get_errno(), 'landlock_restrict_self')
    finally:
        os.close(fd)
    return {'mode': 'landlock', 'ABI': abi, 'read_paths': list(map(str, read_paths)),
            'write_paths': list(map(str, write_paths)),
            'scope': 'filesystem restriction; network/process isolation is not provided'}


def apply_limits(directory, environment, *, source, uploaded, hard_wall_s, output=None, namespace_inner=False):
    import sys
    os.umask(0o077)
    limits = environment.get('limits', {})
    for key, value in ((resource.RLIMIT_NOFILE, limits.get('open_files', 1024)),
                       (resource.RLIMIT_FSIZE, limits.get('file_bytes', 2 * 1024**3)),
                       (resource.RLIMIT_CPU, max(10, int(hard_wall_s) + 5))):
        old = resource.getrlimit(key)
        upper = value if old[1] == resource.RLIM_INFINITY else min(value, old[1])
        resource.setrlimit(key, (upper, upper))
    memory = limits.get('memory_bytes', 4 * 1024**3 if environment['device'] == 'cpu' else None)
    if memory:
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
    reads = [*SYSTEM_READ_PATHS, '/proc',
             sys.prefix, sys.base_prefix, source, *environment.get('read_paths', [])]
    writes = [directory, '/dev', *([output] if output else [])]
    if namespace_inner:
        before = json.loads(Path(directory, 'namespace-before.json').read_text())
        current = namespace_ids()
        required = ['user', 'mnt', 'pid', 'ipc', 'uts'] + (['net'] if before['network_isolated'] else [])
        if any(current[name] == before['namespaces'][name] for name in required):
            raise RuntimeError('Worker namespace separation was not established')
        result = {'mode': 'bubblewrap', 'binary_sha256': before['bubblewrap']['sha256'],
                  'namespaces': current, 'network_isolated': before['network_isolated'],
                  'scope': 'private filesystem/PID/IPC/UTS and configured network namespaces; selected GPU device access'}
    else:
        result = restrict_filesystem(reads, writes)
    if uploaded and environment.get('require_filesystem_restriction', True) and result['mode'] not in ('landlock', 'bubblewrap'):
        raise RuntimeError('this server cannot restrict uploaded-code filesystem access; use a supported Linux Worker')
    return {**result, 'credentials_forwarded': False, 'private_HOME': str(Path(directory) / 'home'),
            'limits': {'open_files': limits.get('open_files', 1024), 'memory_bytes': memory}}
