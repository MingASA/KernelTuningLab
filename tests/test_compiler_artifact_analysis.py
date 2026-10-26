"""Independent ELF section-size cross-check; no CUDA or synthetic GPU results."""
import importlib
from pathlib import Path
import re
import shutil
import subprocess
import sys
import pytest


def test_elf_text_sizes_match_system_readelf(monkeypatch):
    tool=shutil.which('readelf')
    if tool is None: pytest.skip('system readelf unavailable')
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('archive_compiled_phase1')
    blob=Path(sys.executable).read_bytes()
    sections=module.text_sections(blob)
    if sections is None: pytest.skip('Python executable is not ELF64')
    output=subprocess.check_output([tool,'-SW',sys.executable],text=True)
    expected={}
    for line in output.splitlines():
        match=re.match(r'\s*\[\s*\d+\]\s+(\.text(?:\.\S+)?)\s+\S+\s+[0-9a-fA-F]+\s+[0-9a-fA-F]+\s+([0-9a-fA-F]+)',line)
        if match: expected[match[1]]=int(match[2],16)
    assert expected and {s['name']:s['bytes'] for s in sections}==expected
    assert module.text_sections(b'\x7fELF') is None


def test_archive_cache_scan_falls_back_without_rg_and_does_not_follow_links(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('archive_compiled_phase1')
    cache=tmp_path/'own-cache';nested=cache/'.hidden-key';nested.mkdir(parents=True)
    binary=nested/'kernel.cubin';binary.write_bytes(b'unit fixture, not GPU code')
    (nested/'kernel.ptx').write_text('unit fixture')
    external=tmp_path/'unrelated';external.mkdir();(external/'other.cubin').write_bytes(b'do not inspect')
    (cache/'link').symlink_to(external,target_is_directory=True)
    (cache/'file-link.cubin').symlink_to(external/'other.cubin')
    def missing(*args,**kwargs):raise FileNotFoundError('rg unavailable')
    monkeypatch.setattr(module.subprocess,'run',missing)
    assert module.cache_binaries(cache)==[str(binary)]
    with pytest.raises(ValueError,match='missing'):module.cache_binaries(tmp_path/'absent')


def test_archive_cache_scan_reports_rg_errors_instead_of_empty_success(tmp_path,monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    module=importlib.import_module('archive_compiled_phase1')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:subprocess.CompletedProcess(a,2,'','denied'))
    with pytest.raises(ValueError,match='denied'):module.cache_binaries(tmp_path)
