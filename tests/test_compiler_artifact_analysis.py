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
