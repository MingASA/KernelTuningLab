"""A declared Git HEAD must not hide a stale package imported via PYTHONPATH."""
import importlib.util
from pathlib import Path
import pytest
import kernel_tuning_lab

def test_numerical_worker_refuses_foreign_package_before_loading_a_gpu(tmp_path,monkeypatch):
    root=Path(__file__).parents[1];file=root/'examples/production_stability/check_numerics.py'
    spec=importlib.util.spec_from_file_location('stability_worker',file);worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
    worker.assert_runtime_checkout(root)
    foreign=tmp_path/'old_checkout/kernel_tuning_lab/__init__.py';foreign.parent.mkdir(parents=True);foreign.write_text('')
    monkeypatch.setattr(kernel_tuning_lab,'__file__',str(foreign))
    with pytest.raises(RuntimeError,match='refuse GPU work'):worker.assert_runtime_checkout(root)
