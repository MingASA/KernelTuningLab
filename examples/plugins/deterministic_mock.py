"""Example local plugin; customize an adapter factory without a compiler SDK."""

from functools import partial

from kernel_tuning_lab.backends import MockAdapter, register_adapter


def register_plugin():
    register_adapter("deterministic_mock", partial(MockAdapter, compile_failure_rate=0,
                                                   correctness_failure_rate=0, runtime_failure_rate=0), replace=True)
