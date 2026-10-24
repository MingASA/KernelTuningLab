"""A scoped submitter assertion, deliberately distinct from platform evidence."""

from dataclasses import replace

from kernel_tuning_lab.sdk import declare_correctness, get_catalog


catalog = get_catalog()
base = catalog.resolve("kernel", "python/gemm@1.0.0")
operator = catalog.resolve("operator", "gemm@1.0.0")


def build(ctx, tile_size):
    return base.build(ctx, tile_size=tile_size)


kernel = replace(base, id="demo/claimed_gemm", build=build, claims=(),
                 metadata={"delegated_kernel_fingerprint": base.fingerprint})
claim = declare_correctness(
    kernel, operator, id="tutorial-author-assertion-v1", source="checked-in tutorial contributor",
    evidence=("Demonstration submitter assertion; not independent physical GPU test evidence",),
    scope={"runtimes": ["python"], "dtypes": ["float32"], "architectures": ["cpu"],
           "shapes": {"M": {"min": 1, "max": 4}, "N": {"min": 1, "max": 8}, "K": {"min": 1, "max": 8}},
           "configs": "*", "parameters": {}},
)
ASSETS = (replace(kernel, claims=(claim,)),)
