from pathlib import Path
import argparse

from aot_build_utils import generate_dispatch_inc
from aot_build_utils.generate import get_instantiation_cu
from aot_build_utils.generate_aot_default_additional_params_header import (
    get_aot_default_additional_params_header_str,
)
from aot_build_utils.generate_sm90 import get_sm90_instantiation_cu

def write_if_different(path: Path, content: str) -> None:
    if path.exists() and path.read_text() == content:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


gen_dir = Path("generated")
mask_modes = [0, 1, 2]
head_dims = [64, 128, 256]
head_dims_sm90 = [[64, 64], [128, 128], [256, 256]]
enable_f16 = True
enable_bf16 = True
enable_fp8_e4m3 = False
enable_fp8_e5m2 = False
pos_encoding_modes = [0, 1, 2]
use_fp16_qk_reductions = [0]

# dispatch.inc
write_if_different(
    gen_dir / "dispatch.inc",
    generate_dispatch_inc.get_dispatch_inc_str(
        argparse.Namespace(
            head_dims=head_dims,
            head_dims_sm90=head_dims_sm90,
            pos_encoding_modes=pos_encoding_modes,
            use_fp16_qk_reductions=use_fp16_qk_reductions,
            mask_modes=mask_modes,
        )
    ),
)

# _kernels
aot_kernel_uris = get_instantiation_cu(
    argparse.Namespace(
        path=gen_dir,
        head_dims=head_dims,
        pos_encoding_modes=pos_encoding_modes,
        use_fp16_qk_reductions=use_fp16_qk_reductions,
        mask_modes=mask_modes,
        enable_f16=enable_f16,
        enable_bf16=enable_bf16,
        enable_fp8_e4m3=enable_fp8_e4m3,
        enable_fp8_e5m2=enable_fp8_e5m2,
    )
)

aot_kernel_uris += get_sm90_instantiation_cu(
    argparse.Namespace(
        path=gen_dir,
        head_dims=head_dims_sm90,
        pos_encoding_modes=pos_encoding_modes,
        use_fp16_qk_reductions=use_fp16_qk_reductions,
        mask_modes=mask_modes,
        enable_f16=enable_f16,
        enable_bf16=enable_bf16,
    )
)

write_if_different(
    gen_dir / "aot_default_additional_params.h",
    get_aot_default_additional_params_header_str(),
)
