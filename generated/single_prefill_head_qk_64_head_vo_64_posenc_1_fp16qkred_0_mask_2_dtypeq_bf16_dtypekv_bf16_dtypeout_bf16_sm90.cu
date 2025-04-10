 // single_prefill_sm90 template inst
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/prefill_sm90.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <flashinfer/cutlass_utils.cuh>

namespace flashinfer {

using DTypeQ = cutlass_dtype_t<nv_bfloat16>;
using DTypeKV = cutlass_dtype_t<nv_bfloat16>;
using DTypeO = cutlass_dtype_t<nv_bfloat16>;

using Params = SinglePrefillParams<DTypeQ, DTypeKV, DTypeO>;

template cudaError_t SinglePrefillWithKVCacheDispatched
    <64, 64, MaskMode::kCustom, /*USE_SLIDING_WINDOW=*/true, LogitsSoftCap, Params>
    (Params& params, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched
    <64, 64, MaskMode::kCustom, /*USE_SLIDING_WINDOW=*/false, LogitsSoftCap, Params>
    (Params& params, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched
    <64, 64, MaskMode::kCustom, /*USE_SLIDING_WINDOW=*/true, StandardAttention, Params>
    (Params& params, cudaStream_t stream);

template cudaError_t SinglePrefillWithKVCacheDispatched
    <64, 64, MaskMode::kCustom, /*USE_SLIDING_WINDOW=*/false, StandardAttention, Params>
    (Params& params, cudaStream_t stream);

}
    