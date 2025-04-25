#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <flashinfer/cutlass_utils.cuh>

#include "src/flashinfer_ops_sm90.cuh"

#include "capi.h"

__attribute__ ((visibility("default")))
void flashinfer_BatchPrefillHandlerSm90Plan(
    flashinfer_BatchPrefillSm90Handler* handler,
    void* float_buffer,
    size_t float_workspace_size_in_bytes,
    void* int_buffer,
    size_t int_workspace_size_in_bytes,
    int32_t* qo_indptr_h,
    int32_t* kv_indptr_h,
    int32_t* kv_len_arr_h,
    uint32_t total_num_rows,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size
) {
    using DTypeQ = flashinfer::cutlass_dtype_t<__nv_bfloat16>;
    using IdType = int32_t;

    static_assert(sizeof(flashinfer::BatchPrefillSm90Handler) == sizeof(flashinfer_BatchPrefillSm90Handler), "mismatch");

    reinterpret_cast<flashinfer::BatchPrefillSm90Handler*>(handler)->Plan<DTypeQ, IdType>(
        float_buffer,
        float_workspace_size_in_bytes,
        int_buffer,
        int_workspace_size_in_bytes,
        qo_indptr_h,
        kv_indptr_h,
        kv_len_arr_h,
        total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size);
}

__attribute__ ((visibility("default")))
int flashinfer_BatchPrefillWithPagedKVCacheSm90Wrapper(
    flashinfer_BatchPrefillSm90Handler* handler,
    void* q,
    flashinfer_paged_kv_t paged_kv,
    void* o,
    float* lse,
    uint32_t nnz_qo,
    uint32_t num_qo_heads,
    bool causal,
    float sm_scale,
    float rope_scale,
    float rope_theta,
    void* stream
) {
    using DTypeQ = flashinfer::cutlass_dtype_t<__nv_bfloat16>;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;
    using IdType = int32_t;

    static_assert(sizeof(flashinfer::BatchPrefillSm90Handler) == sizeof(flashinfer_BatchPrefillSm90Handler), "mismatch");

    cudaError_t status = flashinfer::BatchPrefillWithPagedKVCacheSm90Wrapper<DTypeQ, DTypeKV, DTypeO, IdType>(
        reinterpret_cast<flashinfer::BatchPrefillSm90Handler*>(handler),
        static_cast<DTypeQ*>(q),
        flashinfer::paged_kv_t<DTypeKV, IdType>(
            paged_kv.num_heads,
            paged_kv.page_size,
            paged_kv.head_dim,
            paged_kv.batch_size,
            flashinfer::QKVLayout::kNHD,
            static_cast<DTypeKV*>(paged_kv.k_data),
            static_cast<DTypeKV*>(paged_kv.v_data),
            paged_kv.kv_strides,
            paged_kv.indices,
            paged_kv.indptr,
            paged_kv.last_page_len,
            paged_kv.rope_pos_offset
        ),
        static_cast<DTypeO*>(o),
        lse,
        nnz_qo,
        num_qo_heads,
        causal,
        flashinfer::PosEncodingMode::kNone,
        std::nullopt,
        1.f,
        1e4,
        reinterpret_cast<cudaStream_t>(stream));
    
    return (int)status;
}
