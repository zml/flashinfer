#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "src/flashinfer_ops.cuh"

#include "capi.h"

__attribute__ ((visibility("default")))
void flashinfer_BatchDecodeHandlerPlan(
    flashinfer_BatchDecodeHandler* handler,
    void* float_buffer,
    size_t float_workspace_size_in_bytes,
    void* int_buffer,
    size_t int_workspace_size_in_bytes,
    int32_t* indptr_h,
    int32_t* last_page_len_h,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size
) {
    using DTypeQ = __nv_bfloat16;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;
    using IdType = int32_t;

    static_assert(sizeof(flashinfer::BatchDecodeHandler) == sizeof(flashinfer_BatchDecodeHandler), "mismatch");

    flashinfer::BatchDecodeHandlerPlan<DTypeQ, DTypeKV, DTypeO, IdType>(
        reinterpret_cast<flashinfer::BatchDecodeHandler*>(handler),
        float_buffer,
        float_workspace_size_in_bytes,
        int_buffer,
        int_workspace_size_in_bytes,
        indptr_h,
        last_page_len_h,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size,
        flashinfer::PosEncodingMode::kRoPELlama);
}

__attribute__ ((visibility("default")))
void flashinfer_BatchDecodeWithPagedKVCacheWrapper(
    flashinfer_BatchDecodeHandler* handler,
    void* q,
    int32_t* q_rope_offset,
    flashinfer_paged_kv_t paged_kv,
    void* o,
    float* lse,
    uint32_t num_qo_heads,
    float sm_scale,
    float rope_scale,
    float rope_theta,
    void* stream
) {
    using DTypeQ = __nv_bfloat16;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;
    using IdType = int32_t;

    // cudaError_t status =
    flashinfer::BatchDecodeWithPagedKVCacheWrapper<DTypeQ, DTypeKV, DTypeO, IdType>(
        reinterpret_cast<flashinfer::BatchDecodeHandler*>(handler),
        static_cast<DTypeQ*>(q),
        q_rope_offset,
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
        num_qo_heads,
        flashinfer::PosEncodingMode::kRoPELlama);
}

__attribute__ ((visibility("default")))
void flashinfer_BatchPrefillHandlerPlan(
    flashinfer_BatchPrefillHandler* handler,
    void* float_buffer,
    size_t float_workspace_size_in_bytes,
    void* int_buffer,
    size_t int_workspace_size_in_bytes,
    int32_t* qo_indptr_h,
    int32_t* kv_indptr_h,
    uint32_t total_num_rows,
    uint32_t batch_size,
    uint32_t num_qo_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t page_size
) {
    using DTypeQ = __nv_bfloat16;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;
    using IdType = int32_t;

    static_assert(sizeof(flashinfer::BatchPrefillHandler) == sizeof(flashinfer_BatchPrefillHandler), "mismatch");

    reinterpret_cast<flashinfer::BatchPrefillHandler*>(handler)->Plan<DTypeQ, IdType>(
        float_buffer,
        float_workspace_size_in_bytes,
        int_buffer,
        int_workspace_size_in_bytes,
        qo_indptr_h,
        kv_indptr_h,
        total_num_rows,
        batch_size,
        num_qo_heads,
        num_kv_heads,
        head_dim,
        page_size);
}

__attribute__ ((visibility("default")))
void flashinfer_BatchPrefillWithPagedKVCacheWrapper(
    flashinfer_BatchPrefillHandler* handler,
    void* q,
    int32_t* qo_indptr,
    int32_t* q_rope_offset,
    flashinfer_paged_kv_t paged_kv,
    void* o,
    float* lse,
    uint32_t num_qo_heads,
    bool causal,
    float sm_scale,
    float rope_scale,
    float rope_theta,
    void* stream
) {
    using DTypeQ = __nv_bfloat16;
    using DTypeKV = DTypeQ;
    using DTypeO = DTypeQ;
    using IdType = int32_t;

    static_assert(sizeof(flashinfer::BatchPrefillHandler) == sizeof(flashinfer_BatchPrefillHandler), "mismatch");

    flashinfer::BatchPrefillWithPagedKVCacheWrapper<DTypeQ, DTypeKV, DTypeO, IdType>(
        reinterpret_cast<flashinfer::BatchPrefillHandler*>(handler),
        static_cast<DTypeQ*>(q),
        qo_indptr,
        q_rope_offset,
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
        num_qo_heads,
        causal,
        flashinfer::PosEncodingMode::kRoPELlama,
        false);

}
