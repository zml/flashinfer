#include <stdlib.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t padded_batch_size;
    int64_t v_offset;
    int64_t s_offset;
    int64_t request_indices_offset;
    int64_t kv_tile_indices_offset;
    int64_t o_indptr_offset;
    int64_t block_valid_mask_offset;
    int64_t kv_chunk_size_ptr_offset;
    bool enable_cuda_graph;
    bool split_kv;
} flashinfer_DecodePlanInfo;

typedef struct {
    void* page_locked_buffer_;
    void* int_buffer_;
    void* float_buffer_;
    flashinfer_DecodePlanInfo plan_info_;
    bool cuda_graph_enabled_;
    void* stream_;
} flashinfer_BatchDecodeHandler;

typedef struct {
    int64_t padded_batch_size;
    int64_t total_num_rows;
    int64_t total_num_rows_offset;
    int64_t cta_tile_q;
    int64_t request_indices_offset;
    int64_t qo_tile_indices_offset;
    int64_t kv_tile_indices_offset;
    int64_t merge_indptr_offset;
    int64_t o_indptr_offset;
    int64_t kv_chunk_size_ptr_offset;
    int64_t v_offset;
    int64_t s_offset;
    int64_t block_valid_mask_offset;
    bool enable_cuda_graph;
    bool split_kv;
} flashinfer_PrefillPlanInfo;

typedef struct {
    void* page_locked_buffer_;
    void* int_buffer_;
    void* float_buffer_;
    flashinfer_PrefillPlanInfo plan_info_;
    bool enable_cuda_graph_;
    void* stream_;
} flashinfer_BatchPrefillHandler;

typedef struct {
    uint32_t num_heads;
    uint32_t page_size;
    uint32_t head_dim;
    uint32_t batch_size;
    void* k_data;
    void* v_data;
    const int64_t* kv_strides;
    int32_t* indices;
    int32_t* indptr;
    int32_t* last_page_len;
    int32_t* rope_pos_offset;
} flashinfer_paged_kv_t;

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
);

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
);

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
);

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
);

#ifdef __cplusplus
}
#endif
