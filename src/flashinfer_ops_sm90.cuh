/*
 * Copyright (c) 2024 by FlashInfer team.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/hopper/default_params.cuh>
#include <flashinfer/attention/hopper/variants.cuh>
#include <optional>

#include "flashinfer/allocator.h"
#include "flashinfer/attention/mask.cuh"
#include "flashinfer/attention/scheduler.cuh"
#include "flashinfer/exception.h"
#include "flashinfer/layout.cuh"
#include "flashinfer/page.cuh"
#include "utils.h"

namespace flashinfer {

class BatchPrefillSm90Handler {
 public:
  void UpdatePageLockedBufferSize(size_t int_workspace_size_in_bytes) {
    cudaFreeHost(page_locked_buffer_);
    cudaMallocHost(&page_locked_buffer_, int_workspace_size_in_bytes);
  }

  template <typename DTypeO, typename IdType>
  cudaError_t Plan(void* float_buffer, size_t float_workspace_size_in_bytes, void* int_buffer,
                   size_t int_workspace_size_in_bytes, IdType* qo_indptr_h, IdType* kv_indptr_h, IdType* kv_len_arr_h,
                   uint32_t total_num_rows, uint32_t batch_size, uint32_t num_qo_heads, uint32_t num_kv_heads, 
                   uint32_t head_dim, uint32_t page_size) {
    int_buffer_ = int_buffer;
    float_buffer_ = float_buffer;
    return PrefillSM90Plan<IdType>(float_buffer, float_workspace_size_in_bytes, int_buffer,
                               page_locked_buffer_, int_workspace_size_in_bytes, plan_info_,
                               qo_indptr_h, kv_indptr_h, kv_len_arr_h, total_num_rows, batch_size, num_qo_heads,
                               num_kv_heads, head_dim, head_dim, page_size, false, enable_cuda_graph_,
                               sizeof(DTypeO), stream_);
  }

  cudaStream_t GetCUDAStream() const { return stream_; }

  void SetCUDAStream(cudaStream_t stream) { stream_ = stream; }

  bool IsCUDAGraphEnabled() const { return enable_cuda_graph_; }

  BatchPrefillSm90Handler(bool enable_cuda_graph = false)
      : enable_cuda_graph_(enable_cuda_graph), stream_(nullptr) {
    cudaMallocHost(&page_locked_buffer_, 8 * 1024 * 1024);
  }
  ~BatchPrefillSm90Handler() { cudaFreeHost(page_locked_buffer_); }

  PrefillPlanSM90Info GetPlanInfo() const { return plan_info_; }

  template <typename IdType>
  IdType* GetQOTileIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.qo_tile_indices_offset);
  }

  template <typename IdType>
  IdType* GetQOIndptr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.qo_indptr_offset);
  }

  template <typename IdType>
  IdType* GetKVIndptr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_indptr_offset);
  }

  template <typename IdType>
  IdType* GetQOLens() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.qo_len_offset);
  }

  template <typename IdType>
  IdType* GetKVLens() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.kv_len_offset);
  }

  template <typename IdType>
  IdType* GetHeadIndices() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.head_indices_offset);
  }

  template <typename IdType>
  IdType* GetWorkIndptr() {
    return GetPtrFromBaseOffset<IdType>(int_buffer_, plan_info_.work_indptr_offset);
  }

  //template <typename DTypeO>
  //DTypeO* GetTmpV() {
  //  if (plan_info_.split_kv) {
  //    return GetPtrFromBaseOffset<DTypeO>(float_buffer_, plan_info_.v_offset);
  //  }
  //  return nullptr;
  //}

  //float* GetTmpS() {
  //  if (plan_info_.split_kv) {
  //    return GetPtrFromBaseOffset<float>(float_buffer_, plan_info_.s_offset);
  //  }
  //  return nullptr;
  //}

  //uint32_t* GetTotalNumRows() {
  //  if (plan_info_.enable_cuda_graph) {
  //    return GetPtrFromBaseOffset<uint32_t>(int_buffer_, plan_info_.total_num_rows_offset);
  //  }
  //  return nullptr;
  //}

  //bool* GetBlockValidMask() {
  //  if (plan_info_.split_kv && plan_info_.enable_cuda_graph) {
  //    return GetPtrFromBaseOffset<bool>(int_buffer_, plan_info_.block_valid_mask_offset);
  //  }
  //  return nullptr;
  //}

 protected:
  void* page_locked_buffer_;
  void* int_buffer_;
  void* float_buffer_;
  PrefillPlanSM90Info plan_info_;
  bool enable_cuda_graph_;
  cudaStream_t stream_;
};

#define DISPATCH_BOOL(expr, const_expr, ...) \
    if (expr) {                              \
      constexpr bool const_expr = true;      \
      __VA_ARGS__                            \
    } else {                                 \
      constexpr bool const_expr = false;     \
      __VA_ARGS__                            \
    }

template <uint32_t HEAD_DIM_QK, uint32_t HEAD_DIM_VO, MaskMode MASK_MODE, bool LEFT_SLIDING_WINDOW,
          bool SAME_SCHEDULE_FOR_ALL_HEADS, typename AttentionVariant, typename Params>
cudaError_t BatchPrefillWithPagedKVCacheDispatched(Params& params, cudaStream_t stream);

template <typename DTypeQ, typename DTypeKV, typename DTypeO, typename IdType>
cudaError_t BatchPrefillWithPagedKVCacheSm90Wrapper(
    BatchPrefillSm90Handler* handler, DTypeQ* q,
    paged_kv_t<DTypeKV, IdType> paged_kv, DTypeO* o, float* lse, uint32_t nnz_qo, uint32_t num_qo_heads,
    bool causal = true, PosEncodingMode pos_encoding_mode = PosEncodingMode::kNone, 
    std::optional<float> maybe_sm_scale = std::nullopt,
    float rope_scale = 1.f, float rope_theta = 1e4, cudaStream_t stream = nullptr) {
  const float sm_scale = maybe_sm_scale.value_or(1.f / std::sqrt(float(paged_kv.head_dim)));
  const uint32_t num_kv_heads = paged_kv.num_heads;
  const uint32_t head_dim = paged_kv.head_dim;
  const uint32_t head_dim_qk = paged_kv.head_dim;
  const MaskMode mask_mode = causal ? MaskMode::kCausal : MaskMode::kNone;
  const int32_t window_left = -1;
  const float logits_soft_cap = 0.0f;
  auto plan_info = handler->GetPlanInfo();


  DISPATCH_mask_mode(mask_mode, MASK_MODE, {                                                
    using RaggedParams = BatchPrefillRaggedSm90Params<DTypeQ, DTypeKV, DTypeO, IdType>;       
    using PagedParams = BatchPrefillPagedSm90Params<DTypeQ, DTypeKV, DTypeO, IdType>;         
    constexpr auto POS_ENCODING_MODE = PosEncodingMode::kNone;                            
    constexpr bool USE_FP16_QK_REDUCTION = false;                                         
    constexpr bool use_custom_mask = MASK_MODE == MaskMode::kCustom;                      
    DISPATCH_head_dim(head_dim_qk, HEAD_DIM_QK, {                              
      [[maybe_unused]] constexpr int HEAD_DIM_VO = HEAD_DIM_QK;                                           
      DISPATCH_BOOL(window_left > -1, USE_SLIDING_WINDOW, {                    
        DISPATCH_BOOL(logits_soft_cap > 0.f, USE_LOGITS_SOFT_CAP, {            
          PagedParams params;

          params.q_ptr = static_cast<DTypeQ*>(q);
          params.k_ptr = static_cast<DTypeKV*>(paged_kv.k_data);
          params.v_ptr = static_cast<DTypeKV*>(paged_kv.v_data);
          params.o_ptr = static_cast<DTypeO*>(o);
          params.lse_ptr = lse;
          params.q_stride_n = num_qo_heads * HEAD_DIM_QK;
          params.q_stride_h = HEAD_DIM_QK;
          params.o_stride_n = num_qo_heads * HEAD_DIM_VO;
          params.o_stride_h = HEAD_DIM_VO;
          params.k_stride_n = paged_kv.stride_n;
          params.k_stride_h = paged_kv.stride_h;
          params.v_stride_n = paged_kv.stride_n;
          params.v_stride_h = paged_kv.stride_h;
          params.nnz_qo = nnz_qo;
          params.num_qo_heads = num_qo_heads;
          params.num_kv_heads = num_kv_heads;
          params.group_size = params.num_qo_heads / num_kv_heads;
          params.page_size = paged_kv.page_size;
          params.window_left = window_left;
          params.causal = causal;
          params.qo_tile_indices = handler->GetQOTileIndices<IdType>();
          params.qo_indptr = handler->GetQOIndptr<IdType>();
          params.kv_indptr = handler->GetKVIndptr<IdType>();
          params.qo_lens = handler->GetQOLens<IdType>();
          params.kv_lens = handler->GetKVLens<IdType>();
          params.head_indices = handler->GetHeadIndices<IdType>();
          params.work_indptr = handler->GetWorkIndptr<IdType>();
          params.kv_indices = static_cast<IdType*>(paged_kv.indices);
          params.additional_params.logits_soft_cap = logits_soft_cap;
          params.additional_params.sm_scale = sm_scale;

          bool same_schedule_for_all_heads = plan_info.same_schedule_for_all_heads;
          DISPATCH_BOOL(same_schedule_for_all_heads, SAME_SCHEDULER_FOR_ALL_HEADS, {
            return BatchPrefillWithPagedKVCacheDispatched<
                HEAD_DIM_QK, HEAD_DIM_VO, MASK_MODE, USE_SLIDING_WINDOW, SAME_SCHEDULER_FOR_ALL_HEADS,
                StandardAttention>(params, stream);
          });
        });                                                                               
      });                                                                                 
    });                                                                                   
  });                                                                                           

  return cudaSuccess;
}

}  // namespace flashinfer
