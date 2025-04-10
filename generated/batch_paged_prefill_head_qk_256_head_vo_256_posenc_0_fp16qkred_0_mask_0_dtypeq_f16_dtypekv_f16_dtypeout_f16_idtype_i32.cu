#include <flashinfer/attention_impl.cuh>

namespace flashinfer {

using Params = BatchPrefillPagedParams<half, half, half, int32_t>;

using AttentionVariant1 = DefaultAttention<false, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<128, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant1, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<64, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant1, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant1, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    

using AttentionVariant2 = DefaultAttention<false, /*use_sliding_window=*/true, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<128, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant2, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<64, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant2, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant2, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    

using AttentionVariant3 = DefaultAttention<false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/false, /*use_alibi_bias=*/false>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<128, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant3, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<64, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant3, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant3, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    

using AttentionVariant4 = DefaultAttention<false, /*use_sliding_window=*/false, /*use_logits_soft_cap=*/true, /*use_alibi_bias=*/false>;

template cudaError_t BatchPrefillWithPagedKVCacheDispatched<128, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant4, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<64, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant4, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    
template cudaError_t BatchPrefillWithPagedKVCacheDispatched<16, 256, 256, PosEncodingMode::kNone, 0, MaskMode::kNone, AttentionVariant4, Params>(
    Params params,
    half* tmp_v,
    float* tmp_s, cudaStream_t stream);
    

}