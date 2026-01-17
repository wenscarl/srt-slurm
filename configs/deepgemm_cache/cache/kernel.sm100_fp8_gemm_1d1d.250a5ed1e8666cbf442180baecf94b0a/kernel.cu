
#include <deep_gemm/impls/sm100_fp8_gemm_1d1d.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&sm100_fp8_gemm_1d1d_impl<
        cute::UMMA::Major::K, cute::UMMA::Major::K,
        0, 7168, 2048,
        128, 224, 128,
        1,
        128, 128, 64,
        4,
        128, 128,
        1, false,
        133,
        GemmType::Normal, false, cutlass::bfloat16_t,
        EpilogueIdentity
    >);
};
