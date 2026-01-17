
#include <deep_gemm/impls/smxx_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {
    auto ptr = reinterpret_cast<void*>(&transpose_and_pack_fp32_into_ue8m0<
        512, 48, 56
    >);
};
