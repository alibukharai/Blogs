#pragma once

#include <stdint.h>
#include "dl_constant.hpp"

namespace handrecognition_coefficient
{
    const dl::Filter<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_filter();
    const dl::Bias<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_bias();
    const dl::Activation<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_3_biasadd_activation();

    const dl::Filter<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_filter();
    const dl::Bias<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_bias();
    const dl::Activation<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_4_biasadd_activation();

    const dl::Filter<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_filter();
    const dl::Bias<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_bias();
    const dl::Activation<int16_t> *get_statefulpartitionedcall_sequential_1_conv2d_5_biasadd_activation();

    const dl::Filter<int16_t> *get_fused_gemm_0_filter();
    const dl::Bias<int16_t> *get_fused_gemm_0_bias();
    const dl::Activation<int16_t> *get_fused_gemm_0_activation();
    
    const dl::Filter<int16_t> *get_fused_gemm_1_filter();
    const dl::Bias<int16_t> *get_fused_gemm_1_bias();
}
