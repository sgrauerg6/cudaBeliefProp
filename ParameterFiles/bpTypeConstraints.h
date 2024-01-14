//bpTypeConstraints.h
//
//Define constraints for data type in belief propagation processing

#ifndef BP_TYPE_CONSTRAINTS_H_
#define BP_TYPE_CONSTRAINTS_H_

//define concepts of allowed data types for belief propagation data storage and processing
#ifdef OPTIMIZED_CUDA_RUN
//set data type used for half-precision with CUDA
#ifdef USE_BFLOAT16_FOR_HALF_PRECISION
#include <cuda_bf16.h>
using halftype = __nv_bfloat16;
#else
#include <cuda_fp16.h>
using halftype = half;
#endif //USE_BFLOAT16_FOR_HALF_PRECISION
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, halftype>;
#else //OPTIMIZED_CPU_RUN
#ifdef COMPILING_FOR_ARM
//float16_t is used for half data type in ARM processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, float16_t>;
#else
//short is used for half data type in x86 processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, short>;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CUDA_RUN

#endif //BP_TYPE_CONSTRAINTS_H_