//bpTypeConstraints.h
//
//Define constraints for data type in belief propagation processing

#ifndef BP_TYPE_CONSTRAINTS_H_
#define BP_TYPE_CONSTRAINTS_H_

//constraint for data type when smoothing images
template <typename T>
concept imData_t = std::is_same_v<T, float> || std::is_same_v<T, unsigned int>;

//define concepts of allowed data types for belief propagation data storage and processing
#if defined(OPTIMIZED_CUDA_RUN)

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

#elif defined(OPTIMIZED_CPU_RUN)
#if defined(COMPILING_FOR_ARM)

#include <arm_neon.h>
//float16_t is used for half data type in ARM processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, float16_t>;

//SIMD types for neon processing on ARM
template <typename T>
concept BpDataVect_t = std::is_same_v<T, float64x2_t> || std::is_same_v<T, float32x4_t> || std::is_same_v<T, float16x4_t>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, float64x2_t> || std::is_same_v<T, float32x4_t>;
#else //COMPILING_FOR_ARM

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

//short is used for half data type in x86 processing
template <typename T>
concept BpData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, short>;

//data processing on CPU only uses float or double
//half type gets converted to float for processing and then back to half for storage
template <typename T>
concept BpDataProcess_t = std::is_same_v<T, float> || std::is_same_v<T, double>;

//SIMD types for AVX processing on x86
#if defined(AVX_512_VECTORIZATION)

template <typename T>
concept BpDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512> || std::is_same_v<T, __m256i>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512>;

#elif defined(AVX_256_VECTORIZATION)

template <typename T>
concept BpDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256>;

#else

//AVX 512 used by default if AVX vectorization setting not specified
template <typename T>
concept BpDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512> || std::is_same_v<T, __m256i>;

template <typename T>
concept BpDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512>;


#endif //defined(AVX_512_VECTORIZATION)
#endif //defined(COMPILING_FOR_ARM)

//concepts that allow both single and vectorized types
template <typename T>
concept BpDataSingOrVect_t = BpData_t<T> || BpDataVect_t<T>;

template <typename T>
concept BpDataProcessSingOrVect_t = BpDataProcess_t<T> || BpDataVectProcess_t<T>;

#endif //defined(OPTIMIZED_CUDA_RUN)

#endif //BP_TYPE_CONSTRAINTS_H_