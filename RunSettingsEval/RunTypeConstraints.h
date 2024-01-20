//RunTypeConstraints.h
//
//Define constraints for data type in belief propagation processing

#ifndef RUN_TYPE_CONSTRAINTS_H_
#define RUN_TYPE_CONSTRAINTS_H_

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
concept RunData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, halftype>;

#elif defined(OPTIMIZED_CPU_RUN)
#if defined(COMPILING_FOR_ARM)

#include <arm_neon.h>
//float16_t is used for half data type in ARM processing
template <typename T>
concept RunData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, float16_t>;

//SIMD types for neon processing on ARM
template <typename T>
concept RunDataVect_t = std::is_same_v<T, float64x2_t> || std::is_same_v<T, float32x4_t> || std::is_same_v<T, float16x4_t>;

template <typename T>
concept RunDataVectProcess_t = std::is_same_v<T, float64x2_t> || std::is_same_v<T, float32x4_t>;
#else //COMPILING_FOR_ARM

#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

//short is used for half data type in x86 processing
template <typename T>
concept RunData_t = std::is_same_v<T, float> || std::is_same_v<T, double> || std::is_same_v<T, short>;

//data processing on CPU only uses float or double
//half type gets converted to float for processing and then back to half for storage
template <typename T>
concept RunDataProcess_t = std::is_same_v<T, float> || std::is_same_v<T, double>;

//SIMD types for AVX processing on x86
#if defined(AVX_512_VECTORIZATION)

template <typename T>
concept RunDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512> || std::is_same_v<T, __m256i>;

template <typename T>
concept RunDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512>;

#elif defined(AVX_256_VECTORIZATION)

template <typename T>
concept RunDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i>;

template <typename T>
concept RunDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256>;

#else

//AVX 512 used by default if AVX vectorization setting not specified
template <typename T>
concept RunDataVect_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m128i> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512> || std::is_same_v<T, __m256i>;

template <typename T>
concept RunDataVectProcess_t = std::is_same_v<T, __m256d> || std::is_same_v<T, __m256> || std::is_same_v<T, __m512d> || std::is_same_v<T, __m512>;


#endif //defined(AVX_512_VECTORIZATION)
#endif //defined(COMPILING_FOR_ARM)

//concepts that allow both single and vectorized types
template <typename T>
concept RunDataSingOrVect_t = RunData_t<T> || RunDataVect_t<T>;

template <typename T>
concept RunDataProcessSingOrVect_t = RunDataProcess_t<T> || RunDataVectProcess_t<T>;

#endif //defined(OPTIMIZED_CUDA_RUN)

//constraint for pointer to RunData_t
template <typename T>
concept RunData_ptr = std::is_pointer_v<T> && RunData_t<std::remove_pointer_t<T>>;

#endif //RUN_TYPE_CONSTRAINTS_H_