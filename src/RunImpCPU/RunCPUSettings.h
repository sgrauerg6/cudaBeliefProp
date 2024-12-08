/*
 * RunCPUSettings.h
 *
 *  Created on: Jan 27, 2024
 *      Author: scott
 */

#ifndef RUNCPUSETTINGS_H_
#define RUNCPUSETTINGS_H_

#include <vector>
#include <array>
#include <thread>
#include "RunSettingsParams/RunSettings.h"

//check if running on ARM architecture
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//define and set CPU vectorization options using preprocessor
//needed to determine what code gets compiled to support vectorization
#define AVX_256_DEFINE 0
#define AVX_512_DEFINE 1
#define NEON_DEFINE 2
#define NO_VECTORIZATION 3
#ifdef COMPILING_FOR_ARM //NEON supported on ARM but AVX is not
#define CPU_VECTORIZATION_DEFINE NEON_DEFINE
#else
//by default CPU vectorization during compilation via Makefile
//use AVX 512 if not set during compilation
#if defined(AVX_512_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#elif defined(AVX_256_VECTORIZATION)
#define CPU_VECTORIZATION_DEFINE AVX_256_DEFINE
#else
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#endif //defined(AVX_512_VECTORIZATION)
#endif //COMPILING_FOR_ARM

/**
 * @brief Namespace with CPU run defaults and constants.
 */
namespace run_cpu {

/** @brief Constant that specifies that run is simulating single CPU on a dual-CPU system */
constexpr std::string_view kSimulateSingleCPU{"SimulateSingleCPU"};

/** @brief Constant corresponding to number of threads on CPU. */
const unsigned int kNumThreadsCPU{std::thread::hardware_concurrency()};
#ifdef LIMITED_TEST_PARAMS_FEWER_RUNS

/** @brief Parallel parameters options that are tested in order to find optimized
 *  configuration in run. */
const std::vector<std::array<unsigned int, 2>> kParallelParameterOptions{
  { kNumThreadsCPU, 1},{ kNumThreadsCPU / 2, 1}};

#else

/** @brief Parallel parameters options that are tested in order to find optimized
 *  configuration in run. */
const std::vector<std::array<unsigned int, 2>> kParallelParameterOptions{
  { kNumThreadsCPU, 1}, { (3 * kNumThreadsCPU) / 4 , 1}, { kNumThreadsCPU / 2, 1},
  { kNumThreadsCPU / 4, 1}, { kNumThreadsCPU / 8, 1}};

#endif //LIMITED_TEST_PARAMS_FEWER_RUNS

/** @brief Minimum number of threads to allow for any parallel parameters
 *  setting on CPU. */
const unsigned int kMinNumThreadsRun{std::min(kNumThreadsCPU, 4u)};

/** @brief Default parallel parameters setting on CPU. */
const std::array<unsigned int, 2> kParallelParamsDefault{{kNumThreadsCPU, 1}};

};

#endif /* RUNCPUSETTINGS_H_ */
