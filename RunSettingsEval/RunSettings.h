/*
 * RunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef RUNSETTINGS_H_
#define RUNSETTINGS_H_

#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <map>
#include <optional>
#include <string_view>
#include <thread>
#include "RunData.h"

//check if running on ARM architecture
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//define and set CPU vectorization options using preprocessor (since needed to determine what code gets compiled to support vectorization)
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

namespace run_environment {

//mapping from data size to data type string
const std::map<std::size_t, std::string> DATA_SIZE_TO_NAME_MAP{
  {sizeof(float), "FLOAT"}, {sizeof(double), "DOUBLE"}, {sizeof(short), "HALF"}
};

//enum for acceleration setting
enum class AccSetting {
  NONE, AVX256, AVX512, NEON, CUDA
};

//parallel parameter options to run to retrieve optimized parallel parameters in optimized CPU implementation
//parallel parameter corresponds to number of OpenMP threads in optimized CPU implementation
const unsigned int NUM_THREADS_CPU{std::thread::hardware_concurrency()};
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{
  { NUM_THREADS_CPU, 1}, { (3 * NUM_THREADS_CPU) / 4 , 1}, { NUM_THREADS_CPU / 2, 1}/*,
  { NUM_THREADS_CPU / 4, 1}, { NUM_THREADS_CPU / 8, 1}*/};
const std::array<unsigned int, 2> PARALLEL_PARAMS_DEFAULT{{NUM_THREADS_CPU, 1}};

//get string corresponding to CPU parallelization method
//currently only OpenMP CPU parallelization supported
constexpr const char* cpuParallelizationString() {
  return "OPEN_MP";
}

//get string corresponding to acceleration method
template <AccSetting ACCELERATION_SETTING>
constexpr const char* accelerationString() {
  if constexpr (ACCELERATION_SETTING == AccSetting::NEON)
    return "NEON";
  else if constexpr (ACCELERATION_SETTING == AccSetting::AVX256)
    return "AVX256";
  else if constexpr (ACCELERATION_SETTING == AccSetting::AVX512)
    return "AVX512";
  else if constexpr (ACCELERATION_SETTING == AccSetting::CUDA)
    return "CUDA";
  else
    return "NO_VECTORIZATION";
}

inline unsigned int getBytesAlignMemory(AccSetting accelSetting) {
  //avx512 requires data to be aligned on 64 bytes
  return (accelSetting == AccSetting::AVX512) ? 64 : 16;
}

inline unsigned int getNumDataAlignWidth(AccSetting accelSetting) {
  //align width with 16 data values in AVX512
  return (accelSetting == AccSetting::AVX512) ? 16 : 8;
}

template <AccSetting ACCELERATION_SETTING>
inline RunData runSettings()  {
  RunData currRunData;
  currRunData.addDataWHeader("BYTES_ALIGN_MEMORY", std::to_string(getBytesAlignMemory(ACCELERATION_SETTING)));
  currRunData.addDataWHeader("NUM_DATA_ALIGN_WIDTH", std::to_string(getNumDataAlignWidth(ACCELERATION_SETTING)));
  return currRunData;
}

enum class TemplatedItersSetting {
  RUN_ONLY_TEMPLATED,
  RUN_ONLY_NON_TEMPLATED,
  RUN_TEMPLATED_AND_NOT_TEMPLATED
};

//enum to specify if optimizing parallel parameters per kernel or using same parallel parameters across all kernels in run
//in initial testing optimizing per kernel is faster on GPU and using same parallel parameters across all kernels is faster
//on CPU
enum class OptParallelParamsSetting { SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN, ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN };

struct RunImpSettings {
  TemplatedItersSetting templatedItersSetting_;
  std::pair<bool, OptParallelParamsSetting> optParallelParmsOptionSetting_;
  std::pair<std::array<unsigned int, 2>, std::vector<std::array<unsigned int, 2>>> pParamsDefaultOptOptions_;
  std::optional<std::string> processorName_;
  std::optional<std::array<std::string_view, 2>> baselineRunDataPathsOptSingThread_;
};

};

#endif /* RUNSETTINGS_H_ */
