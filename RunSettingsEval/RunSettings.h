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

//by default, optimized GPU memory management and optimized indexing used
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these
//optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work)
//by setting USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT to false
//Optimized indexing can be turned off by changing the OPTIMIZED_INDEXING_SETTING value to false
//(not recommended; this slows down processing)
constexpr bool USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT{true};
constexpr bool OPTIMIZED_INDEXING_SETTING{true};
constexpr bool ALLOCATE_FREE_BP_MEMORY_OUTSIDE_RUNS{true};

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
inline void writeRunSettingsToStream(std::ostream& resultsStream)
{
  resultsStream << "Memory Optimization Level: " << USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT << "\n";
  resultsStream << "Indexing Optimization Level: " << OPTIMIZED_INDEXING_SETTING << "\n";
  resultsStream << "BYTES_ALIGN_MEMORY: " << getBytesAlignMemory(ACCELERATION_SETTING) << "\n";
  resultsStream << "NUM_DATA_ALIGN_WIDTH: " << getNumDataAlignWidth(ACCELERATION_SETTING) << "\n";
}

template <AccSetting ACCELERATION_SETTING>
inline RunData runSettings()  {
  RunData currRunData;
  currRunData.addDataWHeader("Memory Optimization Level", std::to_string(USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT));
  currRunData.addDataWHeader("Indexing Optimization Level", std::to_string(OPTIMIZED_INDEXING_SETTING));
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
};

};

#endif /* RUNSETTINGS_H_ */
