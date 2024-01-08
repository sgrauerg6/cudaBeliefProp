/*
 * bpRunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef BPRUNSETTINGS_H_
#define BPRUNSETTINGS_H_

#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <map>
#include "../OutputEvaluation/RunData.h"

//uncomment if compiling/running on ARM architecture
//#define COMPILING_FOR_ARM
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//comment out corresponding define if half or double precision
//not supported (or to not to process half or double precision) 
#define HALF_PRECISION_SUPPORTED
#define DOUBLE_PRECISION_SUPPORTED

#define DATA_TYPE_PROCESSING_FLOAT 0
#define DATA_TYPE_PROCESSING_DOUBLE 1
#define DATA_TYPE_PROCESSING_HALF 2
//not currently supporting half2 data type
#define DATA_TYPE_PROCESSING_HALF_TWO 3

//by default, 32-bit float data is used with optimized GPU memory management and optimized indexing
//See http://scottgg.net/OptimizingGlobalStereoMatchingOnNVIDIAGPUs.pdf for more info on these optimizations (note that the optimized indexing was present in the initial implementation)
//Can remove optimized GPU memory management (making the processing more similar to the initial work) by setting beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT to false
//May be able to speed up processing by switching to using 16-bit half data by setting CURRENT_DATA_TYPE_PROCESSING to DATA_TYPE_PROCESSING_HALF
//Optimized indexing can be turned off by changing the beliefprop::OPTIMIZED_INDEXING_SETTING value to false (not recommended; this slows down processing)
#define CURRENT_DATA_TYPE_PROCESSING DATA_TYPE_PROCESSING_FLOAT

//define and set CPU vectorization options using preprocessor (since needed to determine what code gets compiled to support vectorization)
#define AVX_256_DEFINE 0
#define AVX_512_DEFINE 1
#define NEON_DEFINE 2
#define NO_VECTORIZATION 3
#ifdef COMPILING_FOR_ARM //NEON supported on ARM but AVX is not
#define CPU_VECTORIZATION_DEFINE NEON_DEFINE
#else
#define CPU_VECTORIZATION_DEFINE AVX_512_DEFINE
#endif

namespace beliefprop {

//mapping from data size to data type string
const std::map<std::size_t, std::string> DATA_SIZE_TO_NAME_MAP{
  {sizeof(float), "FLOAT"}, {sizeof(double), "DOUBLE"}, {sizeof(short), "HALF"}
};

//enum for acceleration setting
enum class AccSetting {
  NONE, AVX256, AVX512, NEON, CUDA
};

constexpr bool OPTIMIZED_INDEXING_SETTING{true};
constexpr bool USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT{true};
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

inline unsigned int getBytesAlignMemory(beliefprop::AccSetting accelSetting) {
  //avx512 requires data to be aligned on 64 bytes
  return (accelSetting == AccSetting::AVX512) ? 64 : 16;
}

inline unsigned int getNumDataAlignWidth(beliefprop::AccSetting accelSetting) {
  //align width with 16 data values in AVX512
  return (accelSetting == AccSetting::AVX512) ? 16 : 8;
}

template <AccSetting ACCELERATION_SETTING>
inline void writeRunSettingsToStream(std::ostream& resultsStream)
{
  resultsStream << "Memory Optimization Level: " << beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT << "\n";
  resultsStream << "Indexing Optimization Level: " << beliefprop::OPTIMIZED_INDEXING_SETTING << "\n";
  resultsStream << "BYTES_ALIGN_MEMORY: " << beliefprop::getBytesAlignMemory(ACCELERATION_SETTING) << "\n";
  resultsStream << "NUM_DATA_ALIGN_WIDTH: " << beliefprop::getNumDataAlignWidth(ACCELERATION_SETTING) << "\n";
}

template <AccSetting ACCELERATION_SETTING>
inline RunData runSettings()  {
  RunData currRunData;
  currRunData.addDataWHeader("Memory Optimization Level", std::to_string(beliefprop::USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT));
  currRunData.addDataWHeader("Indexing Optimization Level", std::to_string(beliefprop::OPTIMIZED_INDEXING_SETTING));
  currRunData.addDataWHeader("BYTES_ALIGN_MEMORY", std::to_string(beliefprop::getBytesAlignMemory(ACCELERATION_SETTING)));
  currRunData.addDataWHeader("NUM_DATA_ALIGN_WIDTH", std::to_string(beliefprop::getNumDataAlignWidth(ACCELERATION_SETTING)));

  return currRunData;
}

};

#endif /* BPRUNSETTINGS_H_ */
