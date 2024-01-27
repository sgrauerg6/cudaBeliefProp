/*
 * RunSettings.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef RUNSETTINGS_H_
#define RUNSETTINGS_H_

#include <map>
#include <optional>
#include "RunData.h"

namespace run_environment {

//mapping from data size to data type string
const std::map<std::size_t, std::string> DATA_SIZE_TO_NAME_MAP{
  {sizeof(float), "FLOAT"}, {sizeof(double), "DOUBLE"}, {sizeof(short), "HALF"}
};

//enum for acceleration setting
enum class AccSetting {
  NONE, AVX256, AVX512, NEON, CUDA
};

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
  //path to baseline runtimes for optimized and single thread runs and template setting used to generate baseline runtimes
  std::optional<std::pair<std::array<std::string_view, 2>, TemplatedItersSetting>> baseOptSingThreadRTimeForTSetting_;
};

};

#endif /* RUNSETTINGS_H_ */
