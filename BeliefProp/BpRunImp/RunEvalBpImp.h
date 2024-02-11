/*
 * RunEvalBpImp.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNEVALBPIMP_H_
#define RUNEVALBPIMP_H_

#include <omp.h>
#include <memory>
#include <array>
#include <fstream>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <set>
#include "BpRunImp/RunEvalBPImpOnInput.h"
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "BpFileProcessing/BpFileHandling.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpSingleThreadCPU/stereo.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunEvalUtils.h"
#include "RunImp/RunBenchmarkImp.h"
#include "BpParallelParams.h"

//run and evaluate optimized belief propagation implementation on a number of inputs
//acceleration method of optimized belief propagation implementation is specified in template parameter
template <run_environment::AccSetting OPT_IMP_ACCEL>
class RunEvalBpImp : public RunBenchmarkImp {
public:
  std::pair<MultRunData, std::vector<MultRunSpeedup>> operator()(const run_environment::RunImpSettings& runImpSettings,
    const size_t dataTypeSize) const override;
  
  //retrieve acceleration setting for optimized belief propagation implementation that is being evaluated
  run_environment::AccSetting getAccelerationSetting() const override { return OPT_IMP_ACCEL; }

private:
  //perform runs on multiple data sets using specified data type and acceleration method
  template <RunData_t T>
  std::pair<MultRunData, std::vector<MultRunSpeedup>> operator()(const run_environment::RunImpSettings& runImpSettings) const;
};

//run belief propagation implementation with given data type and acceleration setting
template <run_environment::AccSetting OPT_IMP_ACCEL>
std::pair<MultRunData, std::vector<MultRunSpeedup>> RunEvalBpImp<OPT_IMP_ACCEL>::operator()(const run_environment::RunImpSettings& runImpSettings,
  const size_t dataTypeSize) const
{
  if (dataTypeSize == sizeof(float)) {
    return operator()<float>(runImpSettings);
  }
  else if (dataTypeSize == sizeof(double)) {
    return operator()<double>(runImpSettings);
  }
  else {
    return operator()<halftype>(runImpSettings);
  }
}

//perform runs on multiple data sets using specified data type and acceleration method
template <run_environment::AccSetting OPT_IMP_ACCEL>
template <RunData_t T>
std::pair<MultRunData, std::vector<MultRunSpeedup>> RunEvalBpImp<OPT_IMP_ACCEL>::operator()(const run_environment::RunImpSettings& runImpSettings) const {
  MultRunData runData;
  std::vector<MultRunData> runResultsEachInput;
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 0>().operator()(runImpSettings));
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 1>().operator()(runImpSettings));
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 2>().operator()(runImpSettings));
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 3>().operator()(runImpSettings));
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 4>().operator()(runImpSettings));
#ifndef SMALLER_SETS_ONLY
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 5>().operator()(runImpSettings));
  runResultsEachInput.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 6>().operator()(runImpSettings));
#endif //SMALLER_SETS_ONLY

  //add run results for each input to overall results
  for (auto& runResult : runResultsEachInput) {
    runData.insert(runData.end(), runResult.begin(), runResult.end());
  }

  //initialize speedup results
  std::vector<MultRunSpeedup> speedupResults;

  //get speedup info for using optimized parallel parameters and disparity count as template parameter
  if constexpr (std::is_same_v<T, float>) {
    //get speedup over baseline runtimes...can only compare with baseline runtimes that are
    //generated using same templated iterations setting as current run
    if ((runImpSettings.baseOptSingThreadRTimeForTSetting_) &&
        (runImpSettings.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings.templatedItersSetting_)) {
      const std::vector<std::pair<std::string, std::vector<unsigned int>>> subsetsStrIndices = {
        {"smallest 3 stereo sets", {0, 1, 2, 3, 4, 5}},
#ifndef SMALLER_SETS_ONLY
        {"largest 3 stereo sets", {8, 9, 10, 11, 12, 13}}
#else
        {"largest stereo set", {8, 9}}
#endif //SMALLER_SETS_ONLY
      };
      const auto speedupOverBaseline = run_eval::getAvgMedSpeedupOverBaseline(
        runData, run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)),
        runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first, subsetsStrIndices);
      speedupResults.insert(speedupResults.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
    }
  }
  if (runImpSettings.optParallelParamsOptionSetting_.first) {
    speedupResults.push_back(run_eval::getAvgMedSpeedupOptPParams(runData, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - " +
      run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))));
  }
  if (runImpSettings.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
    speedupResults.push_back(run_eval::getAvgMedSpeedupLoopItersInTemplate(runData, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - " +
      run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))));
  }
    
  //write output corresponding to results for current data type
  constexpr bool MULT_DATA_TYPES{false};
  run_eval::writeRunOutput<MULT_DATA_TYPES, T>({runData, speedupResults}, runImpSettings, OPT_IMP_ACCEL);

  //return data for each run and multiple average and median speedup results across the data
  return {runData, speedupResults};
}

#endif /* RUNEVALBPIMP_H_ */
