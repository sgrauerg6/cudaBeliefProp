/*
 * RunEvalBpImp.cpp
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#include "RunEvalBpImp.h"

//get speedup over baseline data for belief propagation run if data available
std::vector<MultRunSpeedup> RunEvalBpImp::getSpeedupOverBaseline(const run_environment::RunImpSettings& runImpSettings,
  MultRunData& runDataAllRuns, const size_t dataTypeSize) const
{
  //initialize speedup results
  std::vector<MultRunSpeedup> speedupResults;

  //get speedup info for using optimized parallel parameters and disparity count as template parameter
  if (dataTypeSize == sizeof(float)) {
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
        runDataAllRuns, run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize),
        runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first, subsetsStrIndices);
      speedupResults.insert(speedupResults.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
    }
  }

  return speedupResults;
}

MultRunData RunEvalBpImp::runEvalImpMultDataSets(
  const run_environment::RunImpSettings& runImpSettings, const size_t dataTypeSize) const 
{
  if (this->optImpAccel_ == run_environment::AccSetting::CUDA) {
    return runEvalImpMultDataSets<run_environment::AccSetting::CUDA>(runImpSettings, dataTypeSize);
  }
  else if (this->optImpAccel_ == run_environment::AccSetting::AVX512) {
    return runEvalImpMultDataSets<run_environment::AccSetting::AVX512>(runImpSettings, dataTypeSize);
  }
  else if (this->optImpAccel_ == run_environment::AccSetting::AVX256) {
    return runEvalImpMultDataSets<run_environment::AccSetting::AVX256>(runImpSettings, dataTypeSize);
  }
  else if (this->optImpAccel_ == run_environment::AccSetting::NEON) {
    return runEvalImpMultDataSets<run_environment::AccSetting::NEON>(runImpSettings, dataTypeSize);
  }
  //else (this->optImpAccel_ == run_environment::AccSetting::NONE)
  return runEvalImpMultDataSets<run_environment::AccSetting::NONE>(runImpSettings, dataTypeSize);
}

template <run_environment::AccSetting OPT_IMP_ACCEL>
MultRunData RunEvalBpImp::runEvalImpMultDataSets(
  const run_environment::RunImpSettings& runImpSettings, const size_t dataTypeSize) const
{
  if (dataTypeSize == sizeof(float)) {
    return runEvalImpMultDataSets<float, OPT_IMP_ACCEL>(runImpSettings);
  }
  else if (dataTypeSize == sizeof(double)) {
    return runEvalImpMultDataSets<double, OPT_IMP_ACCEL>(runImpSettings);
  }
  else {
    return runEvalImpMultDataSets<halftype, OPT_IMP_ACCEL>(runImpSettings);
  }
}

//run and evaluate implementation on multiple data sets
template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
MultRunData RunEvalBpImp::runEvalImpMultDataSets(const run_environment::RunImpSettings& runImpSettings) const {
  MultRunData runDataAllRuns;
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
    runDataAllRuns.insert(runDataAllRuns.end(), runResult.begin(), runResult.end());
  }
 
  return runDataAllRuns;
}
