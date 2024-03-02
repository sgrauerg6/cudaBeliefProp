/*
 * RunEvalBpImp.cpp
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#include "RunEvalBpImp.h"

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
  MultRunData runDataAllRuns;
  for (auto& runResult : runResultsEachInput) {
    runDataAllRuns.insert(runDataAllRuns.cend(), runResult.cbegin(), runResult.cend());
  }
 
  return runDataAllRuns;
}
