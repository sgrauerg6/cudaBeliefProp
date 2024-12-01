/*
 * RunEvalBpImp.cpp
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#include "RunEvalBpImp.h"

MultRunData RunEvalBpImp::RunEvalImpMultDataSets(
  const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const 
{
  if (this->opt_imp_accel_ == run_environment::AccSetting::kCUDA) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kCUDA>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kAVX512) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kAVX512>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kAVX256) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kAVX256>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kNEON) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kNEON>(run_imp_settings, data_type_size);
  }
  //else (this->opt_imp_accel_ == run_environment::AccSetting::kNone)
  return RunEvalImpMultDataSets<run_environment::AccSetting::kNone>(run_imp_settings, data_type_size);
}

template <run_environment::AccSetting OPT_IMP_ACCEL>
MultRunData RunEvalBpImp::RunEvalImpMultDataSets(
  const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const
{
  if (data_type_size == sizeof(float)) {
    return RunEvalImpMultDataSets<float, OPT_IMP_ACCEL>(run_imp_settings);
  }
  else if (data_type_size == sizeof(double)) {
    return RunEvalImpMultDataSets<double, OPT_IMP_ACCEL>(run_imp_settings);
  }
  else {
    return RunEvalImpMultDataSets<halftype, OPT_IMP_ACCEL>(run_imp_settings);
  }
}

//run and evaluate optimized bp implementation on evaluation stereo sets
template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
MultRunData RunEvalBpImp::RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings) const {
  //run and evaluate bp implementation on all stereo sets used for benchmarking
  std::vector<MultRunData> run_results;
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 0>().operator()(run_imp_settings));
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 1>().operator()(run_imp_settings));
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 2>().operator()(run_imp_settings));
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 3>().operator()(run_imp_settings));
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 4>().operator()(run_imp_settings));
#ifndef SMALLER_SETS_ONLY
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 5>().operator()(run_imp_settings));
  run_results.push_back(RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, 6>().operator()(run_imp_settings));
#endif //SMALLER_SETS_ONLY

  //merge results for each input to overall results
  MultRunData run_data_all_runs;
  for (auto& run_result : run_results) {
    run_data_all_runs.merge(run_result);
  }
 
  return run_data_all_runs;
}
