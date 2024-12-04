/*
 * RunEvalImpMultSettings.cpp
 *
 *  Created on: Feb 14, 2024
 *      Author: scott
 */

#include "RunEval/EvaluateImpAliases.h"
#include "RunEvalImpMultSettings.h"

//run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
void RunEvalImpMultSettings::operator()(
  const std::map<run_environment::AccSetting,
  std::shared_ptr<RunBenchmarkImp>>& run_benchmark_imps_by_acc_setting,
  const run_environment::RunImpSettings& run_imp_settings,
  std::unique_ptr<EvaluateImpResults> evalResultsPtr) const
{
  //get fastest implementation available
  const auto fastest_acc =
    FastestAvailableAcc(run_benchmark_imps_by_acc_setting);

  //get results using each datatype and possible acceleration
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> run_imp_results;
  for (const size_t data_size : run_eval::kDataTypesEvalSizes) {
    run_imp_results[data_size] = MultRunDataWSpeedupByAcc();
    //run implementation using each acceleration setting
    for (auto& run_imp : run_benchmark_imps_by_acc_setting) {
      run_imp_results[data_size][run_imp.first] = run_imp.second->operator()(
        run_imp_settings, data_size, evalResultsPtr);
    }
  }

  //evaluate results including writing results to output file
  evalResultsPtr->EvalAllResultsWriteOutput(
    run_imp_results, run_imp_settings, fastest_acc);
}

//get fastest available acceleration
run_environment::AccSetting RunEvalImpMultSettings::FastestAvailableAcc(
  const std::map<run_environment::AccSetting,
  std::shared_ptr<RunBenchmarkImp>>& run_benchmark_imps_by_acc_setting) const
{
  if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kCUDA))
  {
    return run_environment::AccSetting::kCUDA;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kAVX512))
  {
    return run_environment::AccSetting::kAVX512;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kNEON))
  {
    return run_environment::AccSetting::kNEON;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kAVX256))
  {
    return run_environment::AccSetting::kAVX256;
  }
  else {
    return run_environment::AccSetting::kNone;
  }
}
