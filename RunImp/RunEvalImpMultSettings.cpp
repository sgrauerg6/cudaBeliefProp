/*
 * RunEvalImpMultSettings.cpp
 *
 *  Created on: Feb 14, 2024
 *      Author: scott
 */

#include "RunEvalImpMultSettings.h"
#include "RunSettingsEval/EvaluateImpResults.h"
#include "BpResultsEvaluation/EvaluateBPImpResults.h"

//run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
void RunEvalImpMultSettings::operator()(const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting,
  const run_environment::RunImpSettings& run_imp_settings) const
{
  //get fastest implementation available
  const auto fastest_acc = getFastestAvailableAcc(runBenchmarkImpsByAccSetting);

  //get results using each datatype and possible acceleration
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> runImpResults;
  for (const size_t data_size : {sizeof(float), sizeof(double), sizeof(halftype)}) {
    runImpResults[data_size] = MultRunDataWSpeedupByAcc();
    //run implementation using each acceleration setting
    for (auto& runImp : runBenchmarkImpsByAccSetting) {
      runImpResults[data_size][runImp.first] = runImp.second->operator()(run_imp_settings, data_size);
    }
  }

  //evaluate results including writing results to output file
  EvaluateBPImpResults().operator()(runImpResults, run_imp_settings, fastest_acc);
}

//get fastest available acceleration
run_environment::AccSetting RunEvalImpMultSettings::getFastestAvailableAcc(const std::map<run_environment::AccSetting,
  std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting) const
{
  if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::kCUDA)) {
    return run_environment::AccSetting::kCUDA;
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::kAVX512)) {
    return run_environment::AccSetting::kAVX512;
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::kNEON)) {
    return run_environment::AccSetting::kNEON;
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::kAVX256)) {
    return run_environment::AccSetting::kAVX256;
  }
  else {
    return run_environment::AccSetting::kNone;
  }
}
