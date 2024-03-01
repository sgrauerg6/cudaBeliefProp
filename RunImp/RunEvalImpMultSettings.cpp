/*
 * RunEvalImpMultSettings.cpp
 *
 *  Created on: Feb 14, 2024
 *      Author: scott
 */

#include "RunEvalImpMultSettings.h"
#include "RunSettingsEval/EvaluateImpResults.h"

//run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
void RunEvalImpMultSettings::operator()(const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting,
  const run_environment::RunImpSettings& runImpSettings) const
{
  std::cout << "1a" << std::endl;
  //get fastest implementation available
  const auto fastestAcc = getFastestAvailableAcc(runBenchmarkImpsByAccSetting);

  //get results using each datatype and possible acceleration
  std::unordered_map<size_t, std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>> runImpResults;
  for (const size_t dataSize : {sizeof(float), sizeof(double), sizeof(halftype)}) {
    runImpResults[dataSize] = std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>();
    //run implementation using each acceleration setting
    for (auto& runImp : runBenchmarkImpsByAccSetting) {
      runImpResults[dataSize][runImp.first] = runImp.second->operator()(runImpSettings, dataSize);
    }
  }

  std::cout << "1b" << std::endl;
  //evaluate results including writing results to output file
  EvaluateImpResults().operator()(runImpResults, runImpSettings, fastestAcc);
}

//get fastest available acceleration
run_environment::AccSetting RunEvalImpMultSettings::getFastestAvailableAcc(const std::map<run_environment::AccSetting,
  std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting) const
{
  if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::CUDA)) {
    return run_environment::AccSetting::CUDA;
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::AVX512)) {
    return run_environment::AccSetting::AVX512;
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::NEON)) {
    return run_environment::AccSetting::NEON;
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::AVX256)) {
    return run_environment::AccSetting::AVX256;
  }
  else {
    return run_environment::AccSetting::NONE;
  }
}
