/*
 * RunEvalImpMultSettings.cpp
 *
 *  Created on: Feb 14, 2024
 *      Author: scott
 */

#include "RunEvalImpMultSettings.h"

//run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
void RunEvalImpMultSettings::operator()(const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting,
  const run_environment::RunImpSettings& runImpSettings) const
{
  //get fastest implementation available
  const auto fastestAcc = getFastestAvailableAcc(runBenchmarkImpsByAccSetting);

  //get results and speedup for using each possible acceleration
  std::unordered_map<size_t, std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>> runImpResults;
  std::unordered_map<size_t, std::vector<MultRunSpeedup>> altImpSpeedup;
  std::unordered_map<size_t, MultRunSpeedup> altDataTypeSpeedup;
  for (const size_t dataSize : {sizeof(float), sizeof(double), sizeof(halftype)}) {
    runImpResults[dataSize] = std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>();
    //run implementation using each acceleration setting
    for (auto& runImp : runBenchmarkImpsByAccSetting) {
      runImpResults[dataSize][runImp.first] = runImp.second->operator()(runImpSettings, dataSize);
    }
    //get speedup/slowdown using alternate accelerations
    altImpSpeedup[dataSize] = getAltAccelSpeedups(runImpResults[dataSize], runImpSettings, dataSize, fastestAcc);
    if (dataSize != sizeof(float)) {
      //get speedup or slowdown using alternate data type (double or half) compared with float
      altDataTypeSpeedup[dataSize] = run_eval::getAvgMedSpeedup(runImpResults[sizeof(float)][fastestAcc].first,
        runImpResults[dataSize][fastestAcc].first, (dataSize > sizeof(float)) ? std::string(run_eval::SPEEDUP_DOUBLE) : std::string(run_eval::SPEEDUP_HALF));
    }
  }

  //initialize overall results to float results using fastest acceleration and add double and half-type results to it
  auto resultsWSpeedups = runImpResults[sizeof(float)][fastestAcc];
  resultsWSpeedups.first.insert(resultsWSpeedups.first.end(),
    runImpResults[sizeof(double)][fastestAcc].first.begin(), runImpResults[sizeof(double)][fastestAcc].first.end());
  resultsWSpeedups.first.insert(resultsWSpeedups.first.end(),
    runImpResults[sizeof(halftype)][fastestAcc].first.begin(), runImpResults[sizeof(halftype)][fastestAcc].first.end());

  //add speedup data from double and half precision runs to speedup results
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), altImpSpeedup[sizeof(float)].begin(), altImpSpeedup[sizeof(float)].end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(),
    runImpResults[sizeof(double)][fastestAcc].second.begin(), runImpResults[sizeof(double)][fastestAcc].second.end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), altImpSpeedup[sizeof(double)].begin(), altImpSpeedup[sizeof(double)].end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(),
    runImpResults[sizeof(halftype)][fastestAcc].second.begin(), runImpResults[sizeof(halftype)][fastestAcc].second.end());
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), altImpSpeedup[sizeof(halftype)].begin(), altImpSpeedup[sizeof(halftype)].end());

  //get speedup over baseline runtimes...can only compare with baseline runtimes that are
  //generated using same templated iterations setting as current run
  if ((runImpSettings.baseOptSingThreadRTimeForTSetting_) &&
      (runImpSettings.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings.templatedItersSetting_)) {
    const auto speedupOverBaseline = run_eval::getAvgMedSpeedupOverBaseline(resultsWSpeedups.first, "All Runs",
      runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first);
    resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
  }

  //get speedup info for using optimized parallel parameters
  if (runImpSettings.optParallelParamsOptionSetting_.first) {
    resultsWSpeedups.second.push_back(run_eval::getAvgMedSpeedupOptPParams(
      resultsWSpeedups.first, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - All Runs"));
  }

  //get speedup when using template for loop iteration count
  if (runImpSettings.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
    resultsWSpeedups.second.push_back(run_eval::getAvgMedSpeedupLoopItersInTemplate(
      resultsWSpeedups.first, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - All Runs"));
  }

  //add speedups when using doubles and half precision compared to float to end of speedup data
  resultsWSpeedups.second.insert(resultsWSpeedups.second.end(), {altDataTypeSpeedup[sizeof(double)], altDataTypeSpeedup[sizeof(halftype)]});

  //write output corresponding to results and speedups for all data types
  constexpr bool MULT_DATA_TYPES{true};
  run_eval::writeRunOutput<MULT_DATA_TYPES>(resultsWSpeedups, runImpSettings, fastestAcc);
}

//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
std::vector<MultRunSpeedup> RunEvalImpMultSettings::getAltAccelSpeedups(
  std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<MultRunSpeedup>>>& runImpResultsByAccSetting,
  const run_environment::RunImpSettings& runImpSettings, size_t dataTypeSize, run_environment::AccSetting fastestAcc) const
{
  //set up mapping from acceleration type to description
  const std::map<run_environment::AccSetting, std::string> accToSpeedupStr{
    {run_environment::AccSetting::NONE, 
     std::string(run_eval::SPEEDUP_VECTORIZATION) + " - " + run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)},
    {run_environment::AccSetting::AVX256, 
     std::string(run_eval::SPEEDUP_VS_AVX256_VECTORIZATION) + " - " + run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)}};

  if (runImpResultsByAccSetting.size() == 1) {
    //no alternate run results
    return {{accToSpeedupStr.at(run_environment::AccSetting::NONE), {0.0, 0.0}}, {accToSpeedupStr.at(run_environment::AccSetting::AVX256), {0.0, 0.0}}};
  }
  else {
    //initialize speedup/slowdown using alternate acceleration
    std::vector<MultRunSpeedup> altAccSpeedups;
    for (auto& altAccImpResults : runImpResultsByAccSetting) {
      if ((altAccImpResults.first != fastestAcc) && (accToSpeedupStr.contains(altAccImpResults.first))) {
        //process results using alternate acceleration
        //go through each result and replace initial run data with alternate implementation run data if alternate implementation run is faster
        for (unsigned int i = 0; i < runImpResultsByAccSetting[fastestAcc].first.size(); i++) {
          if (runImpResultsByAccSetting[fastestAcc].first[i] && altAccImpResults.second.first[i]) {
            const double initResultTime = std::stod(runImpResultsByAccSetting[fastestAcc].first[i]->back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
            const double altAccResultTime = std::stod(altAccImpResults.second.first[i]->back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
            if (altAccResultTime < initResultTime) {
              runImpResultsByAccSetting[fastestAcc].first[i] = altAccImpResults.second.first[i];
            }
          }
        }
        //get speedup/slowdown using alternate acceleration compared to fastest implementation and store in speedup results
        altAccSpeedups.push_back(run_eval::getAvgMedSpeedup(
          altAccImpResults.second.first, runImpResultsByAccSetting[fastestAcc].first, accToSpeedupStr.at(altAccImpResults.first)));
      }
    }
    return altAccSpeedups;
  }
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
