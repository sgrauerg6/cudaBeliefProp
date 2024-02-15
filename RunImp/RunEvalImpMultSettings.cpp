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
  std::shared_ptr<RunBenchmarkImp> fastestImp = getFastestAvailableImp(runBenchmarkImpsByAccSetting);
  
  //perform runs with fastest implementation with and without vectorization using floating point
  //initially store output for floating-point runs separate from output using doubles and halfs
  auto runOutput = fastestImp->operator()(runImpSettings, sizeof(float));
  //get results and speedup for using potentially alternate and no vectorization (only applies to CPU)
  const auto altAndNoVectSpeedupFl = getAltAccelSpeedups(runOutput.first, runBenchmarkImpsByAccSetting, runImpSettings, sizeof(float));
  //get run data portion of results
  auto runOutputAltAndNoVect = altAndNoVectSpeedupFl.first;

  //perform runs with and without vectorization using double-precision
  auto runOutputDouble = fastestImp->operator()(runImpSettings, sizeof(double));
  const auto doublesSpeedup = run_eval::getAvgMedSpeedup(runOutput.first, runOutputDouble.first, std::string(run_eval::SPEEDUP_DOUBLE));
  const auto altAndNoVectSpeedupDbl = getAltAccelSpeedups(runOutputDouble.first, runBenchmarkImpsByAccSetting, runImpSettings, sizeof(double));
  for (const auto& runData : altAndNoVectSpeedupDbl.first.first) {
    runOutputAltAndNoVect.first.push_back(runData);
  }
  //perform runs with and without vectorization using half-precision
  auto runOutputHalf = fastestImp->operator()(runImpSettings, sizeof(halftype));
  const auto halfSpeedup = run_eval::getAvgMedSpeedup(runOutput.first, runOutputHalf.first, std::string(run_eval::SPEEDUP_HALF));
  const auto altAndNoVectSpeedupHalf = getAltAccelSpeedups(runOutputHalf.first, runBenchmarkImpsByAccSetting, runImpSettings, sizeof(halftype));
  for (const auto& runData : altAndNoVectSpeedupHalf.first.first) {
    runOutputAltAndNoVect.first.push_back(runData);
  }
  //add output for double and half precision runs to output of floating-point runs to write
  //final output with all data
  runOutput.first.insert(runOutput.first.end(), runOutputDouble.first.begin(), runOutputDouble.first.end());
  runOutput.first.insert(runOutput.first.end(), runOutputHalf.first.begin(), runOutputHalf.first.end());
  //get speedup using vectorization across all runs
  const auto vectorizationSpeedupAll = run_eval::getAvgMedSpeedup(runOutputAltAndNoVect.first, runOutput.first,
    std::string(run_eval::SPEEDUP_VECTORIZATION) + " - All Runs");
  //add speedup data from double and half precision runs to overall data so they are included in final results
  runOutput.second.insert(runOutput.second.end(), altAndNoVectSpeedupFl.second.begin(), altAndNoVectSpeedupFl.second.end());
  runOutput.second.insert(runOutput.second.end(), runOutputDouble.second.begin(), runOutputDouble.second.end());
  runOutput.second.insert(runOutput.second.end(), altAndNoVectSpeedupDbl.second.begin(), altAndNoVectSpeedupDbl.second.end());
  runOutput.second.insert(runOutput.second.end(), runOutputHalf.second.begin(), runOutputHalf.second.end());
  runOutput.second.insert(runOutput.second.end(), altAndNoVectSpeedupHalf.second.begin(), altAndNoVectSpeedupHalf.second.end());

  //get speedup over baseline runtimes...can only compare with baseline runtimes that are
  //generated using same templated iterations setting as current run
  if ((runImpSettings.baseOptSingThreadRTimeForTSetting_) &&
      (runImpSettings.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings.templatedItersSetting_)) {
    const auto speedupOverBaseline = run_eval::getAvgMedSpeedupOverBaseline(runOutput.first, "All Runs",
      runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first);
      runOutput.second.insert(runOutput.second.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
  }
  //get speedup info for using optimized parallel parameters
  if (runImpSettings.optParallelParamsOptionSetting_.first) {
    runOutput.second.push_back(run_eval::getAvgMedSpeedupOptPParams(
      runOutput.first, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - All Runs"));
  }
  if (runImpSettings.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
    //get speedup when using template for loop iteration count
    runOutput.second.push_back(run_eval::getAvgMedSpeedupLoopItersInTemplate(
      runOutput.first, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - All Runs"));
  }
  runOutput.second.push_back(vectorizationSpeedupAll);
  runOutput.second.push_back(doublesSpeedup);
  runOutput.second.push_back(halfSpeedup);

  //write output corresponding to results for all data types
  constexpr bool MULT_DATA_TYPES{true};
  run_eval::writeRunOutput<MULT_DATA_TYPES>(runOutput, runImpSettings, fastestImp->getAccelerationSetting());
}

//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
std::pair<std::pair<MultRunData, std::vector<MultRunSpeedup>>, std::vector<MultRunSpeedup>> RunEvalImpMultSettings::getAltAccelSpeedups(MultRunData& runOutputData,
  const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting, const run_environment::RunImpSettings& runImpSettings,
  size_t dataTypeSize) const
{
  const std::string speedupVsNoVectStr{std::string(run_eval::SPEEDUP_VECTORIZATION) + " - " +
    run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)};
  const std::string speedupVsAVX256Str{std::string(run_eval::SPEEDUP_VS_AVX256_VECTORIZATION) + " - " +
    run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)};
  std::vector<MultRunSpeedup> multRunSpeedupVect;
  //no alternate run if only a single element in runBenchmarkImpsByAccSetting
  if (runBenchmarkImpsByAccSetting.size() == 1) {
    multRunSpeedupVect.push_back({speedupVsNoVectStr, {0.0, 0.0}});
    multRunSpeedupVect.push_back({speedupVsAVX256Str, {0.0, 0.0}});
    return {std::pair<MultRunData, std::vector<MultRunSpeedup>>(), multRunSpeedupVect};
  }
  else {
    std::pair<MultRunData, std::vector<MultRunSpeedup>> runOutputNoVect;
    //if initial speedup is AVX512, also run AVX256
    for (auto& altAccImp : runBenchmarkImpsByAccSetting) {
      if (altAccImp.first == run_environment::AccSetting::AVX256) {
        //run implementation using AVX256 acceleration
        auto runOutputAVX256 = altAccImp.second->operator()(runImpSettings, dataTypeSize);
        //go through each result and replace initial run data with AVX256 run data if AVX256 run is faster
        for (unsigned int i = 0; i < runOutputData.size(); i++) {
          if ((runOutputData[i].first == run_eval::Status::NO_ERROR) && (runOutputAVX256.first[i].first == run_eval::Status::NO_ERROR)) {
            const double initResultTime = std::stod(runOutputData[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
            const double avx256ResultTime = std::stod(runOutputAVX256.first[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
            if (avx256ResultTime < initResultTime) {
              runOutputData[i] = runOutputAVX256.first[i];
            }
          }
        }
        //get speedup using AVX512 compared to AVX256
        const auto speedupOverAVX256 = run_eval::getAvgMedSpeedup(runOutputAVX256.first, runOutputData, speedupVsAVX256Str);
        multRunSpeedupVect.push_back(speedupOverAVX256);
      }
      if (altAccImp.first == run_environment::AccSetting::NONE) {
        //run implementation is no acceleration
        runOutputNoVect = altAccImp.second->operator()(runImpSettings, dataTypeSize);
        //go through each result and replace initial run data with no vectorization run data if no vectorization run is faster
        for (unsigned int i = 0; i < runOutputData.size(); i++) {
          if ((runOutputData[i].first == run_eval::Status::NO_ERROR) && (runOutputNoVect.first[i].first == run_eval::Status::NO_ERROR)) {
            const double initResultTime = std::stod(runOutputData[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
            const double noVectResultTime = std::stod(runOutputNoVect.first[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
            if (noVectResultTime < initResultTime) {
              runOutputData[i] = runOutputNoVect.first[i];
            }
          }
        }
        //get speedup using current acceleration setting compared to no acceleration
        const auto speedupWVectorization = run_eval::getAvgMedSpeedup(runOutputNoVect.first, runOutputData, speedupVsNoVectStr);
        multRunSpeedupVect.push_back(speedupWVectorization);
      }
    }
    return {runOutputNoVect, multRunSpeedupVect};
  }
  return {std::pair<MultRunData, std::vector<MultRunSpeedup>>(), multRunSpeedupVect};
}

//get fastest available implementation
std::shared_ptr<RunBenchmarkImp> RunEvalImpMultSettings::getFastestAvailableImp(const std::map<run_environment::AccSetting,
  std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting) const
{
  if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::CUDA)) {
    return runBenchmarkImpsByAccSetting.at(run_environment::AccSetting::CUDA);
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::AVX512)) {
    return runBenchmarkImpsByAccSetting.at(run_environment::AccSetting::AVX512);
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::NEON)) {
    return runBenchmarkImpsByAccSetting.at(run_environment::AccSetting::NEON);
  }
  else if (runBenchmarkImpsByAccSetting.contains(run_environment::AccSetting::AVX256)) {
    return runBenchmarkImpsByAccSetting.at(run_environment::AccSetting::AVX256);
  }
  else {
    return runBenchmarkImpsByAccSetting.begin()->second;
  }
}
