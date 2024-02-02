/*
 * RunEvalImp.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNEVALIMP_H_
#define RUNEVALIMP_H_

#include <memory>
#include <array>
#include <vector>
#include <numeric>
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunEvalUtils.h"
#include "BpRunImp/RunEvalBpImp.h"

using MultRunData = std::vector<std::pair<run_eval::Status, std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;

namespace RunAndEvaluateImp {

//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
std::pair<std::pair<MultRunData, std::vector<MultRunSpeedup>>, std::vector<MultRunSpeedup>> getAltAndNoVectSpeedup(MultRunData& runOutputData,
  const std::unique_ptr<RunEvalBpImp>& runBpImp, const run_environment::RunImpSettings& runImpSettings,
  size_t dataTypeSize, run_environment::AccSetting accelerationSetting) {
  const std::string speedupHeader{std::string(run_eval::SPEEDUP_VECTORIZATION) + " - " +
    run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)};
  const std::string speedupVsAVX256Str{std::string(run_eval::SPEEDUP_VS_AVX256_VECTORIZATION) + " - " +
    run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)};
  std::vector<MultRunSpeedup> multRunSpeedupVect;
  if ((accelerationSetting == run_environment::AccSetting::CUDA) || (accelerationSetting == run_environment::AccSetting::NONE)) {
    multRunSpeedupVect.push_back({speedupHeader, {0.0, 0.0}});
    multRunSpeedupVect.push_back({speedupVsAVX256Str, {0.0, 0.0}});
    return {std::pair<MultRunData, std::vector<MultRunSpeedup>>(), multRunSpeedupVect};
  }
  else {
    //if initial speedup is AVX512, also run AVX256
    if (accelerationSetting == run_environment::AccSetting::AVX512) {
      //run implementation using AVX256 acceleration
      auto runOutputAVX256 = runBpImp->operator()(runImpSettings, dataTypeSize, run_environment::AccSetting::AVX256);
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
    else {
      multRunSpeedupVect.push_back({speedupVsAVX256Str, {0.0, 0.0}});
    }
    //run implementation is no acceleration
    auto runOutputNoVect = runBpImp->operator()(runImpSettings, dataTypeSize, run_environment::AccSetting::NONE);
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
    const auto speedupWVectorization = run_eval::getAvgMedSpeedup(runOutputNoVect.first, runOutputData, speedupHeader);
    multRunSpeedupVect.push_back(speedupWVectorization);
    return {runOutputNoVect, multRunSpeedupVect};
  }
}

void runBpOnStereoSets(const std::unique_ptr<RunEvalBpImp>& runBpImp, const run_environment::RunImpSettings& runImpSettings,
  run_environment::AccSetting accelerationSetting) {
  //perform runs with and without vectorization using floating point
  //initially store output for floating-point runs separate from output using doubles and halfs
  auto runOutput = runBpImp->operator()(runImpSettings, sizeof(float), accelerationSetting);
  //get results and speedup for using potentially alternate and no vectorization (only applies to CPU)
  const auto altAndNoVectSpeedupFl = getAltAndNoVectSpeedup(runOutput.first, runBpImp, runImpSettings, sizeof(float), accelerationSetting);
  //get run data portion of results
  auto runOutputAltAndNoVect = altAndNoVectSpeedupFl.first;

  //perform runs with and without vectorization using double-precision
  auto runOutputDouble = runBpImp->operator()(runImpSettings, sizeof(double), accelerationSetting);
  const auto doublesSpeedup = run_eval::getAvgMedSpeedup(runOutput.first, runOutputDouble.first, std::string(run_eval::SPEEDUP_DOUBLE));
  const auto altAndNoVectSpeedupDbl = getAltAndNoVectSpeedup(runOutputDouble.first, runBpImp, runImpSettings, sizeof(double), accelerationSetting);
  for (const auto& runData : altAndNoVectSpeedupDbl.first.first) {
    runOutputAltAndNoVect.first.push_back(runData);
  }
  //perform runs with and without vectorization using half-precision
  auto runOutputHalf = runBpImp->operator()(runImpSettings, sizeof(halftype), accelerationSetting);
  const auto halfSpeedup = run_eval::getAvgMedSpeedup(runOutput.first, runOutputHalf.first, std::string(run_eval::SPEEDUP_HALF));
  const auto altAndNoVectSpeedupHalf = getAltAndNoVectSpeedup(runOutputHalf.first, runBpImp, runImpSettings, sizeof(halftype), accelerationSetting);
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
  if (runImpSettings.optParallelParmsOptionSetting_.first) {
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
  run_eval::writeRunOutput<MULT_DATA_TYPES>(runOutput, runImpSettings, accelerationSetting);
}

}

#endif /* RUNEVALIMP_H_ */
