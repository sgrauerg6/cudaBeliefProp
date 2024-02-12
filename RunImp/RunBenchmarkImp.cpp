/*
 * RunBenchmarkImp.cpp
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#include "RunBenchmarkImp.h"

//run and evaluate runs on one or more input of benchmark implementation using multiple settings
std::pair<MultRunData, std::vector<MultRunSpeedup>> RunBenchmarkImp::operator()(const run_environment::RunImpSettings& runImpSettings,
  const size_t dataTypeSize) const
{
  //run belief propagation implementation on multiple datasets and return run data for all runs
  MultRunData runDataAllRuns = runEvalImpMultDataSets(runImpSettings, dataTypeSize);  

  //initialize speedup results
  std::vector<MultRunSpeedup> speedupResults;

  //add speedup results over baseline data if available for current input
  auto speedupOverBaseline = getSpeedupOverBaseline(runImpSettings, runDataAllRuns, dataTypeSize);
  speedupResults.insert(speedupResults.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());

  //get speedup info for using optimized parallel parameters and disparity count as template parameter
  if (runImpSettings.optParallelParamsOptionSetting_.first) {
    speedupResults.push_back(run_eval::getAvgMedSpeedupOptPParams(runDataAllRuns, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - " +
      run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)));
  }
  if (runImpSettings.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
    speedupResults.push_back(run_eval::getAvgMedSpeedupLoopItersInTemplate(runDataAllRuns, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - " +
      run_environment::DATA_SIZE_TO_NAME_MAP.at(dataTypeSize)));
  }

  //write output corresponding to results for current data type
  constexpr bool MULT_DATA_TYPES{false};
  run_eval::writeRunOutput<MULT_DATA_TYPES>({runDataAllRuns, speedupResults}, runImpSettings, optImpAccel_, dataTypeSize);

  //return data for each run and multiple average and median speedup results across the data
  return {runDataAllRuns, speedupResults};
}
