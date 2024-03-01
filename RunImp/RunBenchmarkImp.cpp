/*
 * RunBenchmarkImp.cpp
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#include "RunBenchmarkImp.h"
#include "RunSettingsEval/EvaluateImpResults.h"

//run and evaluate runs on one or more input of benchmark implementation using multiple settings
std::pair<MultRunData, std::vector<MultRunSpeedup>> RunBenchmarkImp::operator()(const run_environment::RunImpSettings& runImpSettings,
  const size_t dataTypeSize) const
{
  std::cout << "2a" << std::endl;
  //run belief propagation implementation on multiple datasets and return run data for all runs
  MultRunData runDataAllRuns = runEvalImpMultDataSets(runImpSettings, dataTypeSize);

  std::cout << "2b" << std::endl;
  //evaluate results
  EvaluateImpResults evalResults;
  evalResults(runDataAllRuns, runImpSettings, optImpAccel_, dataTypeSize);

  std::cout << "2c" << std::endl;
  //return data for each run and multiple average and median speedup results across the data
  return evalResults.getRunDataWSpeedups();
}
