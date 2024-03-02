/*
 * RunBenchmarkImp.cpp
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#include "RunBenchmarkImp.h"
#include "BpOutputEvaluation/EvaluateBPImpResults.h"

//run and evaluate runs on one or more input of benchmark implementation using multiple settings
std::pair<MultRunData, std::vector<MultRunSpeedup>> RunBenchmarkImp::operator()(const run_environment::RunImpSettings& runImpSettings,
  const size_t dataTypeSize) const
{
  //run belief propagation implementation on multiple datasets and return run data for all runs
  MultRunData runDataAllRuns = runEvalImpMultDataSets(runImpSettings, dataTypeSize);

  //evaluate results
  EvaluateBPImpResults evalResults;
  evalResults(runDataAllRuns, runImpSettings, optImpAccel_, dataTypeSize);

  //return data for each run and multiple average and median speedup results across the data
  return evalResults.getRunDataWSpeedups();
}
