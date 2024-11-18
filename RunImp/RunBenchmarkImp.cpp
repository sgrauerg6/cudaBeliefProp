/*
 * RunBenchmarkImp.cpp
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#include "RunBenchmarkImp.h"
#include "BpResultsEvaluation/EvaluateBPImpResults.h"

//run and evaluate runs on one or more input of benchmark implementation using multiple settings
std::pair<MultRunData, std::vector<MultRunSpeedup>> RunBenchmarkImp::operator()(const run_environment::RunImpSettings& run_imp_settings,
  size_t data_type_size) const
{
  //run belief propagation implementation on multiple datasets and return run data for all runs
  MultRunData run_data_all_runs = RunEvalImpMultDataSets(run_imp_settings, data_type_size);

  //evaluate results
  EvaluateBPImpResults evalResults;
  evalResults(run_data_all_runs, run_imp_settings, opt_imp_accel_, data_type_size);

  //return data for each run and multiple average and median speedup results across the data
  return evalResults.RunDataWSpeedups();
}
