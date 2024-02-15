/*
 * RunBenchmarkImp.h
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#ifndef RUN_BENCHMARK_IMP_H
#define RUN_BENCHMARK_IMP_H

#include <utility>
#include <numeric>
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunEvalUtils.h"

//base class for running and evaluating multiple runs of benchmark that may be optimized on CPU or GPU
class RunBenchmarkImp {
public:
  RunBenchmarkImp(const run_environment::AccSetting& optImpAccel) : optImpAccel_(optImpAccel) {}

  //run and evaluate runs on one or more input of benchmark implementation using multiple settings
  std::pair<MultRunData, std::vector<MultRunSpeedup>> operator()(const run_environment::RunImpSettings& runImpSettings,
    const size_t dataTypeSize) const;

  //return acceleration setting for implementation
  run_environment::AccSetting getAccelerationSetting() const { return optImpAccel_; }

protected:
  const run_environment::AccSetting optImpAccel_;

private:
  //run and evaluate implementation on multiple data sets
  virtual MultRunData runEvalImpMultDataSets(const run_environment::RunImpSettings& runImpSettings, const size_t dataTypeSize) const = 0;

  //get speedup over baseline data if data available
  std::vector<MultRunSpeedup> getSpeedupOverBaseline(const run_environment::RunImpSettings& runImpSettings,
    MultRunData& runDataAllRuns, const size_t dataTypeSize) const;
  
  //get speedup over baseline data for belief propagation run for specific input subsets if data available
  //by default returns empty vector and can be overriden in child class to retrieve speedups for specific subsets
  //for benchmark
  virtual std::vector<MultRunSpeedup> getSpeedupOverBaselineSubsets(const run_environment::RunImpSettings& runImpSettings,
    MultRunData& runDataAllRuns, const size_t dataTypeSize) const { return std::vector<MultRunSpeedup>(); }
};

#endif //RUN_BENCHMARK_IMP_H