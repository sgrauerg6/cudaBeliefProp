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

//base class for running and evaluating multiple runs of benchmark that may be optimized on CPU or GPU
class RunBenchmarkImp {
public:
  RunBenchmarkImp(run_environment::AccSetting opt_imp_accel) : opt_imp_accel_(opt_imp_accel) {}

  //run and evaluate runs on one or more input of benchmark implementation using multiple settings
  std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> operator()(const run_environment::RunImpSettings& run_imp_settings,
    size_t data_type_size) const;

  //return acceleration setting for implementation
  run_environment::AccSetting AccelerationSetting() const { return opt_imp_accel_; }

protected:
  const run_environment::AccSetting opt_imp_accel_;

private:
  //run and evaluate implementation on multiple data sets
  virtual MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const = 0;
};

#endif //RUN_BENCHMARK_IMP_H