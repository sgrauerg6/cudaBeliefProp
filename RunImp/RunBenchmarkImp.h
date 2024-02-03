/*
 * RunBenchmarkImp.h
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#ifndef RUN_BENCHMARK_IMP_H
#define RUN_BENCHMARK_IMP_H

#include <utility>
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"

//base class for running and evaluating multiple runs of benchmark that may be optimized on CPU or GPU
class RunBenchmarkImp {
public:
  virtual std::pair<MultRunData, std::vector<MultRunSpeedup>> operator()(const run_environment::RunImpSettings& runImpSettings,
    const size_t dataTypeSize, const run_environment::AccSetting accelerationSetting) const = 0;
};

#endif //RUN_BENCHMARK_IMP_H