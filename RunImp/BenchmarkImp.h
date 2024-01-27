/*
 * BenchmarkImp.h
 *
 *  Created on: Jan 24, 2019
 *      Author: scott
 */

#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"

using MultRunData = std::vector<std::pair<run_eval::Status, std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;

class BenchmarkImp {
  //perform runs on multiple data sets using specified data type and acceleration method
  template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
  virtual std::pair<MultRunData, std::vector<MultRunSpeedup>> operator()() const = 0; 
};