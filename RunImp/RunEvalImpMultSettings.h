/*
 * RunEvalImpMultSettings.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNEVALIMPMULTSETTINGS_H_
#define RUNEVALIMPMULTSETTINGS_H_

#include <memory>
#include <array>
#include <vector>
#include <numeric>
#include <map>
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunEvalUtils.h"
#include "RunImp/RunBenchmarkImp.h"

//class to run and evaluate a optimized benchmark implementation using multiple settings
//optimization may be on the CPU using vectorization or on the GPU using CUDA
class RunEvalImpMultSettings {
public:
  //run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
  void operator()(const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting,
    const run_environment::RunImpSettings& runImpSettings) const;

private:
  //perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
  //CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
  std::pair<std::pair<MultRunData, std::vector<MultRunSpeedup>>, std::vector<MultRunSpeedup>> getAltAccelSpeedups(MultRunData& runOutputData,
    const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting, const run_environment::RunImpSettings& runImpSettings,
    size_t dataTypeSize) const;

  //get fastest available implementation
  std::shared_ptr<RunBenchmarkImp> getFastestAvailableImp(const std::map<run_environment::AccSetting,
    std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting) const;
};

#endif /* RUNEVALIMPMULTSETTINGS_H_ */
