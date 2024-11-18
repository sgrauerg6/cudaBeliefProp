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
#include "RunImp/RunBenchmarkImp.h"

//class to run and evaluate a optimized benchmark implementation using multiple settings
//optimization may be on the CPU using vectorization or on the GPU using CUDA
class RunEvalImpMultSettings {
public:
  //run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
  void operator()(const std::map<run_environment::AccSetting, std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting,
    const run_environment::RunImpSettings& run_imp_settings) const;

private:
  //get fastest available implementation
  run_environment::AccSetting getFastestAvailableAcc(const std::map<run_environment::AccSetting,
    std::shared_ptr<RunBenchmarkImp>>& runBenchmarkImpsByAccSetting) const;
};

#endif /* RUNEVALIMPMULTSETTINGS_H_ */
