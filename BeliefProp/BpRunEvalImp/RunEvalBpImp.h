/*
 * RunEvalBpImp.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNEVALBPIMP_H_
#define RUNEVALBPIMP_H_

#include <omp.h>
#include <memory>
#include <array>
#include <fstream>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <set>
#include "BpRunEvalImp/RunEvalBPImpOnInput.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpResultsEvaluation/DetailedTimingBPConsts.h"
#include "BpFileProcessing/BpFileHandling.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpSingleThreadCPU/stereo.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/EvaluateImpAliases.h"
#include "RunImp/RunBenchmarkImp.h"

//run and evaluate optimized belief propagation implementation on a number of inputs
//acceleration method of optimized belief propagation implementation is specified in template parameter
class RunEvalBpImp final : public RunBenchmarkImp {
public:
  RunEvalBpImp(run_environment::AccSetting opt_imp_accel) : RunBenchmarkImp(opt_imp_accel) {}

private:
  //run and evaluate implementation on multiple data sets
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const override;
  
  template <run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const;

  template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings) const;
};

#endif /* RUNEVALBPIMP_H_ */
