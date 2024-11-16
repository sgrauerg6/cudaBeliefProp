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
#include "BpRunImp/RunEvalBPImpOnInput.h"
#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "BpFileProcessing/BpFileHandling.h"
#include "BpRunProcessing/RunBpStereoSet.h"
#include "BpSingleThreadCPU/stereo.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunImp/RunBenchmarkImp.h"
#include "BpParallelParams.h"

//run and evaluate optimized belief propagation implementation on a number of inputs
//acceleration method of optimized belief propagation implementation is specified in template parameter
class RunEvalBpImp final : public RunBenchmarkImp {
public:
  RunEvalBpImp(const run_environment::AccSetting& optImpAccel) : RunBenchmarkImp(optImpAccel) {}

private:
  //run and evaluate implementation on multiple data sets
  MultRunData runEvalImpMultDataSets(const run_environment::RunImpSettings& runImpSettings, const size_t dataTypeSize) const override;
  
  template <run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData runEvalImpMultDataSets(const run_environment::RunImpSettings& runImpSettings, const size_t dataTypeSize) const;

  template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData runEvalImpMultDataSets(const run_environment::RunImpSettings& runImpSettings) const;
};

#endif /* RUNEVALBPIMP_H_ */
