/*
 * RunBpImpMultInputs.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUN_BP_IMP_MULT_INPUTS_H_
#define RUN_BP_IMP_MULT_INPUTS_H_

#include "RunSettingsParams/RunSettings.h"
#include "RunImp/RunImpMultInputs.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/EvaluateImpAliases.h"

//run and evaluate optimized belief propagation implementation on a number of inputs
//acceleration method of optimized belief propagation implementation is specified in template parameter
class RunBpImpMultInputs final : public RunImpMultInputs {
public:
  RunBpImpMultInputs(run_environment::AccSetting opt_imp_accel) : RunImpMultInputs(opt_imp_accel) {}

private:
  //run and evaluate implementation on multiple data sets
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const override;
  
  template <run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const;

  template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings) const;
};

#endif /* RUN_BP_IMP_MULT_INPUTS_H_ */
