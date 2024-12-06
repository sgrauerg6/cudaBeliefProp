/*
 * RunImpMultTypesAccels.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUN_IMP_MULT_TYPES_ACCELS_H_
#define RUN_IMP_MULT_TYPES_ACCELS_H_

#include <memory>
#include <vector>
#include <map>
#include "RunSettingsParams/RunSettings.h"
#include "RunImp/RunImpMultInputs.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "RunEval/EvaluateImpResults.h"

/**
 * @brief Class to run and evaluate a optimized benchmark implementation using
 * multiple settings
 * Optimization may be on the CPU using vectorization or on the GPU using CUDA
 * 
 */
class RunImpMultTypesAccels {
public:
  /**
   * @brief Run and evaluate benchmark using multiple datatypes,
   * inputs, and implementations if available
   * 
   * @param run_benchmark_imps_by_acc_setting 
   * @param run_imp_settings 
   * @param evalResultsPtr 
   */
  void operator()(
    const std::map<run_environment::AccSetting,
    std::shared_ptr<RunImpMultInputs>>& run_benchmark_imps_by_acc_setting,
    const run_environment::RunImpSettings& run_imp_settings,
    std::unique_ptr<EvaluateImpResults> evalResultsPtr) const;

private:
  /**
   * @brief Get fastest available implementation
   * 
   * @param run_benchmark_imps_by_acc_setting 
   * @return run_environment::AccSetting 
   */
  run_environment::AccSetting FastestAvailableAcc(
    const std::map<run_environment::AccSetting,
    std::shared_ptr<RunImpMultInputs>>& run_benchmark_imps_by_acc_setting) const;
};

#endif /* RUN_IMP_MULT_TYPES_ACCELS_H_ */
