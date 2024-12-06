/*
 * RunImpMultInputs.h
 *
 *  Created on: Feb 3, 2024
 *      Author: scott
 */

#ifndef RUN_IMP_MULT_INPUTS_H_
#define RUN_IMP_MULT_INPUTS_H_

#include <utility>
#include <memory>
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/EvaluateImpResults.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "RunEval/EvaluateImpAliases.h"

/**
 * @brief Base class for running and evaluating multiple runs of benchmark that may be optimized on CPU or GPU
 * 
 */
class RunImpMultInputs {
public:
  RunImpMultInputs(run_environment::AccSetting opt_imp_accel) : opt_imp_accel_(opt_imp_accel) {}

  /**
   * @brief Run and evaluate runs on one or more input of benchmark implementation using multiple settings
   * 
   * @param run_imp_settings 
   * @param data_type_size 
   * @param evalResults 
   * @return std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> 
   */
  std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> operator()(
    const run_environment::RunImpSettings& run_imp_settings,
    size_t data_type_size,
    std::unique_ptr<EvaluateImpResults>& evalResults) const;

  /**
   * @brief Return acceleration setting for implementation
   * 
   * @return run_environment::AccSetting 
   */
  run_environment::AccSetting AccelerationSetting() const { return opt_imp_accel_; }

protected:
  const run_environment::AccSetting opt_imp_accel_;

private:
  /**
   * @brief Run and evaluate implementation on multiple data sets
   * 
   * @param run_imp_settings 
   * @param data_type_size 
   * @return MultRunData 
   */
  virtual MultRunData RunEvalImpMultDataSets(
    const run_environment::RunImpSettings& run_imp_settings,
    size_t data_type_size) const = 0;
};

#endif //RUN_IMP_MULT_INPUTS_H_