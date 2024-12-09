/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file RunImpMultInputs.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
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
 * @brief Base class for running and evaluating multiple runs of an implementation
 * that may be optimized on CPU or GPU
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