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
 * @file RunBpImpMultInputs.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_BP_IMP_MULT_INPUTS_H_
#define RUN_BP_IMP_MULT_INPUTS_H_

#include "RunSettingsParams/RunSettings.h"
#include "RunImp/RunImpMultInputs.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/EvaluateImpAliases.h"

/**
 * @brief Class to run specified belief propagation implementation on a
 * number of inputs.
 * 
 */
class RunBpImpMultInputs final : public RunImpMultInputs {
public:
  RunBpImpMultInputs(run_environment::AccSetting opt_imp_accel) : RunImpMultInputs(opt_imp_accel) {}

private:
  /**
   * @brief Run and evaluate implementation on multiple data sets
   * 
   * @param run_imp_settings 
   * @param data_type_size 
   * @return MultRunData 
   */
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const override;
  
  /**
   * @brief Run and evaluate implementation on multiple data sets
   * 
   * @tparam OPT_IMP_ACCEL 
   * @param run_imp_settings 
   * @param data_type_size 
   * @return MultRunData 
   */
  template <run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size) const;

  /**
   * @brief Run and evaluate implementation on multiple data sets
   * 
   * @tparam T 
   * @tparam OPT_IMP_ACCEL 
   * @param run_imp_settings 
   * @return MultRunData 
   */
  template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData RunEvalImpMultDataSets(const run_environment::RunImpSettings& run_imp_settings) const;
};

#endif /* RUN_BP_IMP_MULT_INPUTS_H_ */
