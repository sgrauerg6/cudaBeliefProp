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
 * @file RunImpMultTypesAccels.h
 * @author Scott Grauer-Gray
 * @brief Declares class to run and evaluate implementation(s) of an algorithm
 * using multiple settings including different datatype, inputs, and
 * acceleration methods.
 * 
 * @copyright Copyright (c) 2024
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
 * @brief Class to run and evaluate implementation(s) of an algorithm using
 * multiple settings including different datatype, inputs, and acceleration
 * methods.
 */
class RunImpMultTypesAccels {
public:
  /**
   * @brief Run and evaluate benchmark using multiple datatypes,
   * inputs, and implementations if available
   * 
   * @param run_benchmark_imps_w_acc 
   * @param run_imp_settings 
   * @param evalResultsPtr 
   */
  void operator()(
    const std::vector<std::shared_ptr<RunImpMultInputs>>& run_benchmark_imps_w_acc,
    const run_environment::RunImpSettings& run_imp_settings,
    std::unique_ptr<EvaluateImpResults> evalResultsPtr) const;

private:
  /**
   * @brief Get fastest acceleration across input run implementations
   * 
   * @param run_benchmark_imps_w_acc 
   * @return run_environment::AccSetting 
   */
  run_environment::AccSetting FastestAvailableAcc(
    const std::vector<std::shared_ptr<RunImpMultInputs>>& run_benchmark_imps_w_acc) const;
};

#endif /* RUN_IMP_MULT_TYPES_ACCELS_H_ */
