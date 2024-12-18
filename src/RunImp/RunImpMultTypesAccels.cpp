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
 * @file RunImpMultTypesAccels.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "RunEval/EvaluateImpAliases.h"
#include "RunImpMultTypesAccels.h"

//run and evaluate benchmark using multiple datatypes, inputs, and implementations if available
void RunImpMultTypesAccels::operator()(
  const std::map<run_environment::AccSetting,
                 std::shared_ptr<RunImpMultInputs>>&
    run_benchmark_imps_by_acc_setting,
  const run_environment::RunImpSettings& run_imp_settings,
  std::unique_ptr<EvaluateImpResults> evalResultsPtr) const
{
  //get fastest implementation available
  const auto fastest_acc =
    FastestAvailableAcc(run_benchmark_imps_by_acc_setting);
  
  std::cout << "FASTEST_ACC: " << run_environment::AccelerationString(fastest_acc) << std::endl;

  //get results using each datatype and possible acceleration
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> run_imp_results;
  for (const size_t data_size : run_imp_settings.datatypes_eval_sizes) {
    run_imp_results[data_size] = MultRunDataWSpeedupByAcc();
    //run implementation using each acceleration setting
    for (auto& run_imp : run_benchmark_imps_by_acc_setting) {
      std::cout << "acc run: " << run_environment::AccelerationString(run_imp.first) << std::endl;
      run_imp_results[data_size][run_imp.first] = run_imp.second->operator()(
        run_imp_settings, data_size, evalResultsPtr);
    }
  }

  //evaluate results including writing results to output file
  evalResultsPtr->EvalAllResultsWriteOutput(
    run_imp_results, run_imp_settings, fastest_acc);
}

//get fastest available acceleration
run_environment::AccSetting RunImpMultTypesAccels::FastestAvailableAcc(
  const std::map<run_environment::AccSetting,
                 std::shared_ptr<RunImpMultInputs>>&
    run_benchmark_imps_by_acc_setting) const
{
  if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kCUDA))
  {
    return run_environment::AccSetting::kCUDA;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kAVX512_F16))
  {
    return run_environment::AccSetting::kAVX512_F16;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kAVX512))
  {
    return run_environment::AccSetting::kAVX512;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kNEON))
  {
    return run_environment::AccSetting::kNEON;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kAVX256_F16))
  {
    return run_environment::AccSetting::kAVX256_F16;
  }
  else if (run_benchmark_imps_by_acc_setting.contains(
    run_environment::AccSetting::kAVX256))
  {
    return run_environment::AccSetting::kAVX256;
  }
  else {
    return run_environment::AccSetting::kNone;
  }
}
