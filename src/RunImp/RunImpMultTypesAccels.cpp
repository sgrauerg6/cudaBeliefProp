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

//run and evaluate benchmark using multiple datatypes, inputs, and
//implementations if available
void RunImpMultTypesAccels::operator()(
  const std::vector<std::shared_ptr<RunImpMultInputs>>& run_benchmark_imps_w_acc,
  const run_environment::RunImpSettings& run_imp_settings,
  std::unique_ptr<EvaluateImpResults> evalResultsPtr) const
{
  //get expected fastest implementation available based on acceleration used
  //in each implementation
  const auto fastest_acc =
    FastestAvailableAcc(run_benchmark_imps_w_acc);
  std::cout << "FASTEST_ACC: " 
            << run_environment::AccelerationString(fastest_acc)
            << std::endl;

  //get results using each datatype and possible acceleration
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> run_imp_results;
  for (const size_t data_size : run_imp_settings.datatypes_eval_sizes) {
    //add data size to run implementation results
    run_imp_results.insert({data_size, MultRunDataWSpeedupByAcc()});
    //run each input implementation using current evaluation data size
    //and add results of run to overall run results with results indexed
    //by data size and acceleration setting
    for (const auto& run_imp : run_benchmark_imps_w_acc) {
      //check if set to only use expected fastest acceleration and if that's
      //the case skip alternate implementations
      if ((run_imp_settings.run_alt_optimized_imps) ||
          (run_imp->AccelerationSetting() == fastest_acc))
      {
        std::cout << "acc run: " 
                  << run_environment::AccelerationString(run_imp->AccelerationSetting())
                  << std::endl;
        run_imp_results.at(data_size).insert(
          {run_imp->AccelerationSetting(),
           run_imp->operator()(run_imp_settings, data_size, evalResultsPtr)});
      }
    }
  }
  
  //print outputs
  std::cout << "RUN RESULTS 2" << std::endl;
  for (const auto& [data_size, run_result_by_acc] : run_imp_results)
  {
    std::cout << "DATA SIZE: " << data_size << std::endl;
    for (const auto& [acc, run_results_w_speedup] : run_result_by_acc)
    {
      std::cout << "ACC: " << run_environment::AccelerationString(acc) << std::endl;
      for (const auto& [input_sig, run_results] : run_results_w_speedup.first) {
        std::cout << "SIG 2: " << input_sig << std::endl;
        if (run_results) {
          std::cout << run_results->begin()->second << std::endl;
        }
        else {
          std::cout << "RESULTS MISSING" << std::endl;
        }
      }
    }
  }
  std::cout << "RUN RESULTS 2 DONE" << std::endl;

  //evaluate results including writing results to output file
  evalResultsPtr->EvalAllResultsWriteOutput(
    run_imp_results, run_imp_settings, fastest_acc);
}

//get fastest acceleration across input run implementations
run_environment::AccSetting RunImpMultTypesAccels::FastestAvailableAcc(
  const std::vector<std::shared_ptr<RunImpMultInputs>>& run_benchmark_imps_w_acc) const
{
  //go through each possible acceleration in order from fastest to slowest and
  //check if any input run implementation contains that acceleration
  //return fastest acceleration across input run implementations
  for (const auto& acceleration : 
    {run_environment::AccSetting::kCUDA,
     run_environment::AccSetting::kAVX512_F16,
     run_environment::AccSetting::kAVX512,
     run_environment::AccSetting::kAVX256_F16,
     run_environment::AccSetting::kAVX256,
     run_environment::AccSetting::kNEON})
  {
    if (std::any_of(run_benchmark_imps_w_acc.cbegin(), run_benchmark_imps_w_acc.cend(),
                    [&acceleration](const auto& run_imp) { 
                      return run_imp->AccelerationSetting() == acceleration;
                    }))
    {
      //return acceleration as fastest acceleration if equal to acceleration
      //setting of any input run implementation
      return acceleration;
    }
  }
  //return "no acceleration" as fastest if no faster acceleration in input run
  //implementations
  return run_environment::AccSetting::kNone;
}
