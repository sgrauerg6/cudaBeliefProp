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
 * @file RunImpOnInput.h
 * @author Scott Grauer-Gray
 * @brief Declares virtual class to run and evaluate implementation on a input
 * specified by index number
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef RUN_IMP_ON_INPUT_H_
#define RUN_IMP_ON_INPUT_H_

#include <utility>
#include <memory>
#include <optional>
#include <vector>
#include <array>
#include <string>
#include "RunSettingsParams/RunSettings.h"
#include "RunSettingsParams/ParallelParams.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/EvaluateImpAliases.h"

/**
 * @brief Virtual class to run and evaluate implementation on a input specified by index number
 * 
 * @tparam T 
 * @tparam OPT_IMP_ACCEL 
 * @tparam NUM_INPUT 
 */
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
class RunImpOnInput {
public:
  virtual MultRunData operator()(const run_environment::RunImpSettings& run_imp_settings) = 0;

protected:
  /**
   * @brief Set up parallel parameters for benchmark
   * 
   * @param run_imp_settings 
   * @return std::shared_ptr<ParallelParams> 
   */
  virtual std::shared_ptr<ParallelParams> SetUpParallelParams(
    const run_environment::RunImpSettings& run_imp_settings) const = 0;

  /**
   * @brief Retrieve input and parameters for run of current benchmark
   * 
   * @param loop_iters_templated 
   * @return RunData 
   */
  virtual RunData InputAndParamsForCurrBenchmark(bool loop_iters_templated) const = 0;

  /**
   * @brief Run one or two implementations of benchmark and compare results if
   * running multiple implementations
   * 
   * @param parallel_params 
   * @param run_opt_imp_only 
   * @param run_imp_templated_loop_iters 
   * @return std::optional<RunData> 
   */
  virtual std::optional<RunData> RunImpsAndCompare(
    std::shared_ptr<ParallelParams> parallel_params,
    bool run_opt_imp_only,
    bool run_imp_templated_loop_iters) const = 0;

  /**
   * @brief Get current run inputs and parameters in RunData structure
   * 
   * @param loop_iters_templated 
   * @return RunData 
   */
  RunData InputAndParamsRunData(bool loop_iters_templated) const {
    RunData curr_run_data;
    curr_run_data.AddDataWHeader(
      std::string(run_eval::kDatatypeHeader),
      std::string(run_environment::kDataSizeToNameMap.at(sizeof(T))));
    curr_run_data.AppendData(run_environment::RunSettings<OPT_IMP_ACCEL>());
    curr_run_data.AddDataWHeader(
      std::string(run_eval::kLoopItersTemplatedHeader), loop_iters_templated);
    return curr_run_data;
  }

  /**
   * @brief Run optimized and single threaded implementations using multiple sets of
   * parallel parameters in optimized implementation if set to optimize parallel
   * parameters
   * Returns data from runs using default and optimized parallel parameters
   * 
   * @param run_imp_settings 
   * @param run_w_loop_iters_templated 
   * @return MultRunData::mapped_type 
   */
  MultRunData::mapped_type RunEvalBenchmark(
    const run_environment::RunImpSettings& run_imp_settings,
    bool run_w_loop_iters_templated) const
  {
    MultRunData::mapped_type::value_type out_run_data;
    enum class RunType { ONLY_RUN, DEFAULT_PARAMS, OPTIMIZED_RUN, TEST_PARAMS };

    //set up parallel parameters for specific benchmark
    std::shared_ptr<ParallelParams> parallel_params = SetUpParallelParams(run_imp_settings);

    //get constant references to default and optimized options
    //for parallel parameters
    const auto& [default_p_params, optimized_p_param_options] =
      run_imp_settings.p_params_default_opt_settings;
      
    //if optimizing parallel parameters, run BP for each parallel parameters option,
    //retrieve best parameters for each kernel or overall for the run,
    //and then run BP with best found parallel parameters
    //if not optimizing parallel parameters, run BP once using default parallel parameters
    for (unsigned int run_num = 0;
         run_num < (optimized_p_param_options.size() + 1);
         run_num++)
    {
      //initialize current run type to specify if current run is only run, run with
      //default params, test params run, or final run with optimized params
      RunType curr_run_type{RunType::TEST_PARAMS};
      if (optimized_p_param_options.size() == 0) {
        curr_run_type = RunType::ONLY_RUN;
      }
      else if (run_num == optimized_p_param_options.size()) {
        curr_run_type = RunType::OPTIMIZED_RUN;
      }

      //get and set parallel parameters for current run if not final run
      //that uses optimized parameters
      std::array<unsigned int, 2> p_params_curr_run{default_p_params};
      if (curr_run_type == RunType::ONLY_RUN) {
        parallel_params->SetParallelDims(
          default_p_params);
      }
      else if (curr_run_type == RunType::TEST_PARAMS) {
        //set parallel parameters to parameters corresponding to current run
        //for each BP processing level
        p_params_curr_run = optimized_p_param_options[run_num];
        parallel_params->SetParallelDims(p_params_curr_run);
        if (p_params_curr_run == default_p_params) {
          //set run type to default parameters if current run uses default parameters
          curr_run_type = RunType::DEFAULT_PARAMS;
        }
      }

      //store input params data if using default parallel parameters or
      //final run with optimized parameters
      RunData curr_run_data;
      if (curr_run_type != RunType::TEST_PARAMS) {
        //add input and parameters data for specific benchmark to current run data
        curr_run_data.AddDataWHeader(
          std::string(run_eval::kInputIdxHeader),
          std::to_string(NUM_INPUT));
        curr_run_data.AppendData(
          InputAndParamsForCurrBenchmark(run_w_loop_iters_templated));
        if ((optimized_p_param_options.size() > 0) &&
            (run_imp_settings.opt_parallel_params_setting ==
             run_environment::OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun))
        {
          //add parallel parameters for each kernel to current input data if
          //allowing different parallel parameters for each kernel
          //in the same run
          curr_run_data.AppendData(parallel_params->AsRunData());
        }
      }

      //run only optimized implementation and not single-threaded run if
      //current run is not final run or is using default parameter parameters
      const bool run_opt_imp_only{curr_run_type == RunType::TEST_PARAMS};

      //run benchmark implementation(s) and return null output if error in run
      //detailed results stored to file that is generated using stream
      const auto run_imps_results =
        RunImpsAndCompare(
          parallel_params,
          run_opt_imp_only,
          run_w_loop_iters_templated);

      //if error in run and run is any type other than for testing parameters,
      //exit function with null output to indicate error
      if ((!run_imps_results) && (curr_run_type != RunType::TEST_PARAMS)) {
        return {};
      }

      //add data results from current run if run successful
      if (run_imps_results) {
        curr_run_data.AppendData(run_imps_results.value());
      }

      //add current run results for output if using default parallel 
      //parameters or is final run w/ optimized parallel parameters
      if (curr_run_type != RunType::TEST_PARAMS) {
        //set output for runs using default parallel parameters and final run
        //(which is the same run if not optimizing parallel parameters)
        if (curr_run_type == RunType::OPTIMIZED_RUN) {
          out_run_data[run_environment::ParallelParamsSetting::kOptimized] = curr_run_data;
        }
        else {
          out_run_data[run_environment::ParallelParamsSetting::kDefault] = curr_run_data;
          if (curr_run_type == RunType::ONLY_RUN) {
            //if current run is only run due to setting of not optimizing parallel parameters,
            //set optimized parallel parameters output to current run data, resulting in same
            //run data mapped to default and optimized parallel parameters enums
            out_run_data[run_environment::ParallelParamsSetting::kOptimized] = curr_run_data;
          }
        }
      }

      if (optimized_p_param_options.size() > 0) {
        //retrieve and store results including runtimes for each kernel if
        //allowing different parallel parameters for each kernel and
        //total runtime for current run
        //if error in run, don't add results for current parallel parameters to results set
        if (run_imps_results) {
          if (curr_run_type != RunType::OPTIMIZED_RUN) {
            parallel_params->AddTestResultsForParallelParams(p_params_curr_run, curr_run_data);
          }
        }

        //set optimized parallel parameters if next run is final run
        //that uses optimized parallel parameters
        //optimized parallel parameters are determined from previous evaluation runs
        //using multiple evaluation parallel parameters
        if (run_num == (optimized_p_param_options.size() - 1)) {
          parallel_params->SetOptimizedParams();
        }
      }
    }
      
    return out_run_data;
  }
};

#endif //RUN_IMP_ON_INPUT_H_