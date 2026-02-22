/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file RunImpOnInputBnchmrks.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_IMP_ON_INPUT_BNCHMRKS_H_
#define RUN_IMP_ON_INPUT_BNCHMRKS_H_

//check if optimized CPU run defined and make any necessary additions to support it
#if defined(OPTIMIZED_CPU_RUN)
//needed to run the optimized CPU implementation
#include "benchmarksOptCPU/RunBnchmrksOptCPU.h"
//set RunBpOptimized alias to correspond to optimized CPU implementation
template <RunData_t T, run_environment::AccSetting ACCELERATION>
using RunBnchmrksOptimized = RunBnchmrksOptCPU<T, ACCELERATION>;
#endif //OPTIMIZED_CPU_RUN

//check if CUDA run defined and make any necessary additions to support it
#if defined(OPTIMIZED_CUDA_RUN)
//needed to run the CUDA implementation
#include "benchmarksCUDA/RunBnchmrksCUDA.h"
//set RunBpOptimized alias to correspond to CUDA implementation
template <RunData_t T, run_environment::AccSetting ACCELERATION>
using RunBnchmrksOptimized = RunBnchmrksCUDA<T, ACCELERATION>;
#endif //OPTIMIZED_CUDA_RUN

#include "RunImpOnInput.h"

/**
 * @brief Child class of RunImpOnInput to run and evaluate benchmark(s)
 * implementation on a specified input
 * 
 * @tparam T 
 * @tparam OPT_IMP_ACCEL 
 * @tparam NUM_INPUT 
 */
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, size_t NUM_INPUT>
class RunImpOnInputBnchmrks final : public RunImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT> {
public:
  MultRunData operator()(
    const run_environment::RunImpSettings& run_imp_settings) override;

protected:
  /**
   * @brief Set up parallel parameters for benchmark
   * 
   * @param run_imp_settings 
   * @return Shared pointer to parallel parameters as set up for run
   */
  std::shared_ptr<ParallelParams> SetUpParallelParams(
    const run_environment::RunImpSettings& run_imp_settings) const override;

  /**
   * @brief Retrieve input and parameters for run of current benchmark
   * 
   * @param loop_iters_templated 
   * @return RunData object with input data and parameter info about current
   * benchmark
   */
  RunData InputAndParamsForCurrBenchmark(
    bool loop_iters_templated) const override;

  /**
   * @brief Run one or two implementations of benchmarks and compare results if
   * running multiple implementations
   * 
   * @param parallel_params 
   * @param run_opt_imp_only 
   * @param run_imp_templated_loop_iters 
   * @return Restults of running and comparing implementations or null output
   * if error in run
   */
  std::optional<RunData> RunImpsAndCompare(
    std::shared_ptr<ParallelParams> parallel_params,
    bool run_opt_imp_only,
    bool run_imp_templated_loop_iters) const override;

private:
  /** @brief Unique pointer to run benchmarks object for single thread
   *  implementation */
  std::unique_ptr<RunBnchmrksOptimized<T, run_environment::AccSetting::kNone>>
    run_benchmarks_single_thread_;
  
  /** @brief Unique pointer to run benchmarks object for optimized 
   *  implementation */
  std::unique_ptr<RunBnchmrksOptimized<T, OPT_IMP_ACCEL>>
    run_benchmarks_opt_;

  /** @brief width and height of matrix used in benchmarks */
  unsigned int matrix_wh_;
};

//run and evaluate optimized belief propagation implementation on evaluation
//stereo set specified by NUM_INPUT
//data type used in implementation specified by T
//bp implemenation optimization specified by OPT_IMP_ACCEL
//evaluation stereo set to run implementation on specified by NUM_INPUT
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, size_t NUM_INPUT>
MultRunData RunImpOnInputBnchmrks<T, OPT_IMP_ACCEL, NUM_INPUT>::operator()(
  const run_environment::RunImpSettings& run_imp_settings)
{
  //set up matrix settings for current run
  matrix_wh_ =
    benchmarks::kMtrxsToProcess[NUM_INPUT].mtrx_wh;

  //initialize run results across multiple implementations
  MultRunData run_results;

  //set up unoptimized single threaded benchmarks implementation
  run_bnchmrks_single_thread_ =
    std::make_unique<RunBnchmrksSingThreadCPU<T, run_environment::AccSetting::kNone>>();

  //set up and run benchmarks using optimized implementation (optimized CPU and
  //CUDA implementations supported) as well as unoptimized implementation for
  //comparison
  run_opt_bnchmrks =
    std::make_unique<RunBnchmrksOptimized<T, OPT_IMP_ACCEL>>();
  constexpr bool run_w_loop_iters_templated{false};
  InputSignature input_sig(
    sizeof(T), NUM_INPUT, run_w_loop_iters_templated);
  run_results.insert(
    {input_sig,
      this->RunEvalBenchmark(
      run_imp_settings,
      run_w_loop_iters_templated)});

  //return bp run results across multiple implementations
  return run_results; 
}

//set up parallel parameters for running belief propagation in parallel on
//CPU or GPU
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, size_t NUM_INPUT>
std::shared_ptr<ParallelParams>
RunImpOnInputBnchmrks<T, OPT_IMP_ACCEL, NUM_INPUT>::SetUpParallelParams(
  const run_environment::RunImpSettings& run_imp_settings) const
{
  //parallel parameters initialized with default thread count dimensions at
  //every level
  return std::make_shared<ParallelParamsBnchmrks>(
    run_imp_settings.opt_parallel_params_setting,
    run_imp_settings.p_params_default_alt_options.first);
}

//get input data and parameter info about current benchmark (belief propagation
//in this case) and return as RunData type
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, size_t NUM_INPUT>
RunData RunImpOnInputBnchmrks<T, OPT_IMP_ACCEL, NUM_INPUT>::InputAndParamsForCurrBenchmark(
  bool loop_iters_templated) const
{
  RunData curr_run_data;
  /*curr_run_data.AddDataWHeader(
    std::string(beliefprop::kStereoSetHeader),
    std::string(beliefprop::kStereoSetsToProcess[NUM_INPUT].name));
  curr_run_data.AppendData(this->InputAndParamsRunData(loop_iters_templated));
  curr_run_data.AppendData(alg_settings_.AsRunData());
  curr_run_data.AppendData(beliefprop::RunSettings());*/
  return curr_run_data;
}

//run and compare output disparity maps using the given optimized and
//single-threaded stereo implementations on the reference and test images
//specified by numStereoSet
//run only optimized implementation if run_opt_imp_only is true
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, size_t NUM_INPUT>
std::optional<RunData> RunImpOnInputBnchmrks<T, OPT_IMP_ACCEL, NUM_INPUT>::RunImpsAndCompare(
  std::shared_ptr<ParallelParams> parallel_params,
  bool run_opt_imp_only,
  bool run_imp_templated_loop_iters) const
{
  //get number of implementations to run output disparity map file path(s)
  //if run_opt_imp_only is false, run single-threaded implementation in
  //addition to optimized implementation
  const unsigned int num_imps_run{run_opt_imp_only ? 1u : 2u};

  //get optimized implementation description and write info about run to
  //std::cout stream
  const std::string opt_imp_run_description{
    run_benchmarks_opt_->RunDescription()};
  std::cout << "Running benchmarks with height/width of " << matrix_wh_ << " on " << opt_imp_run_description;
  if (!run_opt_imp_only) {
    std::cout << " and " << run_bnchmrks_single_thread_->RunDescription();
  }
  std::cout << std::endl;
  std::cout << "Data size: " << sizeof(T) << std::endl;
  std::cout << "Acceleration: " << run_environment::AccelerationString<OPT_IMP_ACCEL>() << std::endl;
  std::cout << std::endl;
      
  //run optimized implementation and retrieve structure with runtime and output
  //disparity map
  std::map<run_environment::AccSetting, std::optional<benchmarks::BnchmrksRunOutput>>
    run_output;
  run_output[OPT_IMP_ACCEL] =
    run_benchmarks_opt_->operator()(matrix_wh_, *parallel_params);
    
  //check if error in run
  RunData run_data;
  run_data.AddDataWHeader(
    std::string(run_environment::kAccelerationDescHeader),
    opt_imp_run_description);
  if (!(run_output[OPT_IMP_ACCEL])) {
    return {};
  }

  //append implementation run data to evaluation run data
  run_data.AppendData(run_output[OPT_IMP_ACCEL]->run_data);
  run_data.AddDataWHeader(
    std::string(run_eval::kOptimizedRuntimeHeader),
    run_output[OPT_IMP_ACCEL]->run_time.count());

  //check if only running optimized implementation or if also running
  //single-threaded implementation
  if (!run_opt_imp_only) {
    //run single-threaded implementation and retrieve structure with runtime
    //and output disparity map
    run_output[run_environment::AccSetting::kNone] =
      run_benchmarks_single_thread_->operator()(matrix_wh_, *parallel_params);
    if (!(run_output[run_environment::AccSetting::kNone])) {
      return {};
    }
    //append run data for single thread run to evaluation run data
    run_data.AppendData(
      run_output[run_environment::AccSetting::kNone]->run_data);
  }

  //return structure indicating that run succeeded along with data from run
  return run_data;
}

#endif //RUN_IMP_ON_INPUT_BNCHMRKS_H_