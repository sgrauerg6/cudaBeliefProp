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
 * @brief Declares child class of RunImpOnInput to run and evaluate belief
 * propagation implementation on a specified input
 * 
 * @copyright Copyright (c) 2024
 */

#include <array>
#include <map>
#include <memory>
#include <filesystem>
#include <optional>
#include <iostream>
#include "RunSettingsParams/RunSettings.h"
#include "RunSettingsParams/InputSignature.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/RunImpOnInput.h"
#include "BpFileProcessing/BpFileHandling.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpSingleThreadCPU/stereo.h"

#ifndef RUN_IMP_ON_INPUT_BP_H_
#define RUN_IMP_ON_INPUT_BP_H_

using filepathtype = std::filesystem::path;

//check if optimized CPU run defined and make any necessary additions to support it
#if defined(OPTIMIZED_CPU_RUN)
//needed to run the optimized implementation a stereo set using CPU
#include "BpOptimizeCPU/RunBpOnStereoSetOptimizedCPU.h"
//set RunBpOptimized alias to correspond to optimized CPU implementation
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
using RunBpOptimized = RunBpOnStereoSetOptimizedCPU<T, DISP_VALS, ACCELERATION>;
#endif //OPTIMIZED_CPU_RUN

//check if CUDA run defined and make any necessary additions to support it
#if defined(OPTIMIZED_CUDA_RUN)
//needed to run the implementation a stereo set using CUDA
#include "BpOptimizeCUDA/RunBpOnStereoSetCUDA.h"
//set RunBpOptimized alias to correspond to CUDA implementation
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
using RunBpOptimized = RunBpOnStereoSetCUDA<T, DISP_VALS, ACCELERATION>;
#endif //OPTIMIZED_CUDA_RUN

/**
 * @brief Namespace with setting of whether or not to run single thread implementation
 * a single time for each input stereo set when running evaluation and storage of 
 * single thread implementation output so it can be re-used so that single thread
 * implementation only needs to be run once for each evaluation stereo set.
 */
namespace bp_single_thread {

/** @brief Setting of whether or not to run single thread implementation once for
 *  each stereo set in evaluation */
constexpr bool kRunSingleThreadOnceForSet{true};

/** @brief Map to store output and run results for single thread implementation for
 *  each stereo set */
inline std::map<std::array<filepathtype, 2>, std::tuple<std::chrono::duration<double>, RunData, filepathtype>>
  single_thread_run_output;

};

/**
 * @brief Child class of RunImpOnInput to run and evaluate belief propagation
 * implementation on a specified input
 * 
 * @tparam T 
 * @tparam OPT_IMP_ACCEL 
 * @tparam NUM_INPUT 
 */
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
class RunImpOnInputBp final : public RunImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT> {
public:
  /**
   * @brief Run and evaluate optimized belief propagation implementation on
   * evaluation stereo set specified by NUM_INPUT.
   * Data type used in implementation specified by class template parameter T.
   * Bp implemenation optimization specified by class template constant OPT_IMP_ACCEL.
   * Evaluation stereo set to run implementation on specified by class template
   * integral constant NUM_INPUT.
   * 
   * @param run_imp_settings 
   * @return MultRunData 
   */
  MultRunData operator()(const run_environment::RunImpSettings& run_imp_settings) override;
  
protected:
  /**
   * @brief Set up parallel parameters for running belief propagation in
   * parallel on CPU or GPU
   * 
   * @param run_imp_settings 
   * @return std::shared_ptr<ParallelParams> 
   */
  std::shared_ptr<ParallelParams> SetUpParallelParams(
    const run_environment::RunImpSettings& run_imp_settings) const override;

  /**
   * @brief Get input data and parameter info about current benchmark (belief
   * propagation in this case) and return as RunData type
   * 
   * @param loop_iters_templated 
   * @return RunData 
   */
  RunData InputAndParamsForCurrBenchmark(bool loop_iters_templated) const override;

  /**
   * @brief Run and compare output disparity maps using the given optimized and single-threaded stereo implementations
   * on the reference and test images specified by NUM_INPUT given as class template integral constant parameter.
   * Run only optimized implementation if run_opt_imp_only is true.
   * 
   * @param parallel_params 
   * @param run_opt_imp_only 
   * @param run_imp_templated_loop_iters 
   * @return std::optional<RunData> 
   */
  std::optional<RunData> RunImpsAndCompare(
    std::shared_ptr<ParallelParams> parallel_params,
    bool run_opt_imp_only,
    bool run_imp_templated_loop_iters) const override;

private:
  /** @brief Unique pointer to run bp implementation object for single thread implementation */
  std::unique_ptr<RunBpOnStereoSet<
    T,
    beliefprop::kStereoSetsToProcess[NUM_INPUT].num_disp_vals,
    run_environment::AccSetting::kNone>>
  run_bp_stereo_single_thread_;
  
  /** @brief Unique pointer to run bp implementation object for optimized 
   *  implementation with loop iters templated */
  std::unique_ptr<RunBpOnStereoSet<
    T,
    beliefprop::kStereoSetsToProcess[NUM_INPUT].num_disp_vals,
    OPT_IMP_ACCEL>>
  run_opt_bp_num_iters_templated_;

  /** @brief Unique pointer to run bp implementation object for optimized
   *  implementation with loop iters not templated */
  std::unique_ptr<RunBpOnStereoSet<T, 0, OPT_IMP_ACCEL>> run_opt_bp_num_iters_no_template_;

  /** @brief Bp parameter settings */
  beliefprop::BpSettings alg_settings_;
};

//run and evaluate optimized belief propagation implementation on evaluation stereo set
//specified by NUM_INPUT
//data type used in implementation specified by T
//bp implemenation optimization specified by OPT_IMP_ACCEL
//evaluation stereo set to run implementation on specified by NUM_INPUT
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
MultRunData RunImpOnInputBp<T, OPT_IMP_ACCEL, NUM_INPUT>::operator()(
  const run_environment::RunImpSettings& run_imp_settings)
{
  //set up BP settings for current run
  alg_settings_.num_disp_vals = beliefprop::kStereoSetsToProcess[NUM_INPUT].num_disp_vals;
  alg_settings_.disc_k_bp = (float)alg_settings_.num_disp_vals / 7.5f;

  //initialize run results across multiple implementations
  MultRunData run_results;

  //set up unoptimized single threaded bp stereo implementation
  run_bp_stereo_single_thread_ =
    std::make_unique<RunBpOnStereoSetSingleThreadCPU<T, beliefprop::kStereoSetsToProcess[NUM_INPUT].num_disp_vals, run_environment::AccSetting::kNone>>();

  //set up and run bp stereo using optimized implementation (optimized CPU and CUDA implementations supported)
  //as well as unoptimized implementation for comparison
  //run optimized implementation with and/or without disparity value count templated depending on setting
  //for running implementation with disparity values templated
  if (run_imp_settings.templated_iters_setting != run_environment::TemplatedItersSetting::kRunOnlyNonTemplated) {
    run_opt_bp_num_iters_templated_ =
      std::make_unique<RunBpOptimized<T, beliefprop::kStereoSetsToProcess[NUM_INPUT].num_disp_vals, OPT_IMP_ACCEL>>();
    constexpr bool run_w_loop_iters_templated{true};
    InputSignature input_sig(sizeof(T), NUM_INPUT, run_w_loop_iters_templated);
    run_results.insert({input_sig, (this->RunEvalBenchmark(run_imp_settings, run_w_loop_iters_templated))});
  }
  if (run_imp_settings.templated_iters_setting != run_environment::TemplatedItersSetting::kRunOnlyTempated) {
    run_opt_bp_num_iters_no_template_ =
      std::make_unique<RunBpOptimized<T, 0, OPT_IMP_ACCEL>>();
    constexpr bool run_w_loop_iters_templated{false};
    InputSignature input_sig(sizeof(T), NUM_INPUT, run_w_loop_iters_templated);
    run_results.insert({input_sig, this->RunEvalBenchmark(run_imp_settings, run_w_loop_iters_templated)});
  }

  //return bp run results across multiple implementations
  return run_results; 
}

//set up parallel parameters for running belief propagation in parallel on CPU or GPU
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
std::shared_ptr<ParallelParams> RunImpOnInputBp<T, OPT_IMP_ACCEL, NUM_INPUT>::SetUpParallelParams(
  const run_environment::RunImpSettings& run_imp_settings) const
{
  //parallel parameters initialized with default thread count dimensions at every level
  return std::make_shared<ParallelParamsBp>(
    run_imp_settings.opt_parallel_params_setting,
    alg_settings_.num_levels,
    run_imp_settings.p_params_default_alt_options.first);
}

//get input data and parameter info about current benchmark (belief propagation in this case) and return as RunData type
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
RunData RunImpOnInputBp<T, OPT_IMP_ACCEL, NUM_INPUT>::InputAndParamsForCurrBenchmark(
  bool loop_iters_templated) const
{
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(
    std::string(beliefprop::kStereoSetHeader), std::string(beliefprop::kStereoSetsToProcess[NUM_INPUT].name));
  curr_run_data.AppendData(this->InputAndParamsRunData(loop_iters_templated));
  curr_run_data.AppendData(alg_settings_.AsRunData());
  curr_run_data.AppendData(beliefprop::RunSettings());
  return curr_run_data;
}

//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
//on the reference and test images specified by numStereoSet
//run only optimized implementation if run_opt_imp_only is true
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
std::optional<RunData> RunImpOnInputBp<T, OPT_IMP_ACCEL, NUM_INPUT>::RunImpsAndCompare(
  std::shared_ptr<ParallelParams> parallel_params,
  bool run_opt_imp_only,
  bool run_imp_templated_loop_iters) const
{
  //get properties of input stereo set from stereo set number
  BpFileHandling bp_file_settings(
    std::string(beliefprop::kStereoSetsToProcess[NUM_INPUT].name));
  const std::array<filepathtype, 2> ref_test_image_path{
    bp_file_settings.RefImagePath(), bp_file_settings.TestImagePath()};

  //get number of implementations to run output disparity map file path(s)
  //if run_opt_imp_only is false, run single-threaded implementation in
  //addition to optimized implementation
  const unsigned int num_imps_run{run_opt_imp_only ? 1u : 2u};
  std::array<filepathtype, 2> output_disp;
  for (unsigned int i=0; i < num_imps_run; i++) {
    output_disp[i] = bp_file_settings.GetCurrentOutputDisparityFilePathAndIncrement();
  }

  //get optimized implementation description and write info about run to std::cout stream
  const std::string opt_imp_run_description{run_imp_templated_loop_iters ?
    run_opt_bp_num_iters_templated_->BpRunDescription() :
    run_opt_bp_num_iters_no_template_->BpRunDescription()};
  std::cout << "Running belief propagation on reference image "
            << ref_test_image_path[0] << " and test image "
            << ref_test_image_path[1] << " on " << opt_imp_run_description;
  if (!run_opt_imp_only) {
    std::cout << " and " << run_bp_stereo_single_thread_->BpRunDescription();
  }
  std::cout << std::endl;
  std::cout << "Data size: " << sizeof(T) << std::endl;
  std::cout << "run_imp_templated_loop_iters: " << run_imp_templated_loop_iters << std::endl;
  std::cout << "Acceleration: " << run_environment::AccelerationString<OPT_IMP_ACCEL>() << std::endl;
  std::cout << std::endl;
      
  //run optimized implementation and retrieve structure with runtime and output disparity map
  std::map<run_environment::AccSetting, std::optional<beliefprop::BpRunOutput>> run_output;
  if (run_imp_templated_loop_iters) {
    run_output[OPT_IMP_ACCEL] = run_opt_bp_num_iters_templated_->operator()(
      {ref_test_image_path[0].string(), ref_test_image_path[1].string()},
      alg_settings_,
      *parallel_params);
  }
  else {
    run_output[OPT_IMP_ACCEL] = run_opt_bp_num_iters_no_template_->operator()(
      {ref_test_image_path[0].string(), ref_test_image_path[1].string()},
      alg_settings_,
      *parallel_params);
  }
    
  //check if error in run
  RunData run_data;
  run_data.AddDataWHeader(
    std::string(run_environment::kAccelerationDescHeader), opt_imp_run_description);
  if (!(run_output[OPT_IMP_ACCEL])) {
    return {};
  }

  //append implementation run data to evaluation run data
  run_data.AppendData(run_output[OPT_IMP_ACCEL]->run_data);
  run_data.AddDataWHeader(
    std::string(run_eval::kOptimizedRuntimeHeader), run_output[OPT_IMP_ACCEL]->run_time.count());

  //save resulting disparity map
  run_output[OPT_IMP_ACCEL]->out_disparity_map.SaveDisparityMap(
    output_disp[0].string(),
    beliefprop::kStereoSetsToProcess[NUM_INPUT].scale_factor);

  //check if only running optimized implementation or if also running single-threaded implementation
  if (!run_opt_imp_only) {
    //check if running single threaded implementation or using stored results from previous run
    if ((!(bp_single_thread::kRunSingleThreadOnceForSet)) ||
        (!(bp_single_thread::single_thread_run_output.contains(ref_test_image_path))))
    {
      //run single-threaded implementation and retrieve structure with runtime and output disparity map
      run_output[run_environment::AccSetting::kNone] = run_bp_stereo_single_thread_->operator()(
        {ref_test_image_path[0].string(), ref_test_image_path[1].string()},
        alg_settings_, *parallel_params);
      if (!(run_output[run_environment::AccSetting::kNone])) {
        return {};
      }

      //save resulting disparity map
      run_output[run_environment::AccSetting::kNone]->out_disparity_map.SaveDisparityMap(
        output_disp[1].string(),
        beliefprop::kStereoSetsToProcess[NUM_INPUT].scale_factor);

      //save run data and result if setting to only run single thread implementation once for each set
      if (bp_single_thread::kRunSingleThreadOnceForSet) {
        bp_single_thread::single_thread_run_output[ref_test_image_path] =
          {run_output[run_environment::AccSetting::kNone]->run_time,
           run_output[run_environment::AccSetting::kNone]->run_data,
           output_disp[1]};
      }
    }
    else {
      run_output[run_environment::AccSetting::kNone] = beliefprop::BpRunOutput();
      //retrieve stored results from previous run on single threaded implementation
      run_output[run_environment::AccSetting::kNone]->run_time =
        std::get<0>(bp_single_thread::single_thread_run_output[ref_test_image_path]);
      run_output[run_environment::AccSetting::kNone]->run_data =
        std::get<1>(bp_single_thread::single_thread_run_output[ref_test_image_path]);
      run_output[run_environment::AccSetting::kNone]->out_disparity_map = 
        DisparityMap<float>(
          std::get<2>(bp_single_thread::single_thread_run_output[ref_test_image_path]).string(),
          beliefprop::kStereoSetsToProcess[NUM_INPUT].scale_factor);
    }
    //append run data for single thread run to evaluation run data
    run_data.AppendData(run_output[run_environment::AccSetting::kNone]->run_data);
  }

  //write location of output disparity maps for each implementation to std::cout
  for (unsigned int i = 0; i < num_imps_run; i++) {
    const std::string run_description{(i == 0) ?
      opt_imp_run_description :
      run_bp_stereo_single_thread_->BpRunDescription()};
    std::cout << "Output disparity map from " << run_description 
              << " run at " << output_disp[i] << std::endl;
  }
  std::cout << std::endl;

  //compare resulting disparity maps with ground truth and to each other
  const filepathtype ground_truth_disp{bp_file_settings.GroundTruthDisparityFilePath()};
  const DisparityMap<float> ground_truth_disparity_map(
    ground_truth_disp.string(),
    beliefprop::kStereoSetsToProcess[NUM_INPUT].scale_factor);
  run_data.AddDataWHeader(opt_imp_run_description + " output vs. Ground Truth result", std::string());
  run_data.AppendData(run_output[OPT_IMP_ACCEL]->out_disparity_map.OutputComparison(
    ground_truth_disparity_map, beliefprop::DisparityMapEvaluationParams()).AsRunData());
  if (!run_opt_imp_only) {
    run_data.AddDataWHeader(
      run_bp_stereo_single_thread_->BpRunDescription() + " output vs. Ground Truth result", std::string());
    run_data.AppendData(
      run_output[run_environment::AccSetting::kNone]->out_disparity_map.OutputComparison(
        ground_truth_disparity_map, beliefprop::DisparityMapEvaluationParams()).AsRunData());
    run_data.AddDataWHeader(
      opt_imp_run_description + " output vs. " + run_bp_stereo_single_thread_->BpRunDescription() +
      " result", std::string());
    run_data.AppendData(run_output[OPT_IMP_ACCEL]->out_disparity_map.OutputComparison(
      run_output[run_environment::AccSetting::kNone]->out_disparity_map,
      beliefprop::DisparityMapEvaluationParams()).AsRunData());
  }

  //return structure indicating that run succeeded along with data from run
  return run_data;
}

#endif //RUN_IMP_ON_INPUT_BP_H_