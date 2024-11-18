/*
 * RunEvalImpOnInput.h
 *
 *  Created on: Feb 6, 2024
 *      Author: scott
 */

#include <memory>
#include <filesystem>
#include <optional>
#include <array>
#include <map>
#include "BpConstsAndParams/BpConsts.h"
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpFileProcessing/BpFileHandling.h"
#include "BpSingleThreadCPU/stereo.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunImp/RunEvalImpOnInput.h"

#ifndef RUN_EVAL_BP_IMP_SINGLE_SET_H_
#define RUN_EVAL_BP_IMP_SINGLE_SET_H_

using filepathtype = std::filesystem::path;

//check if optimized CPU run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CPU_RUN
//needed to run the optimized implementation a stereo set using CPU
#include "BpOptimizeCPU/RunBpStereoOptimizedCPU.h"
//set RunBpOptimized alias to correspond to optimized CPU implementation
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
using RunBpOptimized = RunBpStereoOptimizedCPU<T, DISP_VALS, ACCELERATION>;
#endif //OPTIMIZED_CPU_RUN

//check if CUDA run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CUDA_RUN
//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "BpConstsAndParams/BpStereoCudaParameters.h"
//needed to run the implementation a stereo set using CUDA
#include "BpOptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
//set RunBpOptimized alias to correspond to CUDA implementation
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
using RunBpOptimized = RunBpStereoSetOnGPUWithCUDA<T, DISP_VALS, ACCELERATION>;
#endif //OPTIMIZED_CUDA_RUN

namespace bpSingleThread {
  constexpr bool kRunSingleThreadOnceForSet{true};
  inline std::map<std::array<filepathtype, 2>, std::tuple<std::chrono::duration<double>, RunData, filepathtype>> singleThreadRunOutput;
};

//run and evaluate belief propagation implementation on a specified input
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
class RunEvalBPImpOnInput final : public RunEvalImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT> {
public:
  MultRunData operator()(const run_environment::RunImpSettings& run_imp_settings) override;
  
protected:
  //set up parallel parameters for running belief propagation in parallel on CPU or GPU
  std::shared_ptr<ParallelParams> SetUpParallelParams(const run_environment::RunImpSettings& run_imp_settings) const override;

  //get input data and parameter info about current benchmark (belief propagation in this case) and return as RunData type
  RunData InputAndParamsForCurrBenchmark(bool loopItersTemplated) const override;

  //run and compare output disparity maps using the given optimized and single-threaded stereo implementations
  //on the reference and test images specified by numStereoSet
  //run only optimized implementation if runOptImpOnly is true
  std::optional<RunData> RunImpsAndCompare(std::shared_ptr<ParallelParams> parallelParams, bool runOptImpOnly,
    bool runImpTmpLoopIters) const override;

private:
  std::unique_ptr<RunBpStereoSet<T, bp_params::kStereoSetsToProcess[NUM_INPUT].num_disp_vals_, run_environment::AccSetting::kNone>> run_bp_stereo_single_thread_;
  std::unique_ptr<RunBpStereoSet<T, bp_params::kStereoSetsToProcess[NUM_INPUT].num_disp_vals_, OPT_IMP_ACCEL>> run_opt_bp_num_iters_templated_;
  std::unique_ptr<RunBpStereoSet<T, 0, OPT_IMP_ACCEL>> run_opt_bp_num_iters_no_template_;
  beliefprop::BPsettings alg_settings_;
};

template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
MultRunData RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::operator()(const run_environment::RunImpSettings& run_imp_settings) {
  //set up BP settings for current run
  alg_settings_.num_disp_vals_ = bp_params::kStereoSetsToProcess[NUM_INPUT].num_disp_vals_;
  alg_settings_.disc_k_bp_ = (float)alg_settings_.num_disp_vals_ / 7.5f;

  MultRunData run_results;
  run_bp_stereo_single_thread_ = std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::kStereoSetsToProcess[NUM_INPUT].num_disp_vals_>>();
  //RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
  if (run_imp_settings.templatedItersSetting_ != run_environment::TemplatedItersSetting::kRunOnlyNonTemplated) {
    run_opt_bp_num_iters_templated_ = std::make_unique<RunBpOptimized<T, bp_params::kStereoSetsToProcess[NUM_INPUT].num_disp_vals_, OPT_IMP_ACCEL>>();
    constexpr bool runWLoopItersTemplated{true};
    run_results.push_back(this->RunEvalBenchmark(run_imp_settings, runWLoopItersTemplated));
  }
  if (run_imp_settings.templatedItersSetting_ != run_environment::TemplatedItersSetting::kRunOnlyTempated) {
    run_opt_bp_num_iters_no_template_ = std::make_unique<RunBpOptimized<T, 0, OPT_IMP_ACCEL>>();
    constexpr bool runWLoopItersTemplated{false};
    run_results.push_back(this->RunEvalBenchmark(run_imp_settings, runWLoopItersTemplated));
  }
  return run_results; 
}

//set up parallel parameters for running belief propagation in parallel on CPU or GPU
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
std::shared_ptr<ParallelParams> RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::SetUpParallelParams(const run_environment::RunImpSettings& run_imp_settings) const {
  //parallel parameters initialized with default thread count dimensions at every level
  std::shared_ptr<ParallelParams> parallelParams = std::make_shared<BpParallelParams>(run_imp_settings.optParallelParamsOptionSetting_.second,
  alg_settings_.num_levels_, run_imp_settings.pParamsDefaultOptOptions_.first);
  return parallelParams;
}

//get input data and parameter info about current benchmark (belief propagation in this case) and return as RunData type
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
RunData RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::InputAndParamsForCurrBenchmark(bool loopItersTemplated) const {
  RunData curr_run_data;
  curr_run_data.AddDataWHeader(std::string(belief_prop::kStereoSetHeader), std::string(bp_params::kStereoSetsToProcess[NUM_INPUT].name_));
  curr_run_data.AppendData(this->inputAndParamsRunData(loopItersTemplated));
  curr_run_data.AppendData(alg_settings_.AsRunData());
  curr_run_data.AppendData(bp_params::runSettings());
  return curr_run_data;
}

//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
  //on the reference and test images specified by numStereoSet
  //run only optimized implementation if runOptImpOnly is true
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
std::optional<RunData> RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::RunImpsAndCompare(
  std::shared_ptr<ParallelParams> parallelParams, bool runOptImpOnly, bool runImpTmpLoopIters) const
{ 
  const std::string optImpRunDesc{runImpTmpLoopIters ? run_opt_bp_num_iters_templated_->BpRunDescription() : run_opt_bp_num_iters_no_template_->BpRunDescription()};
  const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
  BpFileHandling bpFileSettings(std::string(bp_params::kStereoSetsToProcess[NUM_INPUT].name_));
  const std::array<filepathtype, 2> refTestImagePath{bpFileSettings.RefImagePath(), bpFileSettings.TestImagePath()};
  std::array<filepathtype, 2> output_disp;
  for (unsigned int i=0; i < numImpsRun; i++) {
    output_disp[i] = bpFileSettings.GetCurrentOutputDisparityFilePathAndIncrement();
  }

  std::cout << "Running belief propagation on reference image " << refTestImagePath[0] << " and test image "
            << refTestImagePath[1] << " on " << optImpRunDesc;
  if (!runOptImpOnly) {
    std::cout << " and " << run_bp_stereo_single_thread_->BpRunDescription();
  }
  std::cout << std::endl;
      
  //run optimized implementation and retrieve structure with runtime and output disparity map
  std::array<std::optional<ProcessStereoSetOutput>, 2> run_output;
  if (runImpTmpLoopIters) {
    run_output[0] = run_opt_bp_num_iters_templated_->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, alg_settings_, *parallelParams);
  }
  else {
    run_output[0] = run_opt_bp_num_iters_no_template_->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, alg_settings_, *parallelParams);
  }
    
  //check if error in run
  RunData run_data;
  run_data.AddDataWHeader("Acceleration", optImpRunDesc);
  if (!(run_output[0])) {
    return {};
  }
  run_data.AppendData(run_output[0]->run_data);

  //save resulting disparity map
  run_output[0]->out_disparity_map.SaveDisparityMap(output_disp[0].string(), bp_params::kStereoSetsToProcess[NUM_INPUT].scale_factor_);
  run_data.AddDataWHeader(std::string(run_eval::kOptimizedRuntimeHeader), run_output[0]->run_time.count());

  if (!runOptImpOnly) {
    //run single-threaded implementation and retrieve structure with runtime and output disparity map
    if ((!(bpSingleThread::kRunSingleThreadOnceForSet)) || (!(bpSingleThread::singleThreadRunOutput.contains(refTestImagePath)))) {
      run_output[1] = run_bp_stereo_single_thread_->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, alg_settings_, *parallelParams);
      if (!(run_output[1])) {
        return {};
      }
      run_output[1]->out_disparity_map.SaveDisparityMap(output_disp[1].string(), bp_params::kStereoSetsToProcess[NUM_INPUT].scale_factor_);
      if (bpSingleThread::kRunSingleThreadOnceForSet) {
        bpSingleThread::singleThreadRunOutput[refTestImagePath] = {run_output[1]->run_time, run_output[1]->run_data, output_disp[1]};
      }
    }
    else {
      run_output[1]->run_time = std::get<0>(bpSingleThread::singleThreadRunOutput[refTestImagePath]);
      run_output[1]->run_data = std::get<1>(bpSingleThread::singleThreadRunOutput[refTestImagePath]);
      run_output[1]->out_disparity_map = DisparityMap<float>(
        std::get<2>(bpSingleThread::singleThreadRunOutput[refTestImagePath]).string(),
        bp_params::kStereoSetsToProcess[NUM_INPUT].scale_factor_);
    }
  }
  run_data.AppendData(run_output[1]->run_data);

  for (unsigned int i = 0; i < numImpsRun; i++) {
    const std::string runDesc{(i == 0) ? optImpRunDesc : run_bp_stereo_single_thread_->BpRunDescription()};
    std::cout << "Output disparity map from " << runDesc << " run at " << output_disp[i] << std::endl;
  }
  std::cout << std::endl;

  //compare resulting disparity maps with ground truth and to each other
  const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};
  DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::kStereoSetsToProcess[NUM_INPUT].scale_factor_);
  run_data.AddDataWHeader(optImpRunDesc + " output vs. Ground Truth result", std::string());
  run_data.AppendData(run_output[0]->out_disparity_map.OutputComparison(groundTruthDisparityMap, BpEvaluationParameters()).AsRunData());
  if (!runOptImpOnly) {
    run_data.AddDataWHeader(run_bp_stereo_single_thread_->BpRunDescription() + " output vs. Ground Truth result", std::string());
    run_data.AppendData(run_output[1]->out_disparity_map.OutputComparison(groundTruthDisparityMap, BpEvaluationParameters()).AsRunData());
    run_data.AddDataWHeader(optImpRunDesc + " output vs. " + run_bp_stereo_single_thread_->BpRunDescription() + " result", std::string());
    run_data.AppendData(run_output[0]->out_disparity_map.OutputComparison(run_output[1]->out_disparity_map, BpEvaluationParameters()).AsRunData());
  }

  //return structure indicating that run succeeded along with data from run
  return run_data;
}

#endif //RUN_EVAL_BP_IMP_SINGLE_SET_H_