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
  constexpr bool RUN_SINGLE_THREAD_ONCE_FOR_SET{true};
  inline std::map<std::array<filepathtype, 2>, std::tuple<std::chrono::duration<double>, RunData, filepathtype>> singleThreadRunOutput;
};

//run and evaluate belief propagation implementation on a specified input
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
class RunEvalBPImpOnInput : public RunEvalImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT> {
public:
  MultRunData operator()(const run_environment::RunImpSettings& runImpSettings) override;
  
protected:
  //set up parallel parameters for running belief propagation in parallel on CPU or GPU
  std::shared_ptr<ParallelParams> setUpParallelParams(const run_environment::RunImpSettings& runImpSettings) const override;

  //get input data and parameter info about current benchmark (belief propagation in this case) and return as RunData type
  RunData inputAndParamsForCurrBenchmark(bool loopItersTemplated) const override;

  //run and compare output disparity maps using the given optimized and single-threaded stereo implementations
  //on the reference and test images specified by numStereoSet
  //run only optimized implementation if runOptImpOnly is true
  std::optional<RunData> runImpsAndCompare(std::shared_ptr<ParallelParams> parallelParams, bool runOptImpOnly,
    bool runImpTmpLoopIters) const override;

private:
  std::unique_ptr<RunBpStereoSet<T, bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].numDispVals_, run_environment::AccSetting::NONE>> runBpStereoSingleThread_;
  std::unique_ptr<RunBpStereoSet<T, bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].numDispVals_, OPT_IMP_ACCEL>> runOptBpNumItersTemplated_;
  std::unique_ptr<RunBpStereoSet<T, 0, OPT_IMP_ACCEL>> runOptBpNumItersNoTemplate_;
  beliefprop::BPsettings algSettings_;
};

template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
MultRunData RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::operator()(const run_environment::RunImpSettings& runImpSettings) {
  //set up BP settings for current run
  algSettings_.numDispVals_ = bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].numDispVals_;
  algSettings_.disc_k_bp_ = (float)algSettings_.numDispVals_ / 7.5f;

  MultRunData runResults;
  runBpStereoSingleThread_ = std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].numDispVals_>>();
  //RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
  if (runImpSettings.templatedItersSetting_ != run_environment::TemplatedItersSetting::RUN_ONLY_NON_TEMPLATED) {
    runOptBpNumItersTemplated_ = std::make_unique<RunBpOptimized<T, bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].numDispVals_, OPT_IMP_ACCEL>>();
    constexpr bool runWLoopItersTemplated{true};
    runResults.push_back(this->runEvalBenchmark(runImpSettings, runWLoopItersTemplated));
  }
  if (runImpSettings.templatedItersSetting_ != run_environment::TemplatedItersSetting::RUN_ONLY_TEMPLATED) {
    runOptBpNumItersNoTemplate_ = std::make_unique<RunBpOptimized<T, 0, OPT_IMP_ACCEL>>();
    constexpr bool runWLoopItersTemplated{false};
    runResults.push_back(this->runEvalBenchmark(runImpSettings, runWLoopItersTemplated));
  }
  return runResults; 
}

//set up parallel parameters for running belief propagation in parallel on CPU or GPU
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
std::shared_ptr<ParallelParams> RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::setUpParallelParams(const run_environment::RunImpSettings& runImpSettings) const {
  //parallel parameters initialized with default thread count dimensions at every level
  std::shared_ptr<ParallelParams> parallelParams = std::make_shared<BpParallelParams>(runImpSettings.optParallelParamsOptionSetting_.second,
  algSettings_.numLevels_, runImpSettings.pParamsDefaultOptOptions_.first);
  return parallelParams;
}

//get input data and parameter info about current benchmark (belief propagation in this case) and return as RunData type
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
RunData RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::inputAndParamsForCurrBenchmark(bool loopItersTemplated) const {
  RunData currRunData;
  currRunData.addDataWHeader(std::string(belief_prop::STEREO_SET_HEADER), std::string(bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].name_));
  currRunData.appendData(this->inputAndParamsRunData(loopItersTemplated));
  currRunData.appendData(algSettings_.runData());
  currRunData.appendData(bp_params::runSettings());
  return currRunData;
}

//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
  //on the reference and test images specified by numStereoSet
  //run only optimized implementation if runOptImpOnly is true
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
std::optional<RunData> RunEvalBPImpOnInput<T, OPT_IMP_ACCEL, NUM_INPUT>::runImpsAndCompare(
  std::shared_ptr<ParallelParams> parallelParams, bool runOptImpOnly, bool runImpTmpLoopIters) const
{ 
  const std::string optImpRunDesc{runImpTmpLoopIters ? runOptBpNumItersTemplated_->getBpRunDescription() : runOptBpNumItersNoTemplate_->getBpRunDescription()};
  const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
  BpFileHandling bpFileSettings(std::string(bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].name_));
  const std::array<filepathtype, 2> refTestImagePath{bpFileSettings.getRefImagePath(), bpFileSettings.getTestImagePath()};
  std::array<filepathtype, 2> output_disp;
  for (unsigned int i=0; i < numImpsRun; i++) {
    output_disp[i] = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
  }

  std::cout << "Running belief propagation on reference image " << refTestImagePath[0] << " and test image "
            << refTestImagePath[1] << " on " << optImpRunDesc;
  if (!runOptImpOnly) {
    std::cout << " and " << runBpStereoSingleThread_->getBpRunDescription();
  }
  std::cout << std::endl;
      
  //run optimized implementation and retrieve structure with runtime and output disparity map
  std::array<std::optional<ProcessStereoSetOutput>, 2> run_output;
  if (runImpTmpLoopIters) {
    run_output[0] = runOptBpNumItersTemplated_->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings_, *parallelParams);
  }
  else {
    run_output[0] = runOptBpNumItersNoTemplate_->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings_, *parallelParams);
  }
    
  //check if error in run
  RunData runData;
  runData.addDataWHeader("Acceleration", optImpRunDesc);
  if (!(run_output[0])) {
    return {};
  }
  runData.appendData(run_output[0]->runData);

  //save resulting disparity map
  run_output[0]->outDisparityMap.saveDisparityMap(output_disp[0].string(), bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].scaleFactor_);
  runData.addDataWHeader(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER), run_output[0]->runTime.count());

  if (!runOptImpOnly) {
    //run single-threaded implementation and retrieve structure with runtime and output disparity map
    if ((!(bpSingleThread::RUN_SINGLE_THREAD_ONCE_FOR_SET)) || (!(bpSingleThread::singleThreadRunOutput.contains(refTestImagePath)))) {
      run_output[1] = runBpStereoSingleThread_->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings_, *parallelParams);
      if (!(run_output[1])) {
        return {};
      }
      run_output[1]->outDisparityMap.saveDisparityMap(output_disp[1].string(), bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].scaleFactor_);
      if (bpSingleThread::RUN_SINGLE_THREAD_ONCE_FOR_SET) {
        bpSingleThread::singleThreadRunOutput[refTestImagePath] = {run_output[1]->runTime, run_output[1]->runData, output_disp[1]};
      }
    }
    else {
      run_output[1]->runTime = std::get<0>(bpSingleThread::singleThreadRunOutput[refTestImagePath]);
      run_output[1]->runData = std::get<1>(bpSingleThread::singleThreadRunOutput[refTestImagePath]);
      run_output[1]->outDisparityMap = DisparityMap<float>(
        std::get<2>(bpSingleThread::singleThreadRunOutput[refTestImagePath]).string(),
        bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].scaleFactor_);
    }
  }
  runData.appendData(run_output[1]->runData);

  for (unsigned int i = 0; i < numImpsRun; i++) {
    const std::string runDesc{(i == 0) ? optImpRunDesc : runBpStereoSingleThread_->getBpRunDescription()};
    std::cout << "Output disparity map from " << runDesc << " run at " << output_disp[i] << std::endl;
  }
  std::cout << std::endl;

  //compare resulting disparity maps with ground truth and to each other
  const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};
  DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::STEREO_SETS_TO_PROCESS[NUM_INPUT].scaleFactor_);
  runData.addDataWHeader(optImpRunDesc + " output vs. Ground Truth result", std::string());
  runData.appendData(run_output[0]->outDisparityMap.getOutputComparison(groundTruthDisparityMap, OutputEvaluationParameters()).runData());
  if (!runOptImpOnly) {
    runData.addDataWHeader(runBpStereoSingleThread_->getBpRunDescription() + " output vs. Ground Truth result", std::string());
    runData.appendData(run_output[1]->outDisparityMap.getOutputComparison(groundTruthDisparityMap, OutputEvaluationParameters()).runData());
    runData.addDataWHeader(optImpRunDesc + " output vs. " + runBpStereoSingleThread_->getBpRunDescription() + " result", std::string());
    runData.appendData(run_output[0]->outDisparityMap.getOutputComparison(run_output[1]->outDisparityMap, OutputEvaluationParameters()).runData());
  }

  //return structure indicating that run succeeded along with data from run
  return runData;
}

#endif //RUN_EVAL_BP_IMP_SINGLE_SET_H_