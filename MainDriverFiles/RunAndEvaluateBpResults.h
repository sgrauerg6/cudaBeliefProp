/*
 * RunAndEvaluateBpResults.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNANDEVALUATEBPRESULTS_H_
#define RUNANDEVALUATEBPRESULTS_H_

#include <memory>
#include <array>
#include <fstream>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include <set>
#include "BpFileProcessing/BpFileHandling.h"
#include "BpConstsAndParams/bpStereoParameters.h"
#include "BpConstsAndParams/bpTypeConstraints.h"
#include "BpConstsAndParams/DetailedTimingBPConsts.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"
#include "RunSettingsEval/RunEvalUtils.h"
#include "BpSingleThreadCPU/stereo.h"
#include "BpRunProcessing/RunBpStereoSet.h"

typedef std::filesystem::path filepathtype;

//check if optimized CPU run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CPU_RUN
//needed to run the optimized implementation a stereo set using CPU
#include "BpOptimizeCPU/RunBpStereoOptimizedCPU.h"
//set RunBpOptimized alias to correspond to optimized CPU implementation
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
using RunBpOptimized = RunBpStereoOptimizedCPU<T, DISP_VALS, ACCELERATION>;
//set data type used for half-precision
#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
using halftype = float16_t;
#else
using halftype = short;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CPU_RUN

//check if CUDA run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CUDA_RUN
//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "BpConstsAndParams/bpStereoCudaParameters.h"
//needed to run the implementation a stereo set using CUDA
#include "BpOptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
//set RunBpOptimized alias to correspond to CUDA implementation
template <RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
using RunBpOptimized = RunBpStereoSetOnGPUWithCUDA<T, DISP_VALS, run_environment::AccSetting::CUDA>;
#endif //OPTIMIZED_CUDA_RUN

using MultRunData = std::vector<std::pair<run_eval::Status, std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::array<double, 2>>;

namespace RunAndEvaluateBpResults {

//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
//on the reference and test images specified by numStereoSet
//run only optimized implementation if runOptImpOnly is true
template<RunData_t T, unsigned int DISP_VALS_OPTIMIZED, unsigned int DISP_VALS_SINGLE_THREAD, run_environment::AccSetting OPT_IMP_ACCEL>
  std::pair<run_eval::Status, RunData> runStereoTwoImpsAndCompare(
  const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_OPTIMIZED, OPT_IMP_ACCEL>>& optimizedImp,
  const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_SINGLE_THREAD, run_environment::AccSetting::NONE>>& singleThreadCPUImp,
  const unsigned int numStereoSet, const beliefprop::BPsettings& algSettings,
  const beliefprop::ParallelParameters& parallelParams,
  bool runOptImpOnly = false)
{
  const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
  BpFileHandling bpFileSettings(bp_params::STEREO_SET[numStereoSet]);
  const std::array<filepathtype, 2> refTestImagePath{bpFileSettings.getRefImagePath(), bpFileSettings.getTestImagePath()};
  std::array<filepathtype, 2> output_disp;
  for (unsigned int i=0; i < numImpsRun; i++) {
    output_disp[i] = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
  }

  std::cout << "Running belief propagation on reference image " << refTestImagePath[0] << " and test image "
            << refTestImagePath[1] << " on " << optimizedImp->getBpRunDescription();
  if (!runOptImpOnly) {
    std::cout << " and " << singleThreadCPUImp->getBpRunDescription();
  }
  std::cout << std::endl;
    
  //run optimized implementation and retrieve structure with runtime and output disparity map
  std::array<ProcessStereoSetOutput, 2> run_output;
  run_output[0] = optimizedImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, parallelParams);
  
  //check if error in run
  RunData runData;
  if ((run_output[0].runTime == 0.0) || (run_output[0].outDisparityMap.getHeight() == 0)) {
    return {run_eval::Status::ERROR, runData};
  }
  runData.appendData(run_output[0].runData);

  //save resulting disparity map
  run_output[0].outDisparityMap.saveDisparityMap(output_disp[0].string(), bp_params::SCALE_BP[numStereoSet]);
  runData.addDataWHeader(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER), std::to_string(run_output[0].runTime));

  if (!runOptImpOnly) {
    //run single-threaded implementation and retrieve structure with runtime and output disparity map
    run_output[1] = singleThreadCPUImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, parallelParams);
    run_output[1].outDisparityMap.saveDisparityMap(output_disp[1].string(), bp_params::SCALE_BP[numStereoSet]);
    runData.appendData(run_output[1].runData);
  }

  for (unsigned int i = 0; i < numImpsRun; i++) {
    const std::string runDesc{(i == 0) ? optimizedImp->getBpRunDescription() : singleThreadCPUImp->getBpRunDescription()};
    std::cout << "Output disparity map from " << runDesc << " run at " << output_disp[i] << std::endl;
  }
  std::cout << std::endl;

  //compare resulting disparity maps with ground truth and to each other
  const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};
  DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::SCALE_BP[numStereoSet]);
  runData.addDataWHeader(optimizedImp->getBpRunDescription() + " output vs. Ground Truth result", std::string());
  runData.appendData(run_output[0].outDisparityMap.getOutputComparison(groundTruthDisparityMap, OutputEvaluationParameters()).runData());
  if (!runOptImpOnly) {
    runData.addDataWHeader(singleThreadCPUImp->getBpRunDescription() + " output vs. Ground Truth result", std::string());
    runData.appendData(run_output[1].outDisparityMap.getOutputComparison(groundTruthDisparityMap, OutputEvaluationParameters()).runData());

    runData.addDataWHeader(optimizedImp->getBpRunDescription() + " output vs. " + singleThreadCPUImp->getBpRunDescription() + " result", std::string());
    runData.appendData(run_output[0].outDisparityMap.getOutputComparison(run_output[1].outDisparityMap, OutputEvaluationParameters()).runData());
  }

  //return structure indicating that run succeeded along with data from run
  return {run_eval::Status::NO_ERROR, runData};
}


//run optimized and single threaded implementations using multiple sets of parallel parameters in optimized implementation if set to optimize parallel parameters
//returns data from runs using default and optimized parallel parameters
template<RunData_t T, unsigned int NUM_SET, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, unsigned int DISP_VALS_TEMPLATE_SINGLE_THREAD>
std::pair<run_eval::Status, std::vector<RunData>> runBpOnSetAndUpdateResults(
  const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_OPTIMIZED, OPT_IMP_ACCEL>>& optimizedImp,
  const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_SINGLE_THREAD, run_environment::AccSetting::NONE>>& singleThreadCPUImp)
{
  std::vector<RunData> outRunData(OPTIMIZE_PARALLEL_PARAMS ? 2 : 1);
  enum class RunType { ONLY_RUN, DEFAULT_PARAMS, OPTIMIZED_RUN, TEST_PARAMS };
  std::array<std::array<std::map<std::string, std::string>, 2>, 2> inParamsResultsDefOptRuns;
  //load all the BP default settings
  beliefprop::BPsettings algSettings;
  algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

  //parallel parameters initialized with default thread count dimensions at every level
  beliefprop::ParallelParameters parallelParams(algSettings.numLevels_, PARALLEL_PARAMS_DEFAULT);

  //if optimizing parallel parameters, parallelParamsVect contains parallel parameter settings to run
  //(and contains only the default parallel parameters if not)
  std::vector<std::array<unsigned int, 2>> parallelParamsVect{
    OPTIMIZE_PARALLEL_PARAMS ? PARALLEL_PARAMETERS_OPTIONS : std::vector<std::array<unsigned int, 2>>()};
    
  //mapping of parallel parameters to runtime for each kernel at each level and total runtime
  std::array<std::vector<std::map<std::array<unsigned int, 2>, double>>, (beliefprop::NUM_KERNELS + 1)> pParamsToRunTimeEachKernel;
  for (unsigned int i=0; i < beliefprop::NUM_KERNELS; i++) {
    //set to vector length for each kernel to corresponding vector length of kernel in parallelParams.parallelDimsEachKernel_
    pParamsToRunTimeEachKernel[i] = std::vector<std::map<std::array<unsigned int, 2>, double>>(parallelParams.parallelDimsEachKernel_[i].size()); 
  }
  pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS] = std::vector<std::map<std::array<unsigned int, 2>, double>>(1); 
    
  //if optimizing parallel parameters, run BP for each parallel parameters option, retrieve best parameters for each kernel or overall for the run,
  //and then run BP with best found parallel parameters
  //if not optimizing parallel parameters, run BP once using default parallel parameters
  for (unsigned int runNum=0; runNum < (parallelParamsVect.size() + 1); runNum++) {
    //initialize current run type to specify if current run is only run, run with default params, test params run, or final run with optiized params
    RunType currRunType{RunType::TEST_PARAMS};
    if constexpr (!OPTIMIZE_PARALLEL_PARAMS) {
      currRunType = RunType::ONLY_RUN;
    }
    else if (runNum == parallelParamsVect.size()) {
      currRunType = RunType::OPTIMIZED_RUN;
    }

    //get and set parallel parameters for current run if not final run that uses optimized parameters
    std::array<unsigned int, 2> pParamsCurrRun{PARALLEL_PARAMS_DEFAULT};
    if (currRunType == RunType::ONLY_RUN) {
      parallelParams.setParallelDims(PARALLEL_PARAMS_DEFAULT, algSettings.numLevels_);
    }
    else if (currRunType == RunType::TEST_PARAMS) {
      //set parallel parameters to parameters corresponding to current run for each BP processing level
      pParamsCurrRun = parallelParamsVect[runNum];
      parallelParams.setParallelDims(pParamsCurrRun, algSettings.numLevels_);
      if (pParamsCurrRun == PARALLEL_PARAMS_DEFAULT) {
        //set run type to default parameters if current run uses default parameters
        currRunType = RunType::DEFAULT_PARAMS;
      }
    }

    //store input params data if using default parallel parameters or final run with optimized parameters
    RunData currRunData;
    if (currRunType != RunType::TEST_PARAMS) {
      currRunData.appendData(run_eval::inputAndParamsRunData<T, NUM_SET, DISP_VALS_TEMPLATE_OPTIMIZED, OPT_IMP_ACCEL>(algSettings));
      if constexpr (OPTIMIZE_PARALLEL_PARAMS &&
        (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN))
      {
        //add parallel parameters for each kernel to current input data if allowing different parallel parameters for each kernel in the same run
        currRunData.appendData(parallelParams.runData());
      }
    }

    //run only optimized implementation and not single-threaded run if current run is not final run or is using default parameter parameters
    const bool runOptImpOnly{currRunType == RunType::TEST_PARAMS};

    //run belief propagation implementation(s) and return whether or not error in run
    //detailed results stored to file that is generated using stream
    const auto runImpsECodeData = runStereoTwoImpsAndCompare<T, DISP_VALS_TEMPLATE_OPTIMIZED, DISP_VALS_TEMPLATE_SINGLE_THREAD, OPT_IMP_ACCEL>(
      optimizedImp, singleThreadCPUImp, NUM_SET, algSettings, parallelParams, runOptImpOnly);
    currRunData.addDataWHeader("Run Success", (runImpsECodeData.first == run_eval::Status::NO_ERROR) ? "Yes" : "No");

    //if error in run and run is any type other than for testing parameters, exit function with error
    if ((runImpsECodeData.first != run_eval::Status::NO_ERROR) && (currRunType != RunType::TEST_PARAMS)) {
      return {run_eval::Status::ERROR, {currRunData}};
    }

    //retrieve results from current run
    currRunData.appendData(runImpsECodeData.second);

    //add current run results for output if using default parallel parameters or is final run w/ optimized parallel parameters
    if (currRunType != RunType::TEST_PARAMS) {
      //set output for runs using default parallel parameters and final run (which is the same run if not optimizing parallel parameters)
      if (currRunType == RunType::OPTIMIZED_RUN) {
        outRunData[1] = currRunData;
      }
      else {
        outRunData[0] = currRunData;
      }
    }

    if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
      //retrieve and store results including runtimes for each kernel if allowing different parallel parameters for each kernel and
      //total runtime for current run
      //if error in run, don't add results for current parallel parameters to results set
      const std::string NUM_RUNS_IN_PARENS{"(" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"};
      if (runImpsECodeData.first == run_eval::Status::NO_ERROR) {
        if (currRunType != RunType::OPTIMIZED_RUN) {
          if constexpr (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
            for (unsigned int level=0; level < algSettings.numLevels_; level++) {
              pParamsToRunTimeEachKernel[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][pParamsCurrRun] =
                std::stod(currRunData.getData(beliefprop::LEVEL_DCOST_BPTIME_CTIME_NAMES[level][0] + " " + NUM_RUNS_IN_PARENS));
              pParamsToRunTimeEachKernel[beliefprop::BpKernel::BP_AT_LEVEL][level][pParamsCurrRun] = 
                std::stod(currRunData.getData(beliefprop::LEVEL_DCOST_BPTIME_CTIME_NAMES[level][1] + " " + NUM_RUNS_IN_PARENS));
              pParamsToRunTimeEachKernel[beliefprop::BpKernel::COPY_AT_LEVEL][level][pParamsCurrRun] =
                std::stod(currRunData.getData(beliefprop::LEVEL_DCOST_BPTIME_CTIME_NAMES[level][2] + " " + NUM_RUNS_IN_PARENS));
            }
            pParamsToRunTimeEachKernel[beliefprop::BpKernel::BLUR_IMAGES][0][pParamsCurrRun] =
              std::stod(currRunData.getData(beliefprop::timingNames.at(beliefprop::Runtime_Type::SMOOTHING) + " " + NUM_RUNS_IN_PARENS));
            pParamsToRunTimeEachKernel[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][pParamsCurrRun] =
              std::stod(currRunData.getData(beliefprop::timingNames.at(beliefprop::Runtime_Type::INIT_MESSAGES_KERNEL) + " " + NUM_RUNS_IN_PARENS));
            pParamsToRunTimeEachKernel[beliefprop::BpKernel::OUTPUT_DISP][0][pParamsCurrRun] =
              std::stod(currRunData.getData(beliefprop::timingNames.at(beliefprop::Runtime_Type::OUTPUT_DISPARITY) + " " + NUM_RUNS_IN_PARENS));
          }
          //get total runtime
          pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0][pParamsCurrRun] =
            std::stod(currRunData.getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
        }
      }

      //get optimized parallel parameters if next run is final run that uses optimized parallel parameters
      if (runNum == (parallelParamsVect.size() - 1)) {
        if constexpr (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
          for (unsigned int numKernelSet = 0; numKernelSet < pParamsToRunTimeEachKernel.size(); numKernelSet++) {
            //retrieve and set optimized parallel parameters for final run
            //std::min_element used to retrieve parallel parameters corresponding to lowest runtime from previous runs
            std::transform(pParamsToRunTimeEachKernel[numKernelSet].begin(),
                           pParamsToRunTimeEachKernel[numKernelSet].end(), 
                           parallelParams.parallelDimsEachKernel_[numKernelSet].begin(),
                           [](const auto& tDimsToRunTimeCurrLevel) /*-> std::array<unsigned int, 2>*/ { 
                             return (std::min_element(tDimsToRunTimeCurrLevel.begin(), tDimsToRunTimeCurrLevel.end(),
                                                      [](const auto& a, const auto& b) { return a.second < b.second; }))->first; });
          }
        }
        else {
          //set optimized parallel parameters for all kernels to parallel parameters that got the best runtime across all kernels
          //seems like setting different parallel parameters for different kernels on GPU decrease runtime but increases runtime on CPU
          const auto bestParallelParams = std::min_element(pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0].begin(),
                                                           pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0].end(),
                                                           [](const auto& a, const auto& b) { return a.second < b.second; })->first;
          parallelParams.setParallelDims(bestParallelParams, algSettings.numLevels_);
        }
      }
    }
  }
    
  return {run_eval::Status::NO_ERROR, outRunData};
}

template<RunData_t T, unsigned int NUM_SET, bool TEMPLATED_DISP_IN_OPT_IMP, run_environment::AccSetting OPT_IMP_ACCEL>
std::pair<run_eval::Status, std::vector<RunData>> runBpOnSetAndUpdateResults() {
  std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], run_environment::AccSetting::NONE>> runBpStereoSingleThread =
    std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
  //RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
  if constexpr (TEMPLATED_DISP_IN_OPT_IMP) {
    std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], OPT_IMP_ACCEL>> runBpOptimizedImp = 
      std::make_unique<RunBpOptimized<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], OPT_IMP_ACCEL>>();
    return RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET, OPT_IMP_ACCEL>(runBpOptimizedImp, runBpStereoSingleThread);
  }
  else {
    std::unique_ptr<RunBpStereoSet<T, 0, OPT_IMP_ACCEL>> runBpOptimizedImp = 
      std::make_unique<RunBpOptimized<T, 0, OPT_IMP_ACCEL>>();
    return RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET, OPT_IMP_ACCEL>(runBpOptimizedImp, runBpStereoSingleThread);
  }
}

//perform runs on multiple data sets using specified data type and acceleration method
template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
std::pair<MultRunData, std::vector<MultRunSpeedup>> runBpOnStereoSets() {
  MultRunData runData;
  runData.push_back(runBpOnSetAndUpdateResults<T, 0, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 0, false, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 1, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 1, false, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 2, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 2, false, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 3, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 3, false, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 4, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 4, false, OPT_IMP_ACCEL>());
#ifndef SMALLER_SETS_ONLY
  runData.push_back(runBpOnSetAndUpdateResults<T, 5, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 5, false, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 6, true, OPT_IMP_ACCEL>());
  runData.push_back(runBpOnSetAndUpdateResults<T, 6, false, OPT_IMP_ACCEL>());
#endif //SMALLER_SETS_ONLY

  //initialize speedup results
  std::vector<MultRunSpeedup> speedupResults;

  //get speedup info for using optimized parallel parameters and disparity count as template parameter
  if (sizeof(T) == sizeof(float)) {
    const auto speedupOverBaseline = run_eval::getAvgMedSpeedupOverBaseline(runData, run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)));
    speedupResults.insert(speedupResults.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
  }
  if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
    speedupResults.push_back(run_eval::getAvgMedSpeedupOptPParams(runData, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - " +
      run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))));
  }
  speedupResults.push_back(run_eval::getAvgMedSpeedupDispValsInTemplate(runData, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - " +
    run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))));
    
  //write output corresponding to results for current data type
  constexpr bool MULT_DATA_TYPES{false};
  run_eval::writeRunOutput<OPT_IMP_ACCEL, MULT_DATA_TYPES, T>({runData, speedupResults});

  //return data for each run and multiple average and median speedup results across the data
  return {runData, speedupResults};
}

//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
std::pair<std::pair<MultRunData, std::vector<MultRunSpeedup>>, std::vector<MultRunSpeedup>> getAltAndNoVectSpeedup(MultRunData& runOutputData) {
  const std::string speedupHeader{std::string(run_eval::SPEEDUP_VECTORIZATION) + " - " + run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))};
  const std::string speedupVsAVX256Str{std::string(run_eval::SPEEDUP_VS_AVX256_VECTORIZATION) + " - " + run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))};
  std::vector<MultRunSpeedup> multRunSpeedupVect;
  if constexpr ((OPT_IMP_ACCEL == run_environment::AccSetting::CUDA) || (OPT_IMP_ACCEL == run_environment::AccSetting::NONE)) {
    multRunSpeedupVect.push_back({speedupHeader, {0.0, 0.0}});
    multRunSpeedupVect.push_back({speedupVsAVX256Str, {0.0, 0.0}});
    return {std::pair<MultRunData, std::vector<MultRunSpeedup>>(), multRunSpeedupVect};
  }
  else {
    //if initial speedup is AVX512, also run AVX256
    if (OPT_IMP_ACCEL == run_environment::AccSetting::AVX512) {
      auto runOutputAVX256 = runBpOnStereoSets<T, run_environment::AccSetting::AVX256>();
      //go through each result and replace initial run data with AVX256 run data if AVX256 run is faster
      for (unsigned int i = 0; i < runOutputData.size(); i++) {
        if ((runOutputData[i].first == run_eval::Status::NO_ERROR) && (runOutputAVX256.first[i].first == run_eval::Status::NO_ERROR)) {
          const double initResultTime = std::stod(runOutputData[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
          const double avx256ResultTime = std::stod(runOutputAVX256.first[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
          if (avx256ResultTime < initResultTime) {
            runOutputData[i] = runOutputAVX256.first[i];
          }
        }
      }
      const auto speedupOverAVX256 = run_eval::getAvgMedSpeedup(runOutputAVX256.first, runOutputData, speedupVsAVX256Str);
      multRunSpeedupVect.push_back(speedupOverAVX256);
    }
    else {
      multRunSpeedupVect.push_back({speedupVsAVX256Str, {0.0, 0.0}});
    }
    auto runOutputNoVect = runBpOnStereoSets<T, run_environment::AccSetting::NONE>();
    //go through each result and replace initial run data with no vectorization run data if no vectorization run is faster
    for (unsigned int i = 0; i < runOutputData.size(); i++) {
      if ((runOutputData[i].first == run_eval::Status::NO_ERROR) && (runOutputNoVect.first[i].first == run_eval::Status::NO_ERROR)) {
        const double initResultTime = std::stod(runOutputData[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
        const double noVectResultTime = std::stod(runOutputNoVect.first[i].second.back().getData(std::string(run_eval::OPTIMIZED_RUNTIME_HEADER)));
        if (noVectResultTime < initResultTime) {
          runOutputData[i] = runOutputNoVect.first[i];
        }
      }
    }
    const auto speedupWVectorization = run_eval::getAvgMedSpeedup(runOutputNoVect.first, runOutputData, speedupHeader);
    multRunSpeedupVect.push_back(speedupWVectorization);
    return {runOutputNoVect, multRunSpeedupVect};
  }
}

template <run_environment::AccSetting OPT_IMP_ACCEL>
void runBpOnStereoSets() {
  //perform runs with and without vectorization using floating point
  //initially store output for floating-point runs separate from output using doubles and halfs
  auto runOutput = runBpOnStereoSets<float, OPT_IMP_ACCEL>();
  //get results and speedup for using potentially alternate and no vectorization (only applies to CPU)
  const auto altAndNoVectSpeedupFl = getAltAndNoVectSpeedup<float, OPT_IMP_ACCEL>(runOutput.first);
  //get run data portion of results 
  auto runOutputAltAndNoVect = altAndNoVectSpeedupFl.first;

  //perform runs with and without vectorization using double-precision
  auto runOutputDouble = runBpOnStereoSets<double, OPT_IMP_ACCEL>();
  const auto doublesSpeedup = run_eval::getAvgMedSpeedup(runOutput.first, runOutputDouble.first, std::string(run_eval::SPEEDUP_DOUBLE));
  const auto altAndNoVectSpeedupDbl = getAltAndNoVectSpeedup<double, OPT_IMP_ACCEL>(runOutputDouble.first);
  for (const auto& runData : altAndNoVectSpeedupDbl.first.first) {
    runOutputAltAndNoVect.first.push_back(runData);
  }
  //perform runs with and without vectorization using half-precision
  auto runOutputHalf = runBpOnStereoSets<halftype, OPT_IMP_ACCEL>();
  const auto halfSpeedup = run_eval::getAvgMedSpeedup(runOutput.first, runOutputHalf.first, std::string(run_eval::SPEEDUP_HALF));
  const auto altAndNoVectSpeedupHalf = getAltAndNoVectSpeedup<halftype, OPT_IMP_ACCEL>(runOutputHalf.first);
  for (const auto& runData : altAndNoVectSpeedupHalf.first.first) {
    runOutputAltAndNoVect.first.push_back(runData);
  }
  //add output for double and half precision runs to output of floating-point runs to write
  //final output with all data
  runOutput.first.insert(runOutput.first.end(), runOutputDouble.first.begin(), runOutputDouble.first.end());
  runOutput.first.insert(runOutput.first.end(), runOutputHalf.first.begin(), runOutputHalf.first.end());
  //get speedup using vectorization across all runs
  const auto vectorizationSpeedupAll = run_eval::getAvgMedSpeedup(runOutputAltAndNoVect.first, runOutput.first, std::string(run_eval::SPEEDUP_VECTORIZATION) + " - All Runs");
  //add speedup data from double and half precision runs to overall data so they are included in final results
  runOutput.second.insert(runOutput.second.end(), altAndNoVectSpeedupFl.second.begin(), altAndNoVectSpeedupFl.second.end());
  runOutput.second.insert(runOutput.second.end(), runOutputDouble.second.begin(), runOutputDouble.second.end());
  runOutput.second.insert(runOutput.second.end(), altAndNoVectSpeedupDbl.second.begin(), altAndNoVectSpeedupDbl.second.end());
  runOutput.second.insert(runOutput.second.end(), runOutputHalf.second.begin(), runOutputHalf.second.end());
  runOutput.second.insert(runOutput.second.end(), altAndNoVectSpeedupHalf.second.begin(), altAndNoVectSpeedupHalf.second.end());

  //get speedup info for using optimized parallel parameters and disparity count as template parameter across all data types
  const auto speedupOverBaseline = run_eval::getAvgMedSpeedupOverBaseline(runOutput.first, "All Runs");
  runOutput.second.insert(runOutput.second.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
  if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
    runOutput.second.push_back(run_eval::getAvgMedSpeedupOptPParams(runOutput.first, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - All Runs"));
  }
  //add more speedup data to overall data for inclusion in final results
  runOutput.second.push_back(run_eval::getAvgMedSpeedupDispValsInTemplate(runOutput.first, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - All Runs"));
  runOutput.second.push_back(vectorizationSpeedupAll);
  runOutput.second.push_back(doublesSpeedup);
  runOutput.second.push_back(halfSpeedup);

  //write output corresponding to results for all data types
  constexpr bool MULT_DATA_TYPES{true};
  run_eval::writeRunOutput<OPT_IMP_ACCEL, MULT_DATA_TYPES>(runOutput);
}

}

#endif /* RUNANDEVALUATEBPRESULTS_H_ */
