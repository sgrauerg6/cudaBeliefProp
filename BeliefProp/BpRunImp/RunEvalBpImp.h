/*
 * RunEvalBpImp.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNEVALBPIMP_H_
#define RUNEVALBPIMP_H_

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

using namespace beliefprop;

class RunEvalBpImp {
public:
  //perform runs on multiple data sets using specified data type and acceleration method
  template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
  std::pair<MultRunData, std::vector<MultRunSpeedup>> operator()(const run_environment::RunImpSettings& runImpSettings) const {
    MultRunData runData;
    std::vector<MultRunData> runResultsEachInput;
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 0, OPT_IMP_ACCEL>(runImpSettings));
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 1, OPT_IMP_ACCEL>(runImpSettings));
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 2, OPT_IMP_ACCEL>(runImpSettings));
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 3, OPT_IMP_ACCEL>(runImpSettings));
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 4, OPT_IMP_ACCEL>(runImpSettings));
#ifndef SMALLER_SETS_ONLY
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 5, OPT_IMP_ACCEL>(runImpSettings));
    runResultsEachInput.push_back(runBpOnSetAndUpdateResults<T, 6, OPT_IMP_ACCEL>(runImpSettings));
#endif //SMALLER_SETS_ONLY

    //add run results for each input to overall results
    for (auto& runResult : runResultsEachInput) {
      runData.insert(runData.end(), runResult.begin(), runResult.end());
    }

    //initialize speedup results
    std::vector<MultRunSpeedup> speedupResults;

    //get speedup info for using optimized parallel parameters and disparity count as template parameter
    if constexpr (std::is_same_v<T, float>) {
      //get speedup over baseline runtimes...can only compare with baseline runtimes that are
      //generated using same templated iterations setting as current run
      if ((runImpSettings.baseOptSingThreadRTimeForTSetting_) &&
          (runImpSettings.baseOptSingThreadRTimeForTSetting_.value().second == runImpSettings.templatedItersSetting_)) {
        const std::vector<std::pair<std::string, std::vector<unsigned int>>> subsetsStrIndices = {
          {"smallest 3 stereo sets", {0, 1, 2, 3, 4, 5}},
  #ifndef SMALLER_SETS_ONLY
          {"largest 3 stereo sets", {8, 9, 10, 11, 12, 13}}
  #else
          {"largest stereo set", {8, 9}}
  #endif //SMALLER_SETS_ONLY
        };
        const auto speedupOverBaseline = run_eval::getAvgMedSpeedupOverBaseline(
          runData, run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)),
          runImpSettings.baseOptSingThreadRTimeForTSetting_.value().first, subsetsStrIndices);
        speedupResults.insert(speedupResults.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
      }
    }
    if (runImpSettings.optParallelParmsOptionSetting_.first) {
      speedupResults.push_back(run_eval::getAvgMedSpeedupOptPParams(runData, std::string(run_eval::SPEEDUP_OPT_PAR_PARAMS_HEADER) + " - " +
        run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))));
    }
    if (runImpSettings.templatedItersSetting_ == run_environment::TemplatedItersSetting::RUN_TEMPLATED_AND_NOT_TEMPLATED) {
      speedupResults.push_back(run_eval::getAvgMedSpeedupDispValsInTemplate(runData, std::string(run_eval::SPEEDUP_DISP_COUNT_TEMPLATE) + " - " +
        run_environment::DATA_SIZE_TO_NAME_MAP.at(sizeof(T))));
    }
    
    //write output corresponding to results for current data type
    constexpr bool MULT_DATA_TYPES{false};
    run_eval::writeRunOutput<OPT_IMP_ACCEL, MULT_DATA_TYPES, T>({runData, speedupResults}, runImpSettings);

    //return data for each run and multiple average and median speedup results across the data
    return {runData, speedupResults};
  }

private:
  //run and compare output disparity maps using the given optimized and single-threaded stereo implementations
  //on the reference and test images specified by numStereoSet
  //run only optimized implementation if runOptImpOnly is true
  template<RunData_t T, unsigned int DISP_VALS_OPTIMIZED, unsigned int DISP_VALS_SINGLE_THREAD, run_environment::AccSetting OPT_IMP_ACCEL>
    std::pair<run_eval::Status, RunData> runStereoTwoImpsAndCompare(
    const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_OPTIMIZED, OPT_IMP_ACCEL>>& optimizedImp,
    const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_SINGLE_THREAD, run_environment::AccSetting::NONE>>& singleThreadCPUImp,
    const unsigned int numStereoSet, const beliefprop::BPsettings& algSettings,
    const beliefprop::ParallelParameters& parallelParams,
    bool runOptImpOnly = false) const
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
    const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_SINGLE_THREAD, run_environment::AccSetting::NONE>>& singleThreadCPUImp,
    const run_environment::RunImpSettings& runImpSettings) const
  {
    std::vector<RunData> outRunData(runImpSettings.optParallelParmsOptionSetting_.first ? 2 : 1);
    enum class RunType { ONLY_RUN, DEFAULT_PARAMS, OPTIMIZED_RUN, TEST_PARAMS };
    std::array<std::array<std::map<std::string, std::string>, 2>, 2> inParamsResultsDefOptRuns;
    //load all the BP default settings
    beliefprop::BPsettings algSettings;
    algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

    //parallel parameters initialized with default thread count dimensions at every level
    beliefprop::ParallelParameters parallelParams(algSettings.numLevels_, runImpSettings.pParamsDefaultOptOptions_.first);

    //if optimizing parallel parameters, parallelParamsVect contains parallel parameter settings to run
    //(and contains only the default parallel parameters if not)
    std::vector<std::array<unsigned int, 2>> parallelParamsVect{
      runImpSettings.optParallelParmsOptionSetting_.first ? runImpSettings.pParamsDefaultOptOptions_.second : std::vector<std::array<unsigned int, 2>>()};
    
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
      //initialize current run type to specify if current run is only run, run with default params, test params run, or final run with optimized params
      RunType currRunType{RunType::TEST_PARAMS};
      if (!runImpSettings.optParallelParmsOptionSetting_.first) {
        currRunType = RunType::ONLY_RUN;
      }
      else if (runNum == parallelParamsVect.size()) {
        currRunType = RunType::OPTIMIZED_RUN;
      }

      //get and set parallel parameters for current run if not final run that uses optimized parameters
      std::array<unsigned int, 2> pParamsCurrRun{runImpSettings.pParamsDefaultOptOptions_.first};
      if (currRunType == RunType::ONLY_RUN) {
        parallelParams.setParallelDims(runImpSettings.pParamsDefaultOptOptions_.first, algSettings.numLevels_);
      }
      else if (currRunType == RunType::TEST_PARAMS) {
        //set parallel parameters to parameters corresponding to current run for each BP processing level
        pParamsCurrRun = parallelParamsVect[runNum];
        parallelParams.setParallelDims(pParamsCurrRun, algSettings.numLevels_);
        if (pParamsCurrRun == runImpSettings.pParamsDefaultOptOptions_.first) {
          //set run type to default parameters if current run uses default parameters
          currRunType = RunType::DEFAULT_PARAMS;
        }
      }

      //store input params data if using default parallel parameters or final run with optimized parameters
      RunData currRunData;
      if (currRunType != RunType::TEST_PARAMS) {
        currRunData.addDataWHeader("Stereo Set", bp_params::STEREO_SET[NUM_SET]);
        currRunData.appendData(run_eval::inputAndParamsRunData<T, beliefprop::BPsettings, DISP_VALS_TEMPLATE_OPTIMIZED, OPT_IMP_ACCEL>(algSettings));
        currRunData.appendData(bp_params::runSettings());
        if (runImpSettings.optParallelParmsOptionSetting_.first &&
          (runImpSettings.optParallelParmsOptionSetting_.second == run_environment::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN))
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

      if (runImpSettings.optParallelParmsOptionSetting_.first) {
        //retrieve and store results including runtimes for each kernel if allowing different parallel parameters for each kernel and
        //total runtime for current run
        //if error in run, don't add results for current parallel parameters to results set
        const std::string NUM_RUNS_IN_PARENS{"(" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"};
        if (runImpsECodeData.first == run_eval::Status::NO_ERROR) {
          if (currRunType != RunType::OPTIMIZED_RUN) {
            if (runImpSettings.optParallelParmsOptionSetting_.second == run_environment::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
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
          if (runImpSettings.optParallelParmsOptionSetting_.second == run_environment::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
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

  template<RunData_t T, unsigned int NUM_SET, run_environment::AccSetting OPT_IMP_ACCEL>
  MultRunData runBpOnSetAndUpdateResults(const run_environment::RunImpSettings& runImpSettings) const {
    std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], run_environment::AccSetting::NONE>> runBpStereoSingleThread =
      std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
    MultRunData runResults;
    //RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
    if (runImpSettings.templatedItersSetting_ != run_environment::TemplatedItersSetting::RUN_ONLY_NON_TEMPLATED) {
      std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], OPT_IMP_ACCEL>> runBpOptimizedImp = 
        std::make_unique<RunBpOptimized<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], OPT_IMP_ACCEL>>();
      runResults.push_back(runBpOnSetAndUpdateResults<T, NUM_SET, OPT_IMP_ACCEL>(runBpOptimizedImp, runBpStereoSingleThread, runImpSettings));
    }
    if (runImpSettings.templatedItersSetting_ != run_environment::TemplatedItersSetting::RUN_ONLY_TEMPLATED) {
      std::unique_ptr<RunBpStereoSet<T, 0, OPT_IMP_ACCEL>> runBpOptimizedImp = 
        std::make_unique<RunBpOptimized<T, 0, OPT_IMP_ACCEL>>();
      runResults.push_back(runBpOnSetAndUpdateResults<T, NUM_SET, OPT_IMP_ACCEL>(runBpOptimizedImp, runBpStereoSingleThread, runImpSettings));
    }
    return runResults;
  }
};

#endif /* RUNEVALBPIMP_H_ */
