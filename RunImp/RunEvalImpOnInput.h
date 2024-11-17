/*
 * RunEvalImpOnInput.h
 *
 *  Created on: Feb 6, 2024
 *      Author: scott
 */

#ifndef RUN_EVAL_BENCHMARK_IMP_SINGLE_SET_H_
#define RUN_EVAL_BENCHMARK_IMP_SINGLE_SET_H_

#include <utility>
#include <memory>
#include <optional>
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "ParallelParams.h"

//virtual class to run and evaluate benchmark on a input specified by index number
template<RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL, unsigned int NUM_INPUT>
class RunEvalImpOnInput {
public:
  virtual MultRunData operator()(const run_environment::RunImpSettings& runImpSettings) = 0;

protected:
  //set up parallel parameters for benchmark
  virtual std::shared_ptr<ParallelParams> setUpParallelParams(const run_environment::RunImpSettings& runImpSettings) const = 0;

  //retrieve input and parameters for run of current benchmark
  virtual RunData inputAndParamsForCurrBenchmark(bool loopItersTemplated) const = 0;

  //run one or two implementations of benchmark and compare results if running multiple implementations
  virtual std::optional<RunData> runImpsAndCompare(std::shared_ptr<ParallelParams> parallelParams,
    bool runOptImpOnly, bool runImpTmpLoopIters) const = 0;

  //get current run inputs and parameters in RunData structure
  RunData inputAndParamsRunData(bool loopItersTemplated) const {
    RunData currRunData;
    currRunData.addDataWHeader(std::string(run_eval::kDatatypeHeader), std::string(run_environment::kDataSizeToNameMap.at(sizeof(T))));
    currRunData.appendData(run_environment::runSettings<OPT_IMP_ACCEL>());
    currRunData.addDataWHeader(std::string(run_eval::kLoopItersTemplatedHeader), loopItersTemplated);
    return currRunData;
  }

  //run optimized and single threaded implementations using multiple sets of parallel parameters in optimized implementation if set
  //to optimize parallel parameters returns data from runs using default and optimized parallel parameters
  MultRunData::value_type runEvalBenchmark(const run_environment::RunImpSettings& runImpSettings,
    bool runWLoopItersTemplated)
  {
    MultRunData::value_type::value_type outRunData(runImpSettings.optParallelParamsOptionSetting_.first ? 2 : 1);
    enum class RunType { ONLY_RUN, DEFAULT_PARAMS, OPTIMIZED_RUN, TEST_PARAMS };
    std::array<std::array<std::map<std::string, std::string>, 2>, 2> inParamsResultsDefOptRuns;

    //set up parallel parameters for specific benchmark
    std::shared_ptr<ParallelParams> parallelParams = setUpParallelParams(runImpSettings);

    //if optimizing parallel parameters, parallelParamsVect contains parallel parameter settings to run
    //(and contains only the default parallel parameters if not)
    std::vector<std::array<unsigned int, 2>> parallelParamsVect{
      runImpSettings.optParallelParamsOptionSetting_.first ? runImpSettings.pParamsDefaultOptOptions_.second : std::vector<std::array<unsigned int, 2>>()};
      
    //if optimizing parallel parameters, run BP for each parallel parameters option, retrieve best parameters for each kernel or overall for the run,
    //and then run BP with best found parallel parameters
    //if not optimizing parallel parameters, run BP once using default parallel parameters
    for (unsigned int runNum=0; runNum < (parallelParamsVect.size() + 1); runNum++) {
      //initialize current run type to specify if current run is only run, run with default params, test params run, or final run with optimized params
      RunType currRunType{RunType::TEST_PARAMS};
      if (!runImpSettings.optParallelParamsOptionSetting_.first) {
        currRunType = RunType::ONLY_RUN;
      }
      else if (runNum == parallelParamsVect.size()) {
        currRunType = RunType::OPTIMIZED_RUN;
      }

      //get and set parallel parameters for current run if not final run that uses optimized parameters
      std::array<unsigned int, 2> pParamsCurrRun{runImpSettings.pParamsDefaultOptOptions_.first};
      if (currRunType == RunType::ONLY_RUN) {
        parallelParams->setParallelDims(runImpSettings.pParamsDefaultOptOptions_.first);
      }
      else if (currRunType == RunType::TEST_PARAMS) {
        //set parallel parameters to parameters corresponding to current run for each BP processing level
        pParamsCurrRun = parallelParamsVect[runNum];
        parallelParams->setParallelDims(pParamsCurrRun);
        if (pParamsCurrRun == runImpSettings.pParamsDefaultOptOptions_.first) {
          //set run type to default parameters if current run uses default parameters
          currRunType = RunType::DEFAULT_PARAMS;
        }
      }

      //store input params data if using default parallel parameters or final run with optimized parameters
      RunData currRunData;
      if (currRunType != RunType::TEST_PARAMS) {
        //add input and parameters data for specific benchmark to current run data
        currRunData.addDataWHeader(std::string(run_eval::kInputIdxHeader), std::to_string(NUM_INPUT));
        currRunData.appendData(inputAndParamsForCurrBenchmark(runWLoopItersTemplated));
        if ((runImpSettings.optParallelParamsOptionSetting_.first) &&
            (runImpSettings.optParallelParamsOptionSetting_.second ==
             run_environment::OptParallelParamsSetting::kAllowDiffKernelParallelParamsInRun))
        {
          //add parallel parameters for each kernel to current input data if allowing different parallel parameters for each kernel in the same run
          currRunData.appendData(parallelParams->runData());
        }
      }

      //run only optimized implementation and not single-threaded run if current run is not final run or is using default parameter parameters
      const bool runOptImpOnly{currRunType == RunType::TEST_PARAMS};

      //run benchmark implementation(s) and return null output if error in run
      //detailed results stored to file that is generated using stream
      const auto runImpsECodeData = runImpsAndCompare(parallelParams, runOptImpOnly, runWLoopItersTemplated);
      currRunData.addDataWHeader(std::string(run_eval::kRunSuccessHeader), runImpsECodeData.has_value());

      //if error in run and run is any type other than for testing parameters, exit function with null output to indicate error
      if ((!runImpsECodeData) && (currRunType != RunType::TEST_PARAMS)) {
        return {};
      }

      //add data results from current run if run successful
      if (runImpsECodeData) {
        currRunData.appendData(runImpsECodeData.value());
      }

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

      if (runImpSettings.optParallelParamsOptionSetting_.first) {
        //retrieve and store results including runtimes for each kernel if allowing different parallel parameters for each kernel and
        //total runtime for current run
        //if error in run, don't add results for current parallel parameters to results set
        if (runImpsECodeData) {
          if (currRunType != RunType::OPTIMIZED_RUN) {
            parallelParams->addTestResultsForParallelParams(pParamsCurrRun, currRunData);
          }
        }

        //set optimized parallel parameters if next run is final run that uses optimized parallel parameters
        //optimized parallel parameters are determined from previous test runs using multiple test parallel parameters
        if (runNum == (parallelParamsVect.size() - 1)) {
          parallelParams->setOptimizedParams();
        }
      }
    }
      
    return outRunData;
  }
  
};

#endif //RUN_BENCHMARK_IMP_SINGLE_SET_H_