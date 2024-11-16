/*
 * EvaluateImpResults.h
 *
 *  Created on: Feb 20, 2024
 *      Author: scott
 * 
 *  Class to evaluate implementation results.
 */

#ifndef EVALUATE_IMP_RESULTS_H_
#define EVALUATE_IMP_RESULTS_H_

#include "RunEvalConstsEnums.h"
#include "RunSettings.h"
#include <filesystem>

#ifdef OPTIMIZED_CUDA_RUN
#include "RunImpCUDA/RunCUDASettings.h"
#endif //OPTIMIZED_CUDA_RUN

//class with operator function to evaluate implementation runs
class EvaluateImpResults {
public:
  //evaluate results for implementation runs on multiple inputs with all the runs having the same data type and acceleration method
  void operator()(const MultRunData& runResults, const run_environment::RunImpSettings runImpSettings, run_environment::AccSetting optImpAcc, size_t dataSize);

  //evaluate results for implementation runs on multiple inputs with the runs having different data type and acceleration methods
  void operator()(const std::unordered_map<size_t, MultRunDataWSpeedupByAcc>& runResultsMultRuns,
    const run_environment::RunImpSettings runImpSettings, run_environment::AccSetting optImpAcc);

  //get run data with speedup from evaluation of implementation runs using multiple inputs with runs having the same data type and acceleration method
  std::pair<MultRunData, std::vector<MultRunSpeedup>> getRunDataWSpeedups() const;

private:
  //process results for implementation runs on multiple inputs with all the runs having the same data type and acceleration method
  void evalResultsSingDTypeAccRun();

  //process results for implementation runs on multiple inputs with the runs having different data type and acceleration methods
  void evalResultsMultDTypeAccRuns();

  //write data for file corresponding to runs for a specified data type or across all data type
  //includes results for each run as well as average and median speedup data across multiple runs
  template <bool MULT_DATA_TYPES>
  void writeRunOutput(const std::pair<MultRunData, std::vector<MultRunSpeedup>>& runOutput, const run_environment::RunImpSettings& runImpSettings,
    run_environment::AccSetting accelerationSetting, const unsigned int dataTypeSize = 0) const;

  //perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
  //CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
  std::vector<MultRunSpeedup> getAltAccelSpeedups(
    MultRunDataWSpeedupByAcc& runImpResultsByAccSetting,
    const run_environment::RunImpSettings& runImpSettings, size_t dataTypeSize, run_environment::AccSetting fastestAcc) const;

  //get speedup over baseline data if data available
  std::vector<MultRunSpeedup> getSpeedupOverBaseline(const run_environment::RunImpSettings& runImpSettings,
    MultRunData& runDataAllRuns, const size_t dataTypeSize) const;

  //get speedup over baseline run for subsets of smallest and largest sets if data available
  std::vector<MultRunSpeedup> getSpeedupOverBaselineSubsets(const run_environment::RunImpSettings& runImpSettings,
    MultRunData& runDataAllRuns, const size_t dataTypeSize) const;

  //get baseline runtime data if available...return null if baseline data not available
  std::optional<std::pair<std::string, std::vector<double>>> getBaselineRuntimeData(std::string_view baselineDataPath) const;

  //get average and median speedup from vector of speedup values
  std::array<double, 2> getAvgMedSpeedup(const std::vector<double>& speedupsVect) const;

  //get average and median speedup of specified subset(s) of runs compared to baseline data from file
  std::vector<MultRunSpeedup> getAvgMedSpeedupOverBaselineSubsets(MultRunData& runOutput,
    std::string_view dataTypeStr, const std::array<std::string_view, 2>& baseDataPathOptSingThrd,
    const std::vector<std::pair<std::string, std::vector<unsigned int>>>& subsetStrIndices = std::vector<std::pair<std::string, std::vector<unsigned int>>>()) const;

  //get average and median speedup of current runs compared to baseline data from file
  std::vector<MultRunSpeedup> getAvgMedSpeedupOverBaseline(MultRunData& runOutput,
    std::string_view dataTypeStr, const std::array<std::string_view, 2>& baselinePathOptSingThread) const;

  //get average and median speedup using optimized parallel parameters compared to default parallel parameters
  MultRunSpeedup getAvgMedSpeedupOptPParams(MultRunData& runOutput, std::string_view speedupHeader) const;

  //get average and median speedup between base and target runtime data
  MultRunSpeedup getAvgMedSpeedup(MultRunData& runOutputBase, MultRunData& runOutputTarget,
    std::string_view speedupHeader) const;

  //get average and median speedup when loop iterations are given at compile time as template value
  MultRunSpeedup getAvgMedSpeedupLoopItersInTemplate(MultRunData& runOutput,
    std::string_view speedupHeader) const;

  //retrieve path of results
  virtual std::filesystem::path getImpResultsPath() const = 0;
  
  //get text at top of results across runs with each string in the vector corresponding to a separate line
  virtual std::vector<std::string> getCombResultsTopText() const = 0;

  //input parameters that are showed in results across runs with runtimes
  virtual std::vector<std::string> getInputParamsShow() const = 0;

  run_environment::RunImpSettings runImpSettings_;
  run_environment::AccSetting optImpAccel_;
  size_t dataSize_;
  MultRunData runImpOrigResults_;
  MultRunData runImpOptResults_;
  std::vector<MultRunSpeedup> runImpSpeedups_;
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> runImpResultsMultRuns_;
  bool writeDebugOutputFiles_{false};
};

#endif //EVALUATE_IMP_RESULTS_H_
