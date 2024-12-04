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

#include <filesystem>
#include <unordered_map>
#include <vector>
#include <array>
#include <utility>
#include <string_view>
#include "RunSettingsParams/RunSettings.h"
#include "RunSettingsParams/InputSignature.h"
#include "EvaluateImpAliases.h"

#ifdef OPTIMIZED_CUDA_RUN
#include "RunImpCUDA/RunCUDASettings.h"
#endif //OPTIMIZED_CUDA_RUN

//enum to define difference between "base" and "target" result sets
//when evaluating speedup
enum class BaseTargetDiff { kDiffAcceleration, kDiffDatatype, kDiffTemplatedSetting };

//class with operator function to evaluate implementations of the same algorithm across
//different data types and acceleration methods
class EvaluateImpResults {
public:
  //evaluate results for implementation runs on multiple inputs with all the runs having the
  //same data type and acceleration method
  //return run data results with run speedups added as well as average and median
  //speedups with headers describing speedups
  std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> EvalResultsSingDataTypeAcc(
    const MultRunData& run_results,
    const run_environment::RunImpSettings run_imp_settings,
    size_t data_size) const;

  //evaluate results for all implementation runs on multiple inputs with the runs
  //potentially having different data types and acceleration methods and
  //write run result and speedup outputs to files
  void EvalAllResultsWriteOutput(
    const std::unordered_map<size_t, MultRunDataWSpeedupByAcc>& run_results_mult_runs,
    const run_environment::RunImpSettings run_imp_settings,
    run_environment::AccSetting opt_imp_acc) const;

private:
  //retrieve file path of implementation run results
  //must be defined in child class
  virtual std::filesystem::path GetImpResultsPath() const = 0;
  
  //get text at top of results across runs with each string in the vector
  //corresponding to a separate line
  //must be defined in child class
  virtual std::vector<std::string> GetCombResultsTopText() const = 0;

  //input parameters that are shown in results across runs with runtimes
  //must be defined in child class
  virtual std::vector<std::string> GetInputParamsShow() const = 0;

  //write data for file corresponding to runs for a specified data type or across all data type
  //includes results for each run as well as average and median speedup data across multiple runs
  void WriteRunOutput(
    const std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>& run_output,
    const run_environment::RunImpSettings& run_imp_settings,
    run_environment::AccSetting acceleration_setting) const;

  //perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
  //CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
  std::vector<RunSpeedupAvgMedian> GetAltAccelSpeedups(
    MultRunDataWSpeedupByAcc& run_imp_results_by_acc_setting,
    const run_environment::RunImpSettings& run_imp_settings, size_t data_type_size,
    run_environment::AccSetting fastest_acc) const;

  //get speedup over baseline data if data available
  std::vector<RunSpeedupAvgMedian> GetSpeedupOverBaseline(
    const run_environment::RunImpSettings& run_imp_settings,
    MultRunData& run_data_all_runs, size_t data_type_size) const;

  //get speedup over baseline run for subsets of smallest and largest sets if data available
  std::vector<RunSpeedupAvgMedian> GetSpeedupOverBaselineSubsets(
    const run_environment::RunImpSettings& run_imp_settings,
    MultRunData& run_data_all_runs, 
    size_t data_type_size) const;

  //get baseline runtime data if available...return null if baseline data not available
  std::optional<std::pair<std::string, std::map<InputSignature, std::string>>> GetBaselineRuntimeData(
    const std::array<std::string_view, 2>& baseline_runtimes_path_desc,
    std::string_view key_runtime_data) const;

  //get average and median speedup from vector of speedup values
  std::array<double, 2> GetAvgMedSpeedup(const std::vector<double>& speedups_vect) const;

  //get average and median speedup of specified subset(s) of runs compared to baseline data from file
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOverBaselineSubsets(
    MultRunData& run_output,
    std::string_view data_type_str,
    const std::array<std::string_view, 2>& baseline_runtimes_path_desc,
    const std::vector<std::pair<std::string, std::vector<InputSignature>>>& subset_desc_input_sig =
      std::vector<std::pair<std::string, std::vector<InputSignature>>>()) const;

  //get average and median speedup of current runs compared to baseline data from file
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOverBaseline(
    MultRunData& run_output,
    std::string_view data_type_str,
    const std::array<std::string_view, 2>& baseline_runtimes_path_desc) const;

  //get average and median speedup using optimized parallel parameters compared to default parallel parameters
  RunSpeedupAvgMedian GetAvgMedSpeedupOptPParams(
    MultRunData& run_output,
    std::string_view speedup_header) const;

  //get average and median speedup between base and target runtime data
  RunSpeedupAvgMedian GetAvgMedSpeedupBaseVsTarget(
    MultRunData& run_output_base,
    MultRunData& run_output_target,
    std::string_view speedup_header,
    BaseTargetDiff base_target_diff) const;

  //get average and median speedup when loop iterations are given at compile time as template value
  RunSpeedupAvgMedian GetAvgMedSpeedupLoopItersInTemplate(MultRunData& run_output,
    std::string_view speedup_header) const;
};

#endif //EVALUATE_IMP_RESULTS_H_
