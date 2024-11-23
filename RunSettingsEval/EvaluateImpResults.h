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

//class with operator function to evaluate implementations of the same algorithm across
//different data types and acceleration methods
class EvaluateImpResults {
public:
  //evaluate results for implementation runs on multiple inputs with all the runs having the
  //same data type and acceleration method
  void operator()(const MultRunData& run_results, const run_environment::RunImpSettings run_imp_settings,
    run_environment::AccSetting opt_imp_acc, size_t data_size);

  //evaluate results for implementation runs on multiple inputs with the runs having
  //different data type and acceleration methods
  void operator()(const std::unordered_map<size_t, MultRunDataWSpeedupByAcc>& run_results_mult_runs,
    const run_environment::RunImpSettings run_imp_settings, run_environment::AccSetting opt_imp_acc);

  //get run data with speedup from evaluation of implementation runs using multiple inputs with
  //runs having the same data type and acceleration method
  std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> RunDataWSpeedups() const;

private:
  //process results for implementation runs on multiple inputs with all the runs having the
  //same data type and acceleration method
  void EvalResultsSingDTypeAccRun();

  //process results for implementation runs on multiple inputs with the runs having
  //different data type and acceleration methods
  void EvalResultsMultDTypeAccRuns();

  //write data for file corresponding to runs for a specified data type or across all data type
  //includes results for each run as well as average and median speedup data across multiple runs
  template <bool kMultDataTypes>
  void WriteRunOutput(const std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>& run_output,
    const run_environment::RunImpSettings& run_imp_settings,
    run_environment::AccSetting acceleration_setting,
    unsigned int data_type_size = 0) const;

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
    MultRunData& run_data_all_runs, size_t data_type_size) const;

  //get baseline runtime data if available...return null if baseline data not available
  std::optional<std::pair<std::string, std::vector<double>>> GetBaselineRuntimeData(
    std::string_view baseline_data_path) const;

  //get average and median speedup from vector of speedup values
  std::array<double, 2> GetAvgMedSpeedup(const std::vector<double>& speedups_vect) const;

  //get average and median speedup of specified subset(s) of runs compared to baseline data from file
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOverBaselineSubsets(MultRunData& run_output,
    std::string_view data_type_str, const std::array<std::string_view, 2>& base_data_path_opt_single_thread,
    const std::vector<std::pair<std::string, std::vector<unsigned int>>>& subset_str_indices =
      std::vector<std::pair<std::string, std::vector<unsigned int>>>()) const;

  //get average and median speedup of current runs compared to baseline data from file
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOverBaseline(MultRunData& run_output,
    std::string_view data_type_str, const std::array<std::string_view, 2>& baseline_path_opt_single_thread) const;

  //get average and median speedup using optimized parallel parameters compared to default parallel parameters
  RunSpeedupAvgMedian GetAvgMedSpeedupOptPParams(MultRunData& run_output, std::string_view speedup_header) const;

  //get average and median speedup between base and target runtime data
  RunSpeedupAvgMedian GetAvgMedSpeedup(MultRunData& run_output_base, MultRunData& run_output_target,
    std::string_view speedup_header) const;

  //get average and median speedup when loop iterations are given at compile time as template value
  RunSpeedupAvgMedian GetAvgMedSpeedupLoopItersInTemplate(MultRunData& run_output,
    std::string_view speedup_header) const;

  //retrieve path of results
  virtual std::filesystem::path GetImpResultsPath() const = 0;
  
  //get text at top of results across runs with each string in the vector corresponding to a separate line
  virtual std::vector<std::string> GetCombResultsTopText() const = 0;

  //input parameters that are showed in results across runs with runtimes
  virtual std::vector<std::string> GetInputParamsShow() const = 0;

  run_environment::RunImpSettings run_imp_settings_;
  run_environment::AccSetting opt_imp_accel_;
  size_t data_size_;
  MultRunData run_imp_orig_results_;
  MultRunData run_imp_opt_results_;
  std::vector<RunSpeedupAvgMedian> run_imp_speedups_;
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> run_imp_results_mult_runs_;
  bool write_debug_output_files_{false};
};

#endif //EVALUATE_IMP_RESULTS_H_
