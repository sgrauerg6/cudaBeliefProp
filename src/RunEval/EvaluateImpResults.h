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
 * @file EvaluateImpResults.h
 * @author Scott Grauer-Gray
 * @brief Declares class with operator function to evaluate implementations of
 * the same algorithm across different data types and acceleration methods
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef EVALUATE_IMP_RESULTS_H_
#define EVALUATE_IMP_RESULTS_H_

#include <filesystem>
#include <unordered_map>
#include <vector>
#include <array>
#include <utility>
#include <string_view>
#include <optional>
#include "RunSettingsParams/RunSettings.h"
#include "RunSettingsParams/InputSignature.h"
#include "EvaluateImpAliases.h"

#if defined(OPTIMIZED_CUDA_RUN)
#include "RunImpCUDA/RunCUDASettings.h"
#endif //OPTIMIZED_CUDA_RUN

/**
 * @brief Enum to define difference between "base" and "target" result sets
 * when evaluating speedup
 */
enum class BaseTargetDiff {
  kDiffAcceleration, kDiffDatatype, kDiffTemplatedSetting};

/**
 * @brief Class with operator function to evaluate implementations of the same
 * algorithm across different data types and acceleration methods
 */
class EvaluateImpResults {
public:
  /**
   * @brief Evaluate results for implementation runs on multiple inputs with
   * all the runs having the same data type and acceleration method
   * return run data results with run speedups added as well as average and
   * median speedups with headers describing speedups
   * 
   * @param run_results 
   * @param run_imp_settings 
   * @param data_size 
   * @return Run results with additional data and computed speedups from run
   */
  std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>
  EvalResultsSingDataTypeAcc(
    const MultRunData& run_results,
    const run_environment::RunImpSettings& run_imp_settings,
    size_t data_size) const;

  /**
   * @brief Evaluate results for all implementation runs on multiple inputs
   * with the runs potentially having different data types and acceleration
   * methods and write run result and speedup outputs to files
   * 
   * @param run_results_mult_runs 
   * @param run_imp_settings 
   * @param opt_imp_acc 
   */
  void EvalAllResultsWriteOutput(
    const std::unordered_map<size_t, MultRunDataWSpeedupByAcc>&
      run_results_mult_runs,
    const run_environment::RunImpSettings& run_imp_settings,
    run_environment::AccSetting opt_imp_acc) const;

private:
  /**
   * @brief Retrieve file path of implementation run results
   * must be defined in child class
   * 
   * @return File path of implementation run results
   */
  virtual std::filesystem::path GetImpResultsPath() const = 0;
  
  /**
   * @brief Get text at top of results across runs with each string in the
   * vector corresponding to a separate line<br>
   * Must be defined in child class
   * 
   * @return Vector of strings corresponding to text to display at top of file
   * showing results across runs
   */
  virtual std::vector<std::string> GetCombResultsTopText() const = 0;

  /**
   * @brief Get input parameters that are shown in results across runs with
   * runtimes<br>
   * Must be defined in child class
   * 
   * @return Vector of strings corresponding to input parameters to show in
   * results across runs
   */
  virtual std::vector<std::string> GetInputParamsShow() const = 0;

  /**
   * @brief Get speedups across runs of all data types
   * 
   * @param run_results
   * @param run_imp_settings
   * @return Speedup results across runs of all data type
   */
  std::vector<RunSpeedupAvgMedian> SpeedupsAllRuns(
    MultRunData& run_results,
    const run_environment::RunImpSettings& run_imp_settings) const;

  /**
   * @brief Write data for file corresponding to runs for a specified data type
   * or across all data type.<br>
   * Includes results for each run as well as average and median speedup data
   * across multiple runs.
   * 
   * @param run_results_w_speedups 
   * @param run_imp_settings 
   * @param acceleration_setting 
   */
  void WriteRunOutput(
    const std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>& run_results_w_speedups,
    const run_environment::RunImpSettings& run_imp_settings,
    run_environment::AccSetting acceleration_setting) const;

  /**
   * @brief Write run results corresponding to runs using a specific
   * acceleration to a file with the acceleration setting as part of the file
   * name
   * 
   * @param run_results
   * @param acceleration_setting
   * @param run_imp_settings
   */
  void WriteRunResultsForAcc(
    const MultRunData& run_results,
    run_environment::AccSetting acceleration_setting,
    const run_environment::RunImpSettings& run_imp_settings) const;

  /**
   * 
   * @brief Get speedup for each run and overall when using optimized
   * vectorization vs "lesser" vectorization and no vectorization
   * if there are runs with "lesser" vectorization and/or no
   * vectorization
   * 
   * @param run_imp_results_by_acc_setting 
   * @param run_imp_settings 
   * @param data_type_size 
   * @param fastest_acc 
   * @return Speedups (or slowdowns) with optimized run results from using
   * alternate acceleration as opposed to assumed fastest acceleration
   */
  std::pair<std::vector<RunSpeedupAvgMedian>, MultRunData> GetAltAccelSpeedups(
    MultRunDataWSpeedupByAcc& run_imp_results_by_acc_setting,
    const run_environment::RunImpSettings& run_imp_settings,
    size_t data_type_size,
    run_environment::AccSetting fastest_acc) const;

  /**
   * @brief Get speedup over baseline data if data available
   * 
   * @param run_imp_settings 
   * @param run_data_all_runs 
   * @param data_type_size 
   * @return Speedup over baseline data
   */
  std::vector<RunSpeedupAvgMedian> GetSpeedupOverBaseline(
    const run_environment::RunImpSettings& run_imp_settings,
    MultRunData& run_data_all_runs,
    size_t data_type_size) const;

  /**
   * @brief Get speedup over baseline run for subsets of baseline data if data
   * available
   * 
   * @param run_imp_settings 
   * @param run_data_all_runs 
   * @param data_type_size 
   * @return Speedups over subsets of baseline data
   */
  std::vector<RunSpeedupAvgMedian> GetSpeedupOverBaselineSubsets(
    const run_environment::RunImpSettings& run_imp_settings,
    MultRunData& run_data_all_runs, 
    size_t data_type_size) const;

  /**
   * @brief Get baseline runtime data (assumed to be available)
   * 
   * @param baseline_runtimes_path_desc 
   * @param key_runtime_data 
   * @return Baseline runtime data
   */
  std::pair<std::string, std::map<InputSignature, std::string>>
  GetBaselineRuntimeData(
    const std::array<std::string_view, 2>& baseline_runtimes_path_desc,
    std::string_view key_runtime_data) const;

  /**
   * @brief Get average and median speedup from vector of speedup values
   * 
   * @param speedups_vect 
   * @return Average and median speedups
   */
  RunSpeedupAvgMedian::second_type GetAvgMedSpeedup(
    const std::vector<double>& speedups_vect) const;

  /**
   * @brief Get average and median speedup of specified subset(s) of runs
   * compared to baseline data from file
   * 
   * @param run_results 
   * @param data_type_str 
   * @param baseline_runtimes_path_desc 
   * @param subset_desc_input_sig 
   * @return Average and median speedups over basline data for specified
   * subsets of runs
   */
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOverBaselineSubsets(
    MultRunData& run_results,
    std::string_view data_type_str,
    const std::array<std::string_view, 2>& baseline_runtimes_path_desc,
    const std::vector<std::pair<std::string, std::vector<InputSignature>>>&
      subset_desc_input_sig =
        std::vector<std::pair<std::string, std::vector<InputSignature>>>())
    const;

  /**
   * @brief Get average and median speedup of current runs compared to baseline
   * data from file
   * 
   * @param run_results 
   * @param data_type_str 
   * @param baseline_runtimes_path_desc 
   * @return Average and median speedups over basline data
   */
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOverBaseline(
    MultRunData& run_results,
    std::string_view data_type_str,
    const std::array<std::string_view, 2>& baseline_runtimes_path_desc) const;

  /**
   * @brief Get average and median speedup using optimized parallel parameters
   * compared to default parallel parameters
   * 
   * @param run_results
   * @param speedup_header
   * @param eval_subsets
   * @return Average and median speedups using optimized parallel parameters
   * compared to default parallel parameters across all runs and subsets of
   * runs if specified
   */
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupOptPParams(
    MultRunData& run_results,
    std::string_view speedup_header,
    const std::optional<std::vector<std::pair<std::string, std::vector<InputSignature>>>>& eval_subsets =
      std::nullopt) const;

  /**
   * @brief Get average and median speedup between base and target runtime data
   * across all runs and any specified subsets of runs
   * 
   * @param run_results_base 
   * @param run_results_target 
   * @param speedup_header 
   * @param base_target_diff
   * @param eval_subsets 
   * @return Average and median speedups between base and target runtime data
   * across all runs and subsets of runs if specified
   */
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupBaseVsTarget(
    MultRunData& run_results_base,
    MultRunData& run_results_target,
    std::string_view speedup_header,
    BaseTargetDiff base_target_diff,
    const std::optional<std::vector<std::pair<std::string, std::vector<InputSignature>>>>& eval_subsets =
      std::nullopt) const;

  /**
   * @brief Get average and median speedup when loop iterations are given at
   * compile time as template value
   * 
   * @param run_results 
   * @param speedup_header 
   * @param run_imp_settings
   * @return Average and median speedups when loop iterations are given at
   * compile time as template value compared to when iteration count only known
   * at runtime across all runs and subsets of runs if specified in
   * implementation settings
   */
  std::vector<RunSpeedupAvgMedian> GetAvgMedSpeedupLoopItersInTemplate(
    MultRunData& run_results,
    std::string_view speedup_header,
    const run_environment::RunImpSettings& run_imp_settings) const;
};

#endif //EVALUATE_IMP_RESULTS_H_
