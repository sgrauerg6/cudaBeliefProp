/*
 * RunResultsSpeedups.h
 *
 *  Created on: Nov 25, 2024
 *      Author: scott
 */

#ifndef RUN_RESULTS_SPEEDUPS_H_
#define RUN_RESULTS_SPEEDUPS_H_

#include <map>
#include <vector>
#include <string>
#include <filesystem>
#include <optional>
#include "RunSettingsParams/InputSignature.h"

/**
 * @brief Class to save, load, and store run results data include speedups from evaluation
 * 
 */
class RunResultsSpeedups {
public:
  /**
   * @brief Constructor that takes in implementation file path and run name and retrieves
   * run results and speedup evaluation for the run if available
   * 
   * @param imp_results_file_path 
   * @param run_name 
   */
  RunResultsSpeedups(
    const std::filesystem::path& imp_results_file_path,
    const std::string& run_name);

  /**
   * @brief Constructor that takes in run results path and processes run results
   * speedups not available when using this constructor
   * 
   * @param run_results_file_path 
   */
  RunResultsSpeedups(const std::filesystem::path& run_results_file_path);

  /**
   * @brief Return speedups from evaluation across all runs on an architecture
   * 
   * @return std::map<std::string, std::vector<std::string>> 
   */
  std::map<std::string, std::vector<std::string>> Speedups() const {
    return speedup_header_to_result_.value();
  }

  /**
   * @brief Get mapping of run input signature to value corresponding to input key
   * for each run result
   * 
   * @param key 
   * @return std::map<InputSignature, std::string> 
   */
  std::map<InputSignature, std::string> InputsToKeyVal(std::string_view key);

  /**
   * @brief Return data for specified input signature
   * 
   * @param input_sig 
   * @return std::map<std::string, std::string> 
   */
  std::map<std::string, std::string> DataForInput(const InputSignature& input_sig) const {
    return input_sig_to_run_data_->at(input_sig);
  }

private:
  std::string run_name_;
  std::optional<std::map<std::string, std::vector<std::string>>> speedup_header_to_result_;
  std::optional<std::map<InputSignature, std::map<std::string, std::string>>> input_sig_to_run_data_;

  /**
   * @brief Generate input signature to run data mappings from run results as read from file
   * 
   * @param run_results_header_to_data 
   */
  void GenInputSignatureToDataMapping(
    const std::optional<std::map<std::string, std::vector<std::string>>>& run_results_header_to_data);

  /**
   * @brief Get mapping of headers to data in csv file for run results and speedups
   * Assumed that there are no commas in data since it is used as delimiter between data
   * First output is headers in order, second output is mapping of headers to results
   * 
   * @param csv_file_path 
   * @return std::map<std::string, std::vector<std::string>> 
   */
  std::map<std::string, std::vector<std::string>> HeaderToDataInCsvFile(
    const std::filesystem::path& csv_file_path) const;
};

#endif //RUN_RESULTS_SPEEDUPS_H_
