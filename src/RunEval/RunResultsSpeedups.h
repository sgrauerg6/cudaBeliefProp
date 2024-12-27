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
 * @file RunResultsSpeedups.h
 * @author Scott Grauer-Gray
 * @brief Declares class to load and store run results data including speedups
 * from evaluation
 * 
 * @copyright Copyright (c) 2024
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
 * @brief Class to load and store run results data including speedups from
 * evaluation
 */
class RunResultsSpeedups {
public:
  /**
   * @brief Constructor that takes in implementation file path and run name and
   * retrieves run results and speedup evaluation for the run if available
   * 
   * @param imp_results_file_path 
   * @param run_name 
   */
  explicit RunResultsSpeedups(
    const std::filesystem::path& imp_results_file_path,
    const std::string& run_name);

  /**
   * @brief Constructor that takes in run results path and processes run
   * results<br>
   * Speedups not available when using this constructor
   * 
   * @param run_results_file_path 
   */
  explicit RunResultsSpeedups(const std::filesystem::path& run_results_file_path);

  /**
   * @brief Return speedups from evaluation across all runs on an architecture
   * 
   * @return std::map<std::string, std::vector<std::string>> 
   */
  std::map<std::string, std::vector<std::string>> Speedups() const {
    return speedup_header_to_result_speedup_order_->first;
  }

  /**
   * @brief Return order of speedup headers
   * 
   * @return std::vector<std::string>
   */
  std::vector<std::string> SpeedupHeadersOrder() const {
    return speedup_header_to_result_speedup_order_->second;
  }

  /**
   * @brief Get mapping of run input signature to value corresponding to input
   * key for each run result
   * 
   * @param key 
   * @return std::map<InputSignature, std::string> 
   */
  std::map<InputSignature, std::string> InputsToKeyVal(std::string_view key) const;

  /**
   * @brief Return data for specified input signature
   * 
   * @param input_sig 
   * @return std::map<std::string, std::string> 
   */
  std::map<std::string, std::string> DataForInput(
    const InputSignature& input_sig) const
  {
    return input_sig_to_run_data_->at(input_sig);
  }

private:
  /** @brief Name for run results where it's recommended that run name contain
   *  information about processor architecture used in run */
  std::string run_name_;
  
  /** @brief Pair with first element being mapping of speedups headers to
   *  speedup results and second element being speedup headers in order */ 
  std::optional<std::pair<std::map<std::string, std::vector<std::string>>,
                          std::vector<std::string>>>
    speedup_header_to_result_speedup_order_;

  /** @brief Mapping of input signature to run data for each run */
  std::optional<std::map<InputSignature, std::map<std::string, std::string>>>
    input_sig_to_run_data_;

  /**
   * @brief Generate input signature to run data mappings from run results as
   * read from file
   * 
   * @param run_results_header_to_data 
   */
  void GenInputSignatureToDataMapping(
    const std::optional<std::map<std::string, std::vector<std::string>>>&
      run_results_header_to_data);

  /**
   * @brief Get mapping of headers to data as well as headers in order in csv
   * file for run results and speedups.
   * Assumed that there are no commas in data since it is used as delimiter
   * between data<br>
   * First output is mapping of headers to results, second output is headers
   * in order
   * 
   * @param csv_file_path 
   * @return std::pair<std::map<std::string, std::vector<std::string>>,
   *                   std::vector<std::string>>
   */
    std::pair<std::map<std::string, std::vector<std::string>>,
              std::vector<std::string>>
    HeaderToDataWOrderedHeadersCsv(
      const std::filesystem::path& csv_file_path) const;
};

#endif //RUN_RESULTS_SPEEDUPS_H_
