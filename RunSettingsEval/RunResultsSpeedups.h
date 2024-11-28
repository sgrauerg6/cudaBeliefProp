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
#include "EvalInputSignature.h"

//class to save, load, and store run results data include speedups from evaluation
class RunResultsSpeedups {
public:
  //constructor that takes in implementation file path and run name and retrieves
  //run results and speedup evaluation for the run if available
  RunResultsSpeedups(
    const std::filesystem::path& imp_results_file_path,
    const std::string& run_name);

  RunResultsSpeedups(const std::filesystem::path& run_results_file_path);

  //constructor that takes in run and speedup data
  RunResultsSpeedups(
    const std::string& run_name,
    const std::map<std::string, std::vector<std::string>>& run_results_header_to_data,
    const std::map<std::string, std::vector<std::string>>& speedup_header_to_result);

  //return run results for all runs on an architecture
  std::map<std::string, std::vector<std::string>> RunResults() const {
    return run_results_header_to_data_.value();
  }

  //return speedups from evaluation across all runs on an architecture
  std::map<std::string, std::vector<std::string>> Speedups() const {
    return speedup_header_to_result_.value();
  }

  std::map<EvalInputSignature, std::string> InputsToKeyVal(std::string_view key);

  //return data for specified input signature
  std::map<std::string, std::string> DataForInput(const EvalInputSignature& input_sig) const {
    return input_sig_to_run_results_->at(input_sig);
  }

private:
  std::string run_name_;
  std::optional<std::map<std::string, std::vector<std::string>>> run_results_header_to_data_;
  std::optional<std::map<std::string, std::vector<std::string>>> speedup_header_to_result_;
  std::optional<std::map<EvalInputSignature, std::map<std::string, std::string>>> input_sig_to_run_results_;

  //process run results, specifically generate input sig to data mappings from run results
  void ProcessRunResults();

  //get mapping of headers to data in csv file for run results and speedups
  //assumed that there are no commas in data since it is used as delimiter between data
  //first output is headers in order, second output is mapping of headers to results
  std::map<std::string, std::vector<std::string>> HeaderToDataInCsvFile(
    const std::filesystem::path& csv_file_path) const;
};

#endif //RUN_RESULTS_SPEEDUPS_H_
