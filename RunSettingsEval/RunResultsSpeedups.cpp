/*
 * RunResultsSpeedups.cpp
 *
 *  Created on: Nov 25, 2024
 *      Author: scott
 */

#include <iostream>
#include <fstream>
#include "RunResultsSpeedups.h"
#include "RunEvalConstsEnums.h"

//constructor that takes in implementation file path and run name and retrieves
//run results and speedup evaluation for the run if available
RunResultsSpeedups::RunResultsSpeedups(
    const std::filesystem::path& imp_results_file_path,
    const std::string& run_name) : run_name_{run_name}
{
  //get run results data from file if available
  std::filesystem::path run_results_fp = imp_results_file_path / run_eval::kImpResultsRunDataFolderName /
    (std::string(run_name_) + '_' + std::string(run_eval::kImpResultsRunDataFolderName) + std::string(run_eval::kCsvFileExtension));
  if (std::filesystem::exists(run_results_fp) && (std::filesystem::is_regular_file(run_results_fp))) {
    run_results_header_to_data_ = HeaderToDataInCsvFile(run_results_fp);
  }

  //get speedup evaluation data from file if available
  std::filesystem::path run_speedup_fp = imp_results_file_path / run_eval::kImpResultsSpeedupsFolderName /
    (std::string(run_name_) + '_' + std::string(run_eval::kSpeedupsDescFileName) + std::string(run_eval::kCsvFileExtension));
  if (std::filesystem::exists(run_speedup_fp) && (std::filesystem::is_regular_file(run_speedup_fp))) {
    speedup_header_to_result_ = HeaderToDataInCsvFile(run_speedup_fp);
  }

  //get input "signature" for run mapped to optimized implementation runtime for each run on input
  //as well as input "signature" for each input mapped to input info to show in evaluation output
  const unsigned int tot_num_runs = run_results_header_to_data_.at(std::string(run_eval::kOptimizedRuntimeHeader)).size();
  for (unsigned int num_run = 0; num_run < tot_num_runs; num_run++) {
    //get unique input signature for evaluation run (evaluation data number, data type, setting of whether to not to
    //have loops with templated iteration counts)
    const EvalInputSignature run_input({
      run_results_header_to_data_.at(std::string(run_eval::kRunInputSigHeaders[0]))[num_run],
      run_results_header_to_data_.at(std::string(run_eval::kRunInputSigHeaders[1]))[num_run],
      run_results_header_to_data_.at(std::string(run_eval::kRunInputSigHeaders[2]))[num_run]});
    //add mapping of total runtime to corresponding run name and input signature
    input_sig_to_runtime_.insert(
      {run_input, run_results_header_to_data_.at(std::string(run_eval::kOptimizedRuntimeHeader))[num_run]});
    //retrieve all data corresponding to run which corresponds to input signature for run
    std::map<std::string, std::string> run_headers_to_data;
    for (const auto& header_data : run_results_header_to_data_) {
      run_headers_to_data[header_data.first] = header_data.second.at(num_run);
    }
    input_sig_to_run_results_.insert({run_input, run_headers_to_data});
  }
}

//constructor that takes in run and speedup data
RunResultsSpeedups::RunResultsSpeedups(
  const std::string& run_name,
  const std::map<std::string, std::vector<std::string>>& run_results_header_to_data,
  const std::map<std::string, std::vector<std::string>>& speedup_header_to_result) :
  run_name_{run_name},
  run_results_header_to_data_{run_results_header_to_data},
  speedup_header_to_result_{speedup_header_to_result} {}

//get mapping of headers to data in csv file for run results and speedups
//assumed that there are no commas in data since it is used as delimiter between data
//first output is headers in order, second output is mapping of headers to results
std::map<std::string, std::vector<std::string>>
RunResultsSpeedups::HeaderToDataInCsvFile(const std::filesystem::path& csv_file_path) const {
  std::ifstream csv_file_str(csv_file_path);
  if (!(csv_file_str.is_open())) {
    std::cout << "ERROR CREATING STREAM: " << csv_file_path << std::endl;
  }
  //retrieve data headers from top row
  std::string headers_line;
  std::getline(csv_file_str, headers_line);
  std::stringstream headers_str(headers_line);
  std::vector<std::string> data_headers;
  std::string header;
  std::map<std::string, std::vector<std::string>> header_to_data;
  while (std::getline(headers_str, header, ',')) {
    data_headers.push_back(header);
    header_to_data[header] = std::vector<std::string>();
  }
  //go through each data line and add to mapping from headers to data
  std::string data_line;
  while (std::getline(csv_file_str, data_line)) {
    std::stringstream data_line_str(data_line);
    std::string data;
    unsigned int num_data{0};
    while (std::getline(data_line_str, data, ',')) {
      header_to_data[data_headers[num_data++]].push_back(data);
    }
  }
  return header_to_data;
}