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
 * @file RunResultsSpeedups.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include <iostream>
#include <fstream>
#include "RunEvalConstsEnums.h"
#include "RunResultsSpeedups.h"

//constructor that takes in implementation file path and run name and retrieves
//run results and speedup evaluation for the run if available
RunResultsSpeedups::RunResultsSpeedups(
  const std::filesystem::path& imp_results_file_path,
  const std::string& run_name) : run_name_{run_name}
{  
  std::pair<std::map<std::string, std::vector<std::string>>,
            std::vector<std::string>>
  run_results_header_to_data_ordered_headers;
  //get run results data from file if available
  if (const auto run_results_fp = 
        imp_results_file_path / run_eval::kImpResultsRunDataFolderName /
        (std::string(run_name_) + '_' +
         std::string(run_eval::kImpResultsRunDataFolderName) +
         std::string(run_eval::kCsvFileExtension));
      ((std::filesystem::exists(run_results_fp))) &&
       (std::filesystem::is_regular_file(run_results_fp)))
  {
    run_results_header_to_data_ordered_headers =
      HeaderToDataWOrderedHeadersCsv(run_results_fp);
  }

  //get speedup evaluation data from file if available
  if (const auto run_speedup_fp =
        imp_results_file_path / run_eval::kImpResultsSpeedupsFolderName /
        (std::string(run_name_) + '_' + 
         std::string(run_eval::kSpeedupsDescFileName) +
         std::string(run_eval::kCsvFileExtension));
      ((std::filesystem::exists(run_speedup_fp)) &&
       (std::filesystem::is_regular_file(run_speedup_fp))))
  {
    speedup_header_to_result_speedup_order_ =
      HeaderToDataWOrderedHeadersCsv(run_speedup_fp);
  }

  //generate input signature to data mappings
  GenInputSignatureToDataMapping(
    run_results_header_to_data_ordered_headers.first);
}

//constructor that takes in run results path and processes run results
//speedups not available when using this constructor
RunResultsSpeedups::RunResultsSpeedups(
  const std::filesystem::path& run_results_file_path)
{
  //get run results data from file if available
  std::pair<std::map<std::string, std::vector<std::string>>,
            std::vector<std::string>>
    run_results_header_to_data_ordered_headers;
  if ((std::filesystem::exists(run_results_file_path)) &&
      (std::filesystem::is_regular_file(run_results_file_path)))
  {
    run_results_header_to_data_ordered_headers =
      HeaderToDataWOrderedHeadersCsv(run_results_file_path);
  }

  //generate input signature to data mappings
  GenInputSignatureToDataMapping(
    run_results_header_to_data_ordered_headers.first);
}

//generate input sig to run data mappings from run results as read from file
void RunResultsSpeedups::GenInputSignatureToDataMapping(
  const std::optional<std::map<std::string,
  std::vector<std::string>>>& run_results_header_to_data)
{
  if (run_results_header_to_data)
  {
    //initialize input signature to run data mapping
    input_sig_to_run_data_ = decltype(input_sig_to_run_data_)::value_type();
    
    //get input "signature" mapped to run data for each run in run results
    const unsigned int tot_num_runs = 
      run_results_header_to_data->at(
        std::string(run_eval::kOptimizedRuntimeHeader)).size();
    for (unsigned int num_run = 0; num_run < tot_num_runs; num_run++)
    {
      //get unique input signature for evaluation run (evaluation data number,
      //data type, setting of whether to not to have loops with templated
      //iteration counts)
      const InputSignature run_input({
        run_results_header_to_data->at(
          std::string(run_eval::kRunInputSigHeaders[0])).at(num_run),
        run_results_header_to_data->at(
          std::string(run_eval::kRunInputSigHeaders[1])).at(num_run),
        run_results_header_to_data->at(
          std::string(run_eval::kRunInputSigHeaders[2])).at(num_run)});

      //retrieve all data for run corresponding to current input signature
      //and generate mapping between input signature and run data
      std::map<std::string, std::string> run_headers_to_data;
      for (const auto& header_data : *run_results_header_to_data) {
        run_headers_to_data.insert(
          {header_data.first, header_data.second.at(num_run)});
      }
      input_sig_to_run_data_->insert({run_input, run_headers_to_data});
    }
  }
}

//get mapping of run input signature to value corresponding to input key
//for each run result
std::map<InputSignature, std::string> RunResultsSpeedups::InputsToKeyVal(
  std::string_view key) const
{
  std::map<InputSignature, std::string> input_sig_to_key_val;
  if (input_sig_to_run_data_) {
    //get input "signature" for run mapped to corresponding key value for each
    //run on input
    for (const auto& [input_sig, run_data] : *input_sig_to_run_data_)
    {
      //add mapping of key value to corresponding input signature
      input_sig_to_key_val.insert(
        {input_sig, run_data.at(std::string(key))});
    }
  }

  return input_sig_to_key_val;
}

//get mapping of headers to data in csv file for run results and speedups
//assumed that there are no commas in data since it is used as delimiter
//between data
//first output is mapping of headers to results, second output is headers
//in order
std::pair<std::map<std::string, std::vector<std::string>>,
          std::vector<std::string>>
RunResultsSpeedups::HeaderToDataWOrderedHeadersCsv(
  const std::filesystem::path& csv_file_path) const
{
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
  return {header_to_data, data_headers};
}