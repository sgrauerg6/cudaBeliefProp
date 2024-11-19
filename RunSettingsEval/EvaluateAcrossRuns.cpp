/*
 * EvaluateAcrossRuns.cpp
 *
 *  Created on: March 2, 2024
 *      Author: scott
 * 
 *  Definition of functions in class for evaluating results across multiple runs, potentially on different architectures.
 */

#include "EvaluateAcrossRuns.h"

void EvaluateAcrossRuns::operator()(const std::filesystem::path& imp_results_file_path,
  const std::vector<std::string>& eval_across_runs_top_text,
  const std::vector<std::string>& eval_across_runs_in_params_show) const
{
  //get header to data of each set of run results
  //iterate through all run results files and get run name to results
  //create directory iterator with all results files
  std::filesystem::directory_iterator results_files_iter = std::filesystem::directory_iterator(imp_results_file_path / run_eval::kImpResultsRunDataFolderName);
  std::vector<std::string> run_names;
  std::map<std::string, std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>>> run_results_name_to_data;
  for (const auto& results_fp : results_files_iter) {
    std::string file_name_no_ext = results_fp.path().stem();
    if (file_name_no_ext.ends_with("_" + std::string(run_eval::kRunResultsDescFileName))) {
      std::string run_name = file_name_no_ext.substr(0, file_name_no_ext.find("_" + std::string(run_eval::kRunResultsDescFileName)));
      run_names.push_back(run_name);
      run_results_name_to_data[run_name] = HeaderToDataInCsvFile(results_fp);
    }
  }

  //get input "signature" for run mapped to optimized implementation runtime for each run on input
  //as well as input "signature" for each input mapped to input info to show in evaluation output
  std::map<std::string, std::map<std::array<std::string, 3>, std::string, run_eval::LessThanRunSigHdrs>> input_to_runtime_across_archs;
  std::set<std::array<std::string, 3>, run_eval::LessThanRunSigHdrs> input_set;
  std::map<std::array<std::string, 3>, std::vector<std::string>, run_eval::LessThanRunSigHdrs> input_set_to_input_disp;
  for (const auto& run_result : run_results_name_to_data) {
    input_to_runtime_across_archs[run_result.first] = std::map<std::array<std::string, 3>, std::string, run_eval::LessThanRunSigHdrs>();
    const auto& result_keys_to_result_vect = run_result.second.second;
    const unsigned int tot_num_runs = result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[0])).size();
    for (size_t num_run = 0; num_run < tot_num_runs; num_run++) {
      const std::array<std::string, 3> run_input{
        result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[0]))[num_run],
        result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[1]))[num_run],
        result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[2]))[num_run]};
      input_to_runtime_across_archs[run_result.first][run_input] = result_keys_to_result_vect.at(std::string(run_eval::kOptimizedRuntimeHeader))[num_run];
      input_set.insert(run_input);
      //add mapping from run input signature to run input to be displayed
      if (!(input_set_to_input_disp.contains(run_input))) {
        input_set_to_input_disp[run_input] = std::vector<std::string>();
        for (const auto& dispParam : eval_across_runs_in_params_show) {
          input_set_to_input_disp[run_input].push_back(result_keys_to_result_vect.at(dispParam)[num_run]);
        }
      }
    }
  }

  //get header to data of each set of speedups
  //iterate through all speedup data files and get run name to results
  std::map<std::string, std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>>> speedup_results_name_to_data;
  std::vector<std::string> speedup_headers_in_order;
  for (const auto& run_name : run_names) {
    std::filesystem::path run_speedup_fp = imp_results_file_path / run_eval::kImpResultsSpeedupsFolderName /
      (std::string(run_name) + '_' + std::string(run_eval::kSpeedupsDescFileName) + std::string(run_eval::kCsvFileExtension));
    if (std::filesystem::is_regular_file(run_speedup_fp)) {
      speedup_results_name_to_data[run_name] = HeaderToDataInCsvFile(run_speedup_fp);
    }
  }

  //generate results across architectures
  std::ostringstream result_across_archs_sstream;
  //add text to display on top of results across architecture comparison file
  for (const auto& comp_file_top_text_line : eval_across_runs_top_text) {
    result_across_archs_sstream << comp_file_top_text_line << std::endl;
  }
  result_across_archs_sstream << std::endl;

  //write out the name of each input parameter to be displayed
  for (const auto& input_param_disp_header : eval_across_runs_in_params_show) {
    result_across_archs_sstream << input_param_disp_header << ',';
  }

  //write each architecture name and save order of architectures with speedup corresponding to first
  //speedup header
  //order of architectures from left to right is in speedup from largest to smallest
  std::set<std::pair<float, std::string>, std::greater<std::pair<float, std::string>>> run_names_in_order_w_speedup;
  std::string first_speedup_header;
  for (const auto& arch_w_speedup_data : speedup_results_name_to_data.cbegin()->second.first) {
    if (!(arch_w_speedup_data.empty())) {
      first_speedup_header = arch_w_speedup_data;
      break;
    }
  }
  //add first speedup with run name to get names of runs in order of fastest to slowest first speedup
  for (const auto& arch_w_speedup_data : speedup_results_name_to_data) {
    const float avgSpeedupVsBase = std::stof(std::string(arch_w_speedup_data.second.second.at(first_speedup_header).at(0)));
    run_names_in_order_w_speedup.insert({avgSpeedupVsBase, arch_w_speedup_data.first});
  }
  //write all the run names in order from fastest to slowest
  for (const auto& run_name : run_names_in_order_w_speedup) {
    result_across_archs_sstream << run_name.second << ',';
  }
  result_across_archs_sstream << std::endl;

  //write input data and runtime for each run for each architecture
  for (const auto& curr_run_input : input_set_to_input_disp) {
    for (const auto& run_input_val : curr_run_input.second) {
      result_across_archs_sstream << run_input_val << ',';
    }
    for (const auto& run_name : run_names_in_order_w_speedup) {
      if (input_to_runtime_across_archs.at(run_name.second).contains(curr_run_input.first)) {
        result_across_archs_sstream << input_to_runtime_across_archs.at(run_name.second).at(curr_run_input.first) << ',';
      }
    }
    result_across_archs_sstream << std::endl;
  }
  result_across_archs_sstream << std::endl;

  //write each average speedup with results for each architecture
  result_across_archs_sstream << "Average Speedups" << std::endl;
  std::string first_run_name = speedup_results_name_to_data.cbegin()->first;
  for (const auto& speedup_header : speedup_results_name_to_data.at(first_run_name).first) {
    //don't process if header is empty
    if (!(speedup_header.empty())) {
      result_across_archs_sstream << speedup_header << ',';
      //add empty cell for each input parameter after the first that's displayed so speedup shown on same line as runtime for architecture
      for (size_t i = 1; i < eval_across_runs_in_params_show.size(); i++) {
        result_across_archs_sstream << ',';
      }
      //write speedup for each architecture in separate cells in horizontal direction
      for (const auto& run_name : run_names_in_order_w_speedup) {
        result_across_archs_sstream << speedup_results_name_to_data.at(run_name.second).second.at(speedup_header).at(0) << ',';
      }
      //continue to next row of table
      result_across_archs_sstream << std::endl;
    }
  }
  //get file path for evaluation across runs and save evaluation across runs to csv file
  std::filesystem::path results_across_run_fp = imp_results_file_path /
    (std::string(run_eval::kEvalAcrossRunsFileName) + std::string(run_eval::kCsvFileExtension));
  std::ofstream eval_results_across_run_str(results_across_run_fp);
  eval_results_across_run_str << result_across_archs_sstream.str();
  std::cout << "Evaluation of results across all runs in " << results_across_run_fp << std::endl;
}

//get mapping of headers to data in csv file for run results and speedups
//assumed that there are no commas in data since it is used as delimiter between data
//first output is headers in order, second output is mapping of headers to results
std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>>
EvaluateAcrossRuns::HeaderToDataInCsvFile(const std::filesystem::path& csv_file_path) const {
  std::ifstream csv_file_str(csv_file_path);
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
  return {data_headers, header_to_data};
}
