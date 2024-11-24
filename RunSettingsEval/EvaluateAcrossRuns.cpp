/*
 * EvaluateAcrossRuns.cpp
 *
 *  Created on: March 2, 2024
 *      Author: scott
 * 
 *  Definition of functions in class for evaluating results across multiple runs, potentially on different architectures.
 */

#include "EvaluateAcrossRuns.h"
#include "EvalInputSignature.h"

//process runtime and speedup data across multiple runs (likely on different architectures)
//from csv files corresponding to each run and
//write results to file where the runtimes and speedups for each run are shown in a single file
//where the runs are displayed from fastest to slowest
void EvaluateAcrossRuns::operator()(
  const std::filesystem::path& imp_results_file_path,
  const std::vector<std::string>& eval_across_runs_top_text,
  const std::vector<std::string>& eval_across_runs_in_params_show,
  const std::pair<std::vector<std::string>, size_t>& speedup_headers_w_ordering_idx) const
{
  //get header to use for speedup ordering
  auto speedup_ordering = speedup_headers_w_ordering_idx.first[
    speedup_headers_w_ordering_idx.second];

  //get header to data of each set of run results
  //iterate through all run results files and get run name to results
  //create directory iterator with all results files
  std::filesystem::directory_iterator results_files_iter =
    std::filesystem::directory_iterator(imp_results_file_path / run_eval::kImpResultsRunDataFolderName);
  std::vector<std::string> run_names;
  std::map<std::string, std::map<std::string, std::vector<std::string>>> run_results_name_to_data;
  for (const auto& results_fp : results_files_iter) {
    std::string file_name_no_ext = results_fp.path().stem();
    if (file_name_no_ext.ends_with("_" + std::string(run_eval::kRunResultsDescFileName))) {
      const std::string run_name = file_name_no_ext.substr(0, file_name_no_ext.find("_" + std::string(run_eval::kRunResultsDescFileName)));
      run_names.push_back(run_name);
      run_results_name_to_data[run_name] = HeaderToDataInCsvFile(results_fp);
    }
  }

  //get input "signature" for run mapped to optimized implementation runtime for each run on input
  //as well as input "signature" for each input mapped to input info to show in evaluation output
  std::map<std::string, std::map<EvalInputSignature, std::string>> input_to_runtime_across_archs;
  std::map<EvalInputSignature, std::vector<std::string>> input_set_to_input_disp;
  std::vector<decltype(run_results_name_to_data)::key_type> key_results_remove;
  for (const auto& run_result : run_results_name_to_data) {
    input_to_runtime_across_archs[run_result.first] = std::map<EvalInputSignature, std::string>();
    const auto& result_keys_to_result_vect = run_result.second;
    if (result_keys_to_result_vect.contains(std::string(run_eval::kRunInputSigHeaders[0]))) {
      const unsigned int tot_num_runs = result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[0])).size();
      for (unsigned int num_run = 0; num_run < tot_num_runs; num_run++) {
        //get unique input signature for evaluation run (evaluation data number, data type, setting of whether to not to
        //have loops with templated iteration counts)
        const EvalInputSignature run_input({
          result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[0]))[num_run],
          result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[1]))[num_run],
          result_keys_to_result_vect.at(std::string(run_eval::kRunInputSigHeaders[2]))[num_run]});
        //add mapping of total runtime to corresponding run name and input signature
        input_to_runtime_across_archs[run_result.first][run_input] =
          result_keys_to_result_vect.at(std::string(run_eval::kOptimizedRuntimeHeader))[num_run];
        //add mapping from run input signature to run input to be displayed
        if (!(input_set_to_input_disp.contains(run_input))) {
          input_set_to_input_disp[run_input] = std::vector<std::string>();
          //add input parameters to display in evaluation across runs
          for (const auto& disp_param : eval_across_runs_in_params_show) {
            input_set_to_input_disp[run_input].push_back(result_keys_to_result_vect.at(disp_param)[num_run]);
          }
        }
      }
    }
    else {
      std::cout << "Invalid data in " << run_result.first << std::endl;
      key_results_remove.push_back(run_result.first);
    }
  }
  //remove any results that have been flagged to remove due to not having
  //the necessary data
  for (const auto& key_data_remove : key_results_remove) {
    std::cout << "Remove run result: " << key_data_remove << std::endl;
    run_results_name_to_data.erase(key_data_remove);
  }

  //get header to data of each set of speedups
  //iterate through all speedup data files and get run name to results
  std::map<std::string, std::map<std::string, std::vector<std::string>>> speedup_results_name_to_data;
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

  //write each evaluation run name and save order of runs with speedup corresponding to first
  //speedup header
  //order of run names is in speedup from largest to smallest
  std::set<std::pair<float, std::string>, std::greater<std::pair<float, std::string>>> run_names_in_order_w_speedup;

  //add first speedup with run name to order runs from fastest to slowest based on first speedup
  for (const auto& arch_w_speedup_data : speedup_results_name_to_data) {
    if (arch_w_speedup_data.second.contains(speedup_ordering)) {
      const float avgSpeedupVsBase = std::stof(std::string(arch_w_speedup_data.second.at(speedup_ordering).at(0)));
      run_names_in_order_w_speedup.insert({avgSpeedupVsBase, arch_w_speedup_data.first});
    }
    else {
      run_names_in_order_w_speedup.insert({0, arch_w_speedup_data.first});
    }
  }

  //write all the run names in order from fastest to slowest
  for (const auto& run_name : run_names_in_order_w_speedup) {
    result_across_archs_sstream << run_name.second << ',';
  }
  result_across_archs_sstream << std::endl;

  //write evaluation stereo set info, bp parameters, and total runtime for optimized bp implementation
  //across each run in the evaluation
  for (const auto& curr_run_input : input_set_to_input_disp) {
    for (const auto& run_input_val : curr_run_input.second) {
      result_across_archs_sstream << run_input_val << ',';
    }
    for (const auto& run_name : run_names_in_order_w_speedup) {
      if (input_to_runtime_across_archs.at(run_name.second).contains(curr_run_input.first)) {
        result_across_archs_sstream << input_to_runtime_across_archs.at(run_name.second).at(curr_run_input.first) << ',';
      }
      else {
        result_across_archs_sstream << "NO RESULT" << ',';
      }
    }
    result_across_archs_sstream << std::endl;
  }
  result_across_archs_sstream << std::endl;

  //write average speedup results for each run that correspond to a number of
  //different evaluations of runtimes compared to a baseline
  result_across_archs_sstream << "Average Speedups" << std::endl;
  const std::string first_run_name = speedup_results_name_to_data.cbegin()->first;
  for (const auto& speedup_header : speedup_headers_w_ordering_idx.first) {
    //don't process if header is empty
    if (!(speedup_header.empty())) {
      result_across_archs_sstream << speedup_header << ',';
      //add empty cell for each input parameter after the first that's displayed so speedup shown in the same column same line as runtime
      for (size_t i = 1; i < eval_across_runs_in_params_show.size(); i++) {
        result_across_archs_sstream << ',';
      }
      //write speedup for each run in separate cells in horizontal direction where each column corresponds to a different evaluation run
      for (const auto& run_name : run_names_in_order_w_speedup) {
        if (speedup_results_name_to_data.at(run_name.second).contains(speedup_header)) {
          result_across_archs_sstream << speedup_results_name_to_data.at(run_name.second).at(speedup_header).at(0) << ',';
        }
        else {
          result_across_archs_sstream << "NO DATA" << ',';
        }
      }
      //continue to next row of table to write data for next speedup result
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
std::map<std::string, std::vector<std::string>>
EvaluateAcrossRuns::HeaderToDataInCsvFile(const std::filesystem::path& csv_file_path) const {
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
