/*
 * EvaluateAcrossRuns.cpp
 *
 *  Created on: March 2, 2024
 *      Author: scott
 * 
 *  Definition of functions in class for evaluating results across multiple runs, potentially on different architectures.
 */

#include <fstream>
#include <algorithm>
#include <set>
#include <utility>
#include <iostream>
#include "RunSettingsParams/InputSignature.h"
#include "EvaluateAcrossRuns.h"
#include "RunResultsSpeedups.h"

//process runtime and speedup data across multiple runs (likely on different architectures)
//from csv files corresponding to each run and
//write results to file where the runtimes and speedups for each run are shown in a single file
//where the runs are displayed from fastest to slowest
void EvaluateAcrossRuns::operator()(
  const std::filesystem::path& imp_results_file_path,
  const std::vector<std::string>& eval_across_runs_top_text,
  const std::vector<std::string>& eval_across_runs_in_params_show,
  const std::vector<std::string>& speedup_headers) const
{
  //retrieve names of runs with results
  //run names usually correspond to architecture of run
  const std::vector<std::string> run_names = GetRunNames(imp_results_file_path);
  
  //initialize vector of run results with speedups for each run name
  std::map<std::string, RunResultsSpeedups> run_results_by_name;
  for (const auto& run_name : run_names) {
    run_results_by_name.insert(
      {run_name, RunResultsSpeedups(imp_results_file_path, run_name)});
  }

  //get header to data of run results and speedups for each run
  //as well as mapping from input signature to runtime
  std::map<std::string, std::map<std::string, std::vector<std::string>>> speedup_results_name_to_data;
  std::map<std::string, std::map<InputSignature, std::string>> input_to_runtime_across_archs;
  for (const auto& run_name : run_names) {
    speedup_results_name_to_data[run_name] = run_results_by_name.at(run_name).Speedups();
    input_to_runtime_across_archs[run_name] = run_results_by_name.at(run_name).InputsToKeyVal(
      run_eval::kOptimizedRuntimeHeader);
  }

  //get run inputs to parameters to display in evaluation across runs
  const auto& inputs_to_runtimes = run_results_by_name.at(run_names[0]).InputsToKeyVal(
    run_eval::kOptimizedRuntimeHeader);
  std::map<InputSignature, std::vector<std::string>> input_set_to_input_disp;
  for (const auto& input_runtime : inputs_to_runtimes) {
    input_set_to_input_disp.insert({input_runtime.first, std::vector<std::string>()});
    //add input parameters to display in evaluation across runs
    for (const auto& disp_param : eval_across_runs_in_params_show) {
      input_set_to_input_disp.at(input_runtime.first).push_back(
        run_results_by_name.at(run_names[0]).DataForInput(input_runtime.first).at(disp_param));
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

  //get header to use for speedup ordering
  //use first speedup in speedup headers for ordering
  //of runs from fastest to slowest
  const auto speedup_ordering = speedup_headers.front();

  //write each evaluation run name and save order of runs with speedup corresponding to first
  //speedup header
  //order of run names is in speedup from largest to smallest
  std::set<std::pair<float, std::string>, std::greater<std::pair<float, std::string>>>
    run_names_in_order_w_speedup;

  //add first speedup with run name to order runs from fastest to slowest based on first speedup
  for (const auto& arch_w_speedup_data : speedup_results_name_to_data) {
    if (arch_w_speedup_data.second.contains(speedup_ordering)) {
      const float avgSpeedupVsBase = 
        std::stof(std::string(arch_w_speedup_data.second.at(speedup_ordering).at(0)));
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
    for (const auto& run_name : run_names_in_order_w_speedup)
    {
      if (input_to_runtime_across_archs.at(run_name.second).contains(curr_run_input.first))
      {
        result_across_archs_sstream << 
          input_to_runtime_across_archs.at(run_name.second).at(curr_run_input.first);
      }
      result_across_archs_sstream << ',';
    }
    result_across_archs_sstream << std::endl;
  }
  result_across_archs_sstream << std::endl;

  //write average speedup results for each run that correspond to a number of
  //different evaluations of runtimes compared to a baseline
  result_across_archs_sstream << "Average Speedups" << std::endl;
  const std::string first_run_name = speedup_results_name_to_data.cbegin()->first;
  for (const auto& speedup_header : speedup_headers) {
    //don't process if header is empty
    if (!(speedup_header.empty())) {
      result_across_archs_sstream << speedup_header << ',';
      //add empty cell for each input parameter after the first that's displayed
      //so speedup shown in the same column same line as runtime
      for (size_t i = 1; i < eval_across_runs_in_params_show.size(); i++) {
        result_across_archs_sstream << ',';
      }
      //write speedup for each run in separate cells in horizontal direction
      //where each column corresponds to a different evaluation run
      for (const auto& run_name : run_names_in_order_w_speedup) {
        if (speedup_results_name_to_data.at(run_name.second).contains(speedup_header)) {
          result_across_archs_sstream <<
            speedup_results_name_to_data.at(run_name.second).at(speedup_header).at(0) << ',';
        }
        else {
          result_across_archs_sstream << ',';
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

//function to get names of runs with results from implementation results file path
std::vector<std::string> EvaluateAcrossRuns::GetRunNames(
  const std::filesystem::path& imp_results_file_path) const
{
  //iterate through all run results files all run names with results and speedups
  //create directory iterator with all results files
  std::filesystem::directory_iterator results_files_iter =
    std::filesystem::directory_iterator(imp_results_file_path / run_eval::kImpResultsRunDataFolderName);
  std::filesystem::directory_iterator speedups_files_iter =
    std::filesystem::directory_iterator(imp_results_file_path / run_eval::kImpResultsSpeedupsFolderName);
  std::vector<std::string> run_names;
  for (const auto& results_fp : results_files_iter)
  {
    std::string file_name_no_ext = results_fp.path().stem();
    if (file_name_no_ext.ends_with("_" + std::string(run_eval::kRunResultsDescFileName)))
    {
      const std::string run_name =
        file_name_no_ext.substr(
          0, file_name_no_ext.find("_" + std::string(run_eval::kRunResultsDescFileName)));
      run_names.push_back(run_name);
    }
  }

  //remove run name from runs to evaluate across runs if no speedup data
  //example where this could be the case is baseline data that is used for comparison with other runs
  for (auto run_names_iter = run_names.begin(); run_names_iter != run_names.end();)
  {
    std::filesystem::path run_speedup_fp = 
      imp_results_file_path / run_eval::kImpResultsSpeedupsFolderName /
      (std::string(*run_names_iter) + '_' + std::string(run_eval::kSpeedupsDescFileName) +
        std::string(run_eval::kCsvFileExtension));
    //remove run from evaluation if no valid speedup data file that corresponds to run results data file
    if ((!(std::filesystem::exists(run_speedup_fp))) ||
        (!(std::filesystem::is_regular_file(run_speedup_fp))))
    {
      run_names_iter = run_names.erase(std::ranges::find(run_names, *run_names_iter));
    }
    else {
      run_names_iter++;
    }
  }

  return run_names;
}
