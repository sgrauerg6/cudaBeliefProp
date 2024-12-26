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
 * @file EvaluateAcrossRuns.cpp
 * @author Scott Grauer-Gray
 * @brief Definition of functions in class for evaluating results across multiple runs, potentially on different architectures.
 * 
 * @copyright Copyright (c) 2024
 */

#include <fstream>
#include <algorithm>
#include <set>
#include <utility>
#include <iostream>
#include <algorithm>
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
  //initialize speedup headers for output results across runs
  auto speedup_headers_eval = speedup_headers;

  //retrieve names of runs with results
  //run names usually correspond to architecture of run
  const std::vector<std::string> run_names =
    GetRunNames(imp_results_file_path);
  
  //initialize vector of run results with speedups for each run name
  std::map<std::string, RunResultsSpeedups> run_results_by_name;
  for (const auto& run_name : run_names) {
    run_results_by_name.insert(
      {run_name, RunResultsSpeedups(imp_results_file_path, run_name)});
  }

  //get header to data of run results and speedups for each run, mapping from
  //input signature to runtime, and run inputs to parameters to display in
  //evaluation across runs
  std::map<std::string, std::map<std::string, std::vector<std::string>>> speedup_results_name_to_data;
  std::map<std::string, std::map<InputSignature, std::string>> input_to_runtime_across_archs;
  std::map<InputSignature, std::vector<std::string>> inputs_to_params_disp_ordered;
  for (const auto& [run_name, run_results] : run_results_by_name) {
    speedup_results_name_to_data.insert(
      {run_name,
       run_results.Speedups()});
    input_to_runtime_across_archs.insert(
      {run_name,
       run_results.InputsToKeyVal(run_eval::kOptimizedRuntimeHeader)});
    const auto& inputs_to_runtimes = run_results.InputsToKeyVal(
      run_eval::kOptimizedRuntimeHeader);
    //go through inputs for current run results
    //need to go through every run so that all inputs in every run are included
    for (const auto& [input_sig, _] : inputs_to_runtimes) {
      //check if input already addded to set of inputs
      if (!(inputs_to_params_disp_ordered.contains(input_sig))) {
        inputs_to_params_disp_ordered.insert({input_sig, std::vector<std::string>()});
        //add input parameters to display in evaluation across runs
        for (const auto& disp_param : eval_across_runs_in_params_show) {
          inputs_to_params_disp_ordered.at(input_sig).push_back(
            run_results.DataForInput(input_sig).at(disp_param));
        }
      }
    }
    //go through speedups for run and add to speedup headers if not already included
    const auto run_speedups_ordered = run_results.SpeedupHeadersOrder();
    for (auto run_speedup_iter = run_speedups_ordered.cbegin();
              run_speedup_iter < run_speedups_ordered.cend();
              run_speedup_iter++)
    {
      //check if speedup in run is included in current evaluation speedups and add it
      //in expected position in evaluation speedups if not
      if (std::none_of(speedup_headers_eval.cbegin(), 
                       speedup_headers_eval.cend(),
                       [run_speedup_iter](const auto& header){
                         return (header == *run_speedup_iter);
                       }))
      {
        //add speedup header in front of previous ordered speedup header if not first
        //ordered header
        if (run_speedup_iter != run_speedups_ordered.cbegin()) {
          //find position in evaluation speedups of previous ordered header
          //and add new header in front of it
          speedup_headers_eval.insert(
            (std::find(speedup_headers_eval.cbegin(),
                       speedup_headers_eval.cend(),
                       *(run_speedup_iter-1)) + 1),
            *run_speedup_iter);
        }
        else {
          //add speedup header to front of evaluation speedup headers if no previous
          //ordered header in speedups for run
          speedup_headers_eval.insert(speedup_headers_eval.cbegin(), *run_speedup_iter);
        }
      }
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
  auto cmp_speedup = 
    [](const std::pair<std::string, float>& a, const std::pair<std::string, float>& b) {
      return a.second > b.second; };
  std::set<std::pair<std::string, float>, decltype(cmp_speedup)> run_names_in_order_w_speedup;

  //generate and add pairs of run name with corresponding "ordering" speedup
  //for each run to set to get sorted order of runs from fastest to slowest
  //based on "ordering" speedup
  for (const auto& [run_name, speedup_results] : speedup_results_name_to_data) {
    run_names_in_order_w_speedup.insert(
      {run_name, 
       speedup_results.contains(speedup_ordering) ?
       std::stof(std::string(speedup_results.at(speedup_ordering).at(0))) :
       0});
  }

  //write all the run names in order from fastest to slowest
  for (const auto& [run_name, _] : run_names_in_order_w_speedup) {
    result_across_archs_sstream << run_name << ',';
  }
  result_across_archs_sstream << std::endl;

  //write evaluation stereo set info, bp parameters, and total runtime for optimized bp implementation
  //across each run in the evaluation
  for (const auto& [input_sig, params_display_ordered] : inputs_to_params_disp_ordered) {
    for (const auto& param_val_disp : params_display_ordered) {
      result_across_archs_sstream << param_val_disp << ',';
    }
    for (const auto& [run_name, _] : run_names_in_order_w_speedup)
    {
      if (input_to_runtime_across_archs.at(run_name).contains(input_sig))
      {
        result_across_archs_sstream << 
          input_to_runtime_across_archs.at(run_name).at(input_sig);
      }
      result_across_archs_sstream << ',';
    }
    result_across_archs_sstream << std::endl;
  }
  result_across_archs_sstream << std::endl;

  //write average speedup results for each run that correspond to a number of
  //different evaluations of runtimes compared to a baseline
  result_across_archs_sstream << "Average Speedups" << std::endl;
  for (const auto& speedup_header : speedup_headers_eval) {
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
      for (const auto& [run_name, _] : run_names_in_order_w_speedup) {
        if (speedup_results_name_to_data.at(run_name).contains(speedup_header)) {
          result_across_archs_sstream <<
            speedup_results_name_to_data.at(run_name).at(speedup_header).at(0) << ',';
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
  const std::filesystem::path results_across_run_fp = imp_results_file_path /
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
  for (auto run_names_iter = run_names.cbegin(); run_names_iter != run_names.cend();)
  {
    const std::filesystem::path run_speedup_fp = 
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
