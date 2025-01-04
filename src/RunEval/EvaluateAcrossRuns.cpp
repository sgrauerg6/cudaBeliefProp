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
 * @brief Definitions of functions in class for evaluating results across
 * multiple runs, potentially on different architectures.
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

//process runtime and speedup data across multiple runs (likely on different
//architectures) from csv files corresponding to each run and write results to
//file where the runtimes and speedups for each run are shown in a single file
//and where the runs are displayed left to right from fastest to slowest
void EvaluateAcrossRuns::operator()(
  const std::filesystem::path& imp_results_file_path,
  const std::vector<std::string>& eval_across_runs_top_text,
  const std::vector<std::string>& eval_across_runs_in_params_show) const
{
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

  //genereation data mappings for evaluation including run results and speedups
  //for each run, mapping from input signature to runtime, and run inputs to
  //parameters to display in evaluation across runs as well as speedup headers
  //in order
  const EvalAcrossRunsData eval_data =
    GenEvalAcrossRunsData(
      run_results_by_name,
      eval_across_runs_in_params_show);

  //get run names in order from fastest to slowest based on first speedup
  //header
  std::vector<std::string> run_names_ordered =
    OrderedRunNames(eval_data);
  
  //add any run names not in ordered run names (due to not having data
  //corresponding to first speedup header) to end of ordered runs
  std::copy_if(
    run_names.begin(),
    run_names.end(),
    std::back_inserter(run_names_ordered),
    [&run_names_ordered](const auto& run_name) {
      return (
        std::find(
          run_names_ordered.begin(),
          run_names_ordered.end(),
          run_name) == run_names_ordered.end());
    });

  //write output evaluation across runs to file
  WriteEvalAcrossRunsToFile(
    imp_results_file_path,
    eval_across_runs_top_text,
    eval_across_runs_in_params_show,
    eval_data,
    run_names_ordered);
}

/**
  * @brief Generate data for evaluating results across runs
  * 
  * @param run_results_by_name
  * @param eval_across_runs_in_params_show
  * @return EvalAcrossRunsData structure with data for evaluating results
  * across runs
  */
EvalAcrossRunsData EvaluateAcrossRuns::GenEvalAcrossRunsData(
  const std::map<std::string, RunResultsSpeedups>& run_results_by_name,
  const std::vector<std::string>& eval_across_runs_in_params_show) const
{
  EvalAcrossRunsData eval_data;
  for (const auto& [run_name, run_results] : run_results_by_name)
  {
    eval_data.speedup_results_name_to_data.insert(
      {run_name,
       run_results.Speedups()});
    eval_data.input_to_runtime_across_archs.insert(
      {run_name,
       run_results.InputsToKeyVal(run_eval::kOptimizedRuntimeHeader)});
    const auto& inputs_to_runtimes = run_results.InputsToKeyVal(
      run_eval::kOptimizedRuntimeHeader);
    //go through inputs for current run results
    //need to go through every run so that all inputs in every run are included
    for (const auto& [input_sig, _] : inputs_to_runtimes) {
      //check if input already addded to set of inputs
      if (!(eval_data.inputs_to_params_disp_ordered.contains(input_sig))) {
        eval_data.inputs_to_params_disp_ordered.insert(
          {input_sig, std::vector<std::string>()});
        //add input parameters to display in evaluation across runs
        for (const auto& disp_param : eval_across_runs_in_params_show) {
          eval_data.inputs_to_params_disp_ordered.at(input_sig).push_back(
            run_results.DataForInput(input_sig).at(disp_param));
        }
      }
    }
    //go through speedups for run and add to speedup headers if not already
    //included
    const auto run_speedups_ordered = run_results.SpeedupHeadersOrder();
    for (auto run_speedup_iter = run_speedups_ordered.cbegin();
              run_speedup_iter < run_speedups_ordered.cend();
              run_speedup_iter++)
    {
      //ignore speedup if all whitespace
      if (std::all_of(
            run_speedup_iter->begin(),
            run_speedup_iter->end(),
            [](unsigned char c){ return std::isspace(c); }))
      {
        continue;
      }
      //check if speedup in run is included in current evaluation speedups and
      //add it in expected position in evaluation speedups if not
      if (std::none_of(eval_data.speedup_headers.cbegin(), 
                       eval_data.speedup_headers.cend(),
                       [run_speedup_iter](const auto& header){
                         return (header == *run_speedup_iter);
                       }))
      {
        //get relative position of previous ordered speedup header
        auto iter_prev_ordered_header = run_speedup_iter;
        while (iter_prev_ordered_header != run_speedups_ordered.cbegin())
        {
          iter_prev_ordered_header--;
          if (std::find(eval_data.speedup_headers.cbegin(),
                        eval_data.speedup_headers.cend(),
                        *iter_prev_ordered_header) !=
                eval_data.speedup_headers.cend())
          {
            break;
          }
        }
        
        //add speedup header in front of previous ordered speedup header if
        //not first speedup header for run and previous ordered header exists
        //in speedup headers
        if (iter_prev_ordered_header != run_speedups_ordered.cbegin())
        {
          //find position in evaluation speedups of previous ordered header
          //and add new header in front of it
          eval_data.speedup_headers.insert(
            (std::find(eval_data.speedup_headers.cbegin(),
                       eval_data.speedup_headers.cend(),
                       *iter_prev_ordered_header) + 1),
            *run_speedup_iter);
        }
        else {
          //add speedup header to front of evaluation speedup headers if no
          //previous ordered header in speedups for run
          eval_data.speedup_headers.insert(
            eval_data.speedup_headers.cbegin(),
            *run_speedup_iter);
        }
      }
    }
  }

  //return resulting evaluation data
  return eval_data;
}

/**
 * @brief Get run names in order from fastest to slowest based on
 * speedup header.<br>
 * If no speedup header is given, use the first speedup header in evaluation
 * data.
 * 
 * @param eval_data
 */
std::vector<std::string> EvaluateAcrossRuns::OrderedRunNames(
  const EvalAcrossRunsData& eval_data,
  const std::optional<std::string>& speedup_header) const
{
  //get header to use for speedup ordering of runs
  //use first speedup in speedup headers if no input speedup header given
  const std::string speedup_ordering =
    speedup_header.value_or(eval_data.speedup_headers.front());
  
  //declare vector with run names paired with reference wrapper containing
  //speedup results for run
  //reference wrapper used to prevent need to copy speedup results
  std::vector<std::pair<
    std::string,
    std::reference_wrapper<const std::map<std::string, std::vector<std::string>>>>>
  runs_w_speedup_data;

  //populate vector of run names paired with reference wrapper containing speedup data
  std::transform(
    eval_data.speedup_results_name_to_data.cbegin(),
    eval_data.speedup_results_name_to_data.cend(),
    std::back_inserter(runs_w_speedup_data),
    [](const auto& run_name_w_speedup) {
      return std::pair<std::string, std::reference_wrapper<const std::map<std::string, std::vector<std::string>>>>{
        run_name_w_speedup.first, std::cref(run_name_w_speedup.second)};
    });
  
  //remove runs where current speedup ordering header does not have data or
  //data is blank (currently assumed that data is a valid float)
  std::erase_if(
    runs_w_speedup_data,
    [&speedup_ordering](const auto& run_name_w_speedup) {
      if (!(run_name_w_speedup.second.get().contains(speedup_ordering))) {
        return true;
      }
      else {
        const std::string speedup_str =
          run_name_w_speedup.second.get().at(speedup_ordering).at(0);
        return std::all_of(
          speedup_str.begin(),
          speedup_str.end(),
          [](unsigned char c) { return isspace(c); });
      }
    });
  
  //sort runs from greatest speedup to lowest speedup according to speedup
  //header
  std::sort(runs_w_speedup_data.begin(), runs_w_speedup_data.end(),
    [&speedup_ordering](const auto& run_w_speedups_1, const auto& run_w_speedups_2) {
      //return true if run 1 has greater speedup than run 2
      return (std::stof(run_w_speedups_1.second.get().at(speedup_ordering).at(0)) >
              std::stof(run_w_speedups_2.second.get().at(speedup_ordering).at(0)));
    });

  //generate ordered run names from greatest speedup to least speedup
  //according to speedup header
  std::vector<std::string> ordered_run_names;
  std::transform(
    runs_w_speedup_data.cbegin(),
    runs_w_speedup_data.cend(),
    std::back_inserter(ordered_run_names),
    [](const auto& run_name_speedup) { return run_name_speedup.first; });

  //return run names ordered from greatest speedup to least speedup
  return ordered_run_names;

  //write each evaluation run name and save order of runs with speedup
  //corresponding to first speedup header
  //order of run names is in speedup from largest to smallest
  /*auto cmp_speedup = 
    [](const std::pair<std::string, float>& a,
       const std::pair<std::string, float>& b)
       { return a.second > b.second; };
  std::set<std::pair<std::string, float>, decltype(cmp_speedup)>
    run_names_in_order_w_speedup;

  //generate and add pairs of run name with corresponding "ordering" speedup
  //for each run to set to get sorted order of runs from fastest to slowest
  //based on "ordering" speedup
  for (const auto& [run_name, speedup_results] :
       eval_data.speedup_results_name_to_data)
  {
    run_names_in_order_w_speedup.insert(
      {run_name, 
       speedup_results.contains(speedup_ordering) ?
       std::stof(std::string(speedup_results.at(speedup_ordering).at(0))) :
       0});
  }*

  //generate ordered run names from fastest to slowest
  std::vector<std::string> ordered_run_names;
  std::transform(
    run_names_in_order_w_speedup.cbegin(),
    run_names_in_order_w_speedup.cend(),
    std::back_inserter(ordered_run_names),
    [](const auto& run_name_speedup) { return run_name_speedup.first; });
  
  return ordered_run_names;*/
}

//write output evaluation across runs to file
void EvaluateAcrossRuns::WriteEvalAcrossRunsToFile(
  const std::filesystem::path& imp_results_file_path,
  const std::vector<std::string>& eval_across_runs_top_text,
  const std::vector<std::string>& eval_across_runs_in_params_show,
  const EvalAcrossRunsData& eval_data,
  const std::vector<std::string>& run_names_ordered) const
{
  //write results across architectures to output file
  //file path for evaluation across runs
  const std::filesystem::path results_across_run_fp = 
    imp_results_file_path /
    (std::string(run_eval::kEvalAcrossRunsFileName) +
     std::string(run_eval::kCsvFileExtension));
  
  //initialize output stream for file showing evaluation across runs
  std::ofstream eval_results_across_run_str(results_across_run_fp);

  //add text to display on top of results across architecture comparison file
  for (const auto& comp_file_top_text_line : eval_across_runs_top_text) {
    eval_results_across_run_str << comp_file_top_text_line << std::endl;
  }
  eval_results_across_run_str << std::endl;

  //write out the name of each input parameter to be displayed
  for (const auto& input_param_disp_header : eval_across_runs_in_params_show) {
    eval_results_across_run_str << input_param_disp_header << ',';
  }

  //write all the run names in order from fastest to slowest
  for (const auto& run_name : run_names_ordered) {
    eval_results_across_run_str << run_name << ',';
  }
  eval_results_across_run_str << std::endl;

  //write evaluation stereo set info, bp parameters, and total runtime for
  //optimized bp implementation across each run in the evaluation
  for (const auto& [input_sig, params_display_ordered] :
       eval_data.inputs_to_params_disp_ordered)
  {
    for (const auto& param_val_disp : params_display_ordered) {
      eval_results_across_run_str << param_val_disp << ',';
    }
    for (const auto& run_name : run_names_ordered)
    {
      if (eval_data.input_to_runtime_across_archs.at(
            run_name).contains(input_sig))
      {
        eval_results_across_run_str << 
          eval_data.input_to_runtime_across_archs.at(run_name).at(input_sig);
      }
      eval_results_across_run_str << ',';
    }
    eval_results_across_run_str << std::endl;
  }
  eval_results_across_run_str << std::endl;

  //write average speedup results for each run that correspond to a number of
  //different evaluations of runtimes compared to a baseline
  eval_results_across_run_str << "Average Speedups" << std::endl;
  for (const auto& speedup_header : eval_data.speedup_headers) {
    //don't process if header is empty
    if (!(speedup_header.empty())) {
      eval_results_across_run_str << speedup_header << ',';
      //add empty cell for each input parameter after the first that's
      //displayed so speedup shown in the same column same line as runtime
      for (size_t i = 1; i < eval_across_runs_in_params_show.size(); i++) {
        eval_results_across_run_str << ',';
      }
      //write speedup for each run in separate cells in horizontal direction
      //where each column corresponds to a different evaluation run
      for (const auto& run_name : run_names_ordered) {
        if (eval_data.speedup_results_name_to_data.at(run_name).contains(
              speedup_header))
        {
          eval_results_across_run_str <<
            eval_data.speedup_results_name_to_data.at(run_name).at(
              speedup_header).at(0) << ',';
        }
        else {
          eval_results_across_run_str << ',';
        }
      }
      //continue to next row of table to write data for next speedup result
      eval_results_across_run_str << std::endl;
    }
  }

  //go through each speedup and display runs ordered from highest speedup to lowest
  //speedup
  eval_results_across_run_str << std::endl;
  eval_results_across_run_str << "Runs ordered by speedup (highest to lowest)" << std::endl;
  for (const auto& speedup_header : eval_data.speedup_headers) {
    //don't process if header is empty
    if (!(speedup_header.empty())) {
      eval_results_across_run_str << speedup_header << ',';
      //get runs ordered by speedup
      const auto ordered_runs_speedup =
        OrderedRunNames(
          eval_data,
          speedup_header);
      for (const auto& run_name : ordered_runs_speedup) {
        const auto speedup =
          eval_data.speedup_results_name_to_data.at(run_name).at(
            speedup_header).at(0);
        eval_results_across_run_str << run_name << " - " << std::setprecision(3)
                                    << std::fixed << std::stof(speedup) << ',';
      }
    }
    //continue to next row of table to write data for next speedup result
    eval_results_across_run_str << std::endl;
  }
  
  //write location of evaluation results across runs to output console
  std::cout << "Evaluation of results across all runs in "
            << results_across_run_fp << std::endl;
}

//function to get names of runs with results from implementation results
//file path
std::vector<std::string> EvaluateAcrossRuns::GetRunNames(
  const std::filesystem::path& imp_results_file_path) const
{
  //iterate through all run results files
  //create directory iterator with path to run results files
  std::filesystem::directory_iterator results_files_iter =
    std::filesystem::directory_iterator(
      imp_results_file_path / run_eval::kImpResultsRunDataFolderName);
  std::vector<std::string> run_names;
  for (const auto& results_fp : results_files_iter)
  {
    //get run name from run results file name
    //and add to vector of run names
    std::string file_name_no_ext = results_fp.path().stem();
    if (file_name_no_ext.ends_with(
      "_" + std::string(run_eval::kRunResultsDescFileName)))
    {
      const std::string run_name =
        file_name_no_ext.substr(
          0,
          file_name_no_ext.find(
            "_" + std::string(run_eval::kRunResultsDescFileName)));
      run_names.push_back(run_name);
    }
  }

  //remove run name from runs to evaluate across runs if no speedup data
  //example where this could be the case is baseline data that is used for
  //comparison with other runs
  for (auto run_names_iter = run_names.cbegin();
       run_names_iter != run_names.cend();)
  {
    const std::filesystem::path run_speedup_fp = 
      imp_results_file_path / run_eval::kImpResultsSpeedupsFolderName /
      (std::string(*run_names_iter) + '_' +
       std::string(run_eval::kSpeedupsDescFileName) +
       std::string(run_eval::kCsvFileExtension));
    //remove run from evaluation if no valid speedup data file that corresponds
    //to run results data file
    if ((!(std::filesystem::exists(run_speedup_fp))) ||
        (!(std::filesystem::is_regular_file(run_speedup_fp))))
    {
      run_names_iter =
        run_names.erase(std::ranges::find(run_names, *run_names_iter));
    }
    else {
      run_names_iter++;
    }
  }

  return run_names;
}
