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
 * @file EvaluateAcrossRuns.h
 * @author Scott Grauer-Gray
 * @brief Declares class with operator function to evaluate implementation runs
 * across multiple architectures. Outputs a file with speedup data on every run
 * with the runs ordered from fastest to slowest.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef EVALUATE_ACROSS_RUNS_H_
#define EVALUATE_ACROSS_RUNS_H_

#include <string>
#include <vector>
#include <filesystem>
#include "RunEvalConstsEnums.h"
#include "RunResultsSpeedups.h"

/**
 * @brief Structure to store data with mappings for evaluating results across
 * runs as well as speedup headers in order
 */
struct EvalAcrossRunsData {
  std::map<std::string, std::map<std::string, std::vector<std::string>>> speedup_results_name_to_data;
  std::map<std::string, std::map<InputSignature, std::string>> input_to_runtime_across_archs;
  std::map<InputSignature, std::vector<std::string>> inputs_to_params_disp_ordered;
  std::vector<std::string> speedup_headers;
};

/**
 * @brief Class with operator function to evaluate implementation runs across
 * multiple architectures. Outputs a file with speedup data on every run
 * with the runs ordered from fastest to slowest.
 */
class EvaluateAcrossRuns {
public:
  /**
   * @brief Evaluate all runs with results in specified file path and
   * generate csv file with evaluation of results across runs
   * generated file starts with specified top text and includes
   * specified input parameters for each input as well as the optimized
   * implementation runtime for each run on each input
   * 
   * @param imp_results_file_path 
   * @param eval_across_runs_top_text 
   * @param eval_across_runs_in_params_show 
   */
  void operator()(
    const std::filesystem::path& imp_results_file_path,
    const std::vector<std::string>& eval_across_runs_top_text,
    const std::vector<std::string>& eval_across_runs_in_params_show) const;

private:
  
  /**
   * @brief Generate data for evaluating results across runs
   * 
   * @param run_results_by_name
   * @param eval_across_runs_in_params_show
   * @return EvalAcrossRunsData
   */
  EvalAcrossRunsData GenEvalAcrossRunsData(
    const std::map<std::string, RunResultsSpeedups>& run_results_by_name,
    const std::vector<std::string>& eval_across_runs_in_params_show) const;

  /**
   * @brief Get run names in order from fastest to slowest based on first
   * speedup header
   * 
   * @param eval_data
   */
  std::vector<std::string> OrderedRunNames(
    const EvalAcrossRunsData& eval_data) const;

  /**
   * @brief Function to get names of runs with results from implementation
   * results file path
   * 
   * @param imp_results_file_path 
   * @return std::vector<std::string> 
   */
  std::vector<std::string> GetRunNames(
    const std::filesystem::path& imp_results_file_path) const;
};

#endif //EVALUATE_ACROSS_RUNS_H_