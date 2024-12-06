/*
 * EvaluateAcrossRuns.h
 *
 *  Created on: Feb 25, 2024
 *      Author: scott
 * 
 *  Class for evaluating results across multiple runs, potentially on different architectures.
 */

#ifndef EVALUATE_ACROSS_RUNS_H_
#define EVALUATE_ACROSS_RUNS_H_

#include <string>
#include <vector>
#include <filesystem>
#include "RunEvalConstsEnums.h"

class EvaluateAcrossRuns {
public:
  //evaluate all runs with results in specified file path and
  //generate csv file with evaluation of results across runs
  //generated file starts with specified top text and includes
  //specified input parameters for each input as well as the optimized
  //implementation runtime for each run on each input
  void operator()(
    const std::filesystem::path& imp_results_file_path,
    const std::vector<std::string>& eval_across_runs_top_text,
    const std::vector<std::string>& eval_across_runs_in_params_show,
    const std::vector<std::string>& speedup_headers) const;

private:
  //function to get names of runs with results from implementation results file path
  std::vector<std::string> GetRunNames(
    const std::filesystem::path& imp_results_file_path) const;
};

#endif //EVALUATE_ACROSS_RUNS_H_