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

#include <map>
#include <set>
#include <string>
#include <vector>
#include <filesystem>
#include <fstream>
#include <string>
#include <ranges>
#include <algorithm>
#include "RunEvalConstsEnums.h"

class EvaluateAcrossRuns {
public:
  //evaluate all runs with results in specified file path and
  //generate csv file with evaluation of results across runs
  //generated file starts with specified top text and includes
  //specified input parameters for each input as well as the optimized
  //implementation runtime for each run on each input
  void operator()(const std::filesystem::path& imp_results_file_path,
    const std::vector<std::string>& eval_across_runs_top_text,
    const std::vector<std::string>& eval_across_runs_in_params_show,
    const std::pair<std::vector<std::string>, size_t>& speedup_headers_w_ordering_idx) const;

private:
  //get mapping of headers to data in csv file for run results and speedups
  //assumed that there are no commas in data since it is used as delimiter between data
  //first output is headers in order, second output is mapping of headers to results
  std::map<std::string, std::vector<std::string>> HeaderToDataInCsvFile(
    const std::filesystem::path& csv_file_path) const;
};

#endif //EVALUATE_ACROSS_RUNS_H_