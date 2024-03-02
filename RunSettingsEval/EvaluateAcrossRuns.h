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
  //evaluate all runs with results in specified file path
  //generate csv file with evaluation of results across runs
  //generated file starts with specified top text and includes
  //specified input parameters for each input as well as the optimized
  //implementation runtime for each run on each input
  void operator()(const std::filesystem::path& impResultsFilePath,
    const std::vector<std::string>& evalAcrossRunsTopText,
    const std::vector<std::string>& evalAcrossRunsInParamsShow) const;

private:
  //get mapping of headers to data in csv file for run results and speedups
  //assumed that there are no commas in data since it is used as delimiter between data
  //first output is headers in order, second output is mapping of headers to results
  std::pair<std::vector<std::string>, std::map<std::string, std::vector<std::string>>> getHeaderToDataInCsvFile(
    const std::filesystem::path& csvFilePath) const;
};

#endif //EVALUATE_ACROSS_RUNS_H_