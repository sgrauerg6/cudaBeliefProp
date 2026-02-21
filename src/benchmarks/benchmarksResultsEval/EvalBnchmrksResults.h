/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file EvalBnchmrksResults.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef EVAL_BNCHMRKS_RESULTS_H
#define EVAL_BNCHMRKS_RESULTS_H

#include "RunEval/EvaluateImpResults.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "BnchmrksEvaluationInputs.h"
#include <filesystem>
#include <vector>
#include <string>

/**
 * @brief Child class of EvaluateImpResults that defines member functions for
 * benchmarks evaluation which override pure virtual functions in
 * parent class
 */
class EvalBnchmrksResults final : public EvaluateImpResults {
public:
  /**
   * @brief Constructor for initialization and to set name of directory
   * with results
   */
  EvalBnchmrksResults(const std::string& results_dir_name) : 
    EvaluateImpResults(results_dir_name) {}

private:
  /**
   * @brief Get text at top of evaluation across runs file with each
   * string in the vector corresponding to a separate line
   * 
   * @return Vector of strings corresponding to text to show at top of
   * evaluation across architectures
   */
  std::vector<std::string> GetCombResultsTopText() const override {
    return {{"Benchmark Processing using optimized CUDA and optimized CPU belief propagation implementations"},
            {"Code available at https://github.com/sgrauerg6/cudaBeliefProp"},
            {"Results shown in this comparison for each run are for total runtime including any time for data transfer between device and host"}};
  }

  /**
   * @brief Input parameters that are shown in evaluation across runs with
   * runtimes
   * 
   * @return Benchmark input parameters to show in evaluation across runtimes
   */
  std::vector<std::string> GetInputParamsShow() const override {
    return {
      std::string(run_eval::kDatatypeHeader),
      std::string(benchmarks::kMatWidthHeader),
      std::string(benchmarks::kMatHeightHeader)};
  }
};

#endif //EVAL_BNCHMRKS_RESULTS_H