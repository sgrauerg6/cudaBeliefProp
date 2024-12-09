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
 * @file EvaluateBpImpResults.h
 * @author Scott Grauer-Gray
 * @brief Class to evaluate belief propagation implementation results.
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef EVALUATE_BP_IMP_RESULTS_H_
#define EVALUATE_BP_IMP_RESULTS_H_

#include "RunEval/EvaluateImpResults.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include <filesystem>
#include <vector>
#include <string>

/**
 * @brief Child class of EvaluateImpResults that defines member functions for
 * belief propagation evaluation which override pure virtual functions in parent
 * class
 * 
 */
class EvaluateBpImpResults final : public EvaluateImpResults {
private:
  /**
   * @brief Retrieve path of belief propagation implementation results
   * 
   * @return std::filesystem::path 
   */
  std::filesystem::path GetImpResultsPath() const override;

  /**
   * @brief Get text at top of results summary file with each string_view
   * in the vector corresponding to a separate line
   * 
   * @return std::vector<std::string> 
   */
  std::vector<std::string> GetCombResultsTopText() const override {
    return {{"Stereo Processing using optimized CUDA and optimized CPU belief propagation implementations"},
            {"Code available at https://github.com/sgrauerg6/cudaBeliefProp"},
            {"All stereo sets used in evaluation are from (or adapted from) Middlebury stereo datasets at https://vision.middlebury.edu/stereo/data/"},
            {"\"tsukubaSetHalfSize: tsukubaSet with half the height, width, and disparity count of tsukubaSet\""},
            {"conesFullSizeCropped: 900 x 750 region in center of the reference and test cones stereo set images"},
            {"Results shown in this comparison for each run are for total runtime including any time for data transfer between device and host"}};
  }

  /**
   * @brief Input parameters that are showed in results summary with runtimes
   * 
   * @return std::vector<std::string> 
   */
  std::vector<std::string> GetInputParamsShow() const override {
    return {std::string(beliefprop::kStereoSetHeader), std::string(run_eval::kDatatypeHeader), std::string(beliefprop::kImageWidthHeader),
            std::string(beliefprop::kImageHeightHeader), std::string(beliefprop::kNumDispValsHeader),
            std::string(run_eval::kLoopItersTemplatedHeader)};
  }
};

#endif //EVALUATE_BP_IMP_RESULTS_H_