/*
 * EvaluateBPImpResults.h
 *
 *  Created on: March 2, 2024
 *      Author: scott
 * 
 *  Class to evaluate belief propagation implementation results.
 */

#ifndef EVALUATE_BP_IMP_RESULTS_H_
#define EVALUATE_BP_IMP_RESULTS_H_

#include "RunEval/EvaluateImpResults.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include <filesystem>
#include <vector>
#include <string>

//class with operator function to evaluate implementation runs
class EvaluateBPImpResults final : public EvaluateImpResults {
private:
  //retrieve path of belief propagation implementation results
  std::filesystem::path GetImpResultsPath() const override;

  //get text at top of results summary file with each string_view in the vector corresponding to a separate line
  std::vector<std::string> GetCombResultsTopText() const override {
    return {{"Stereo Processing using optimized CUDA and optimized CPU belief propagation implementations"},
            {"Code available at https://github.com/sgrauerg6/cudaBeliefProp"},
            {"All stereo sets used in evaluation are from (or adapted from) Middlebury stereo datasets at https://vision.middlebury.edu/stereo/data/"},
            {"\"tsukubaSetHalfSize: tsukubaSet with half the height, width, and disparity count of tsukubaSet\""},
            {"conesFullSizeCropped: 900 x 750 region in center of the reference and test cones stereo set images"},
            {"Results shown in this comparison for each run are for total runtime including any time for data transfer between device and host"}};
  }

  //input parameters that are showed in results summary with runtimes
  std::vector<std::string> GetInputParamsShow() const override {
    return {std::string(beliefprop::kStereoSetHeader), std::string(run_eval::kDatatypeHeader), std::string(beliefprop::kImageWidthHeader),
            std::string(beliefprop::kImageHeightHeader), std::string(beliefprop::kNumDispValsHeader),
            std::string(run_eval::kLoopItersTemplatedHeader)};
  }
};

#endif //EVALUATE_BP_IMP_RESULTS_H_