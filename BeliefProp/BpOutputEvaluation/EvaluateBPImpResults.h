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

#include "RunSettingsEval/EvaluateImpResults.h"
#include "RunSettingsEval/RunEvalConstsEnums.h"
#include "BpConstsAndParams/BpConsts.h"
#include <filesystem>
#include <vector>
#include <string>

//class with operator function to evaluate implementation runs
class EvaluateBPImpResults : public EvaluateImpResults {
private:
  //retrieve path of belief propagation implementation results
  std::filesystem::path getImpResultsPath() const override;

  //get text at top of results summary file with each string_view in the vector corresponding to a separate line
  std::vector<std::string> getCombResultsTopText() const override {
    return {{"Stereo Processing using optimized CUDA and optimized CPU belief propagation implementations"},
            {"Code available at https://github.com/sgrauerg6/cudaBeliefProp"},
            {"All stereo sets used in evaluation are from (or adapted from) Middlebury stereo datasets at https://vision.middlebury.edu/stereo/data/"},
            {"\"tsukubaSetHalfSize: tsukubaSet with half the height, width, and disparity count of tsukubaSet\""},
            {"conesFullSizeCropped: 900 x 750 region in center of the reference and test cones stereo set images"},
            {"Results shown in this comparison for each run are for total runtime including any time for data transfer between device and host"}};
  }

  //input parameters that are showed in results summary with runtimes
  std::vector<std::string> getInputParamsShow() const override {
    return {std::string(belief_prop::STEREO_SET_HEADER), std::string(run_eval::DATATYPE_HEADER), std::string(belief_prop::IMAGE_WIDTH_HEADER),
            std::string(belief_prop::IMAGE_HEIGHT_HEADER), std::string(belief_prop::NUM_DISP_VALS_HEADER),
            std::string(run_eval::LOOP_ITERS_TEMPLATED_HEADER)};
  }
};

#endif //EVALUATE_BP_IMP_RESULTS_H_