/*
 * BpSettings.h
 *
 *  Created on: Nov 18, 2024
 *      Author: scott
 */

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <string>
#include "BpConstsAndParams/BpStereoParameters.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"

#ifndef BP_SETTINGS_H
#define BP_SETTINGS_H

//parameters type requires AsRunData() function to return the parameters as a
//RunData object
template <typename T>
concept Params_t =
  requires(T t) {
    { t.AsRunData() } -> std::same_as<RunData>;
  };

namespace beliefprop {

//default values for BP settings
//number of scales/levels in the pyramid to run BP
constexpr unsigned int kDefaultLevelsBp{5};

//number of BP iterations at each scale/level
constexpr unsigned int kDefaultItersBp{7};

//amount to smooth the input images
constexpr float kDefaultSigmaBp{0.0};

//weighing of data cost
constexpr float kDefaultLambdaBp{0.1};

//truncation of data cost
constexpr float kDefaultDataKBp{15.0};

//structure to store the settings for the number of levels and iterations
struct BpSettings
{
  //initally set to default values
  unsigned int num_levels{kDefaultLevelsBp};
  unsigned int num_iterations{kDefaultItersBp};
  float smoothing_sigma{kDefaultSigmaBp};
  float lambda_bp{kDefaultLambdaBp};
  float data_k_bp{kDefaultDataKBp};
  //discontinuity cost cap set to infinity by default but is
  //expected to be dependent on number of disparity values and set when
  //number of disparity values is set
  float disc_k_bp{bp_consts::kInfBp};
  //number of disparity values must be set for each stereo set
  unsigned int num_disp_vals{0};

  //retrieve bp settings as RunData object containing description headers with corresponding values
  //for each setting
  RunData AsRunData() const {
    RunData curr_run_data;
    curr_run_data.AddDataWHeader("Num Possible Disparity Values", num_disp_vals);
    curr_run_data.AddDataWHeader("Num BP Levels", num_levels);
    curr_run_data.AddDataWHeader("Num BP Iterations", num_iterations);
    curr_run_data.AddDataWHeader("DISC_K_BP", (double)disc_k_bp);
    curr_run_data.AddDataWHeader("DataKBp", (double)data_k_bp);
    curr_run_data.AddDataWHeader("LambdaBp", (double)lambda_bp);
    curr_run_data.AddDataWHeader("SigmaBp", (double)smoothing_sigma);

    return curr_run_data;
  }
};

}

#endif //BP_SETTINGS_H