/*
 * BpSettings.h
 *
 *  Created on: Nov 18, 2024
 *      Author: scott
 */

#ifndef BP_SETTINGS_H_
#define BP_SETTINGS_H_

#include <vector>
#include <string>
#include <string_view>
#include "RunSettingsParams/InputSignature.h"
#include "RunEval/RunData.h"
#include "BpConstsEnumsAliases.h"

/**
 * @brief Parameters type requires AsRunData() function to return the parameters
 * as a RunData object
 * 
 * @tparam T 
 */
template <typename T>
concept Params_t =
  requires(T t) {
    { t.AsRunData() } -> std::same_as<RunData>;
  };

namespace beliefprop {

//constants for headers corresponding to belief propagation settings in evaluation
constexpr std::string_view kNumDispValsHeader{"Num Possible Disparity Values"};
constexpr std::string_view kNumBpLevelsHeader{"Num BP Levels",};
constexpr std::string_view kNumBpItersHeader{"Num BP Iterations"};
constexpr std::string_view kDiscCostCapHeader{"DISC_K_BP"};
constexpr std::string_view kDataCostCapHeader{"DataKBp"};
constexpr std::string_view kBpSettingsLambdaHeader{"LambdaBp"};
constexpr std::string_view kBpSettingsSigmaHeader{"SigmaBp"};

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

/**
 * @brief Structure to store the settings for the number of levels and iterations
 * 
 */
struct BpSettings
{
  //initally set to default values
  unsigned int num_levels{kDefaultLevelsBp};
  unsigned int num_iterations{kDefaultItersBp};
  float smoothing_sigma{kDefaultSigmaBp};
  float lambda_bp{kDefaultLambdaBp};
  float data_k_bp{kDefaultDataKBp};
  /**
   * @brief Discontinuity cost cap set to infinity by default but is
   * expected to be dependent on number of disparity values and set when
   * number of disparity values is set
   * 
   */
  float disc_k_bp{beliefprop::kInfBp};
  
  /**
   * @brief Number of disparity values must be set for each stereo set
   * 
   */
  unsigned int num_disp_vals{0};

  /**
   * @brief Retrieve bp settings as RunData object containing description headers
   * with corresponding values for each setting
   * 
   * @return RunData 
   */
  RunData AsRunData() const {
    RunData curr_run_data;
    curr_run_data.AddDataWHeader(std::string(kNumDispValsHeader), num_disp_vals);
    curr_run_data.AddDataWHeader(std::string(kNumBpLevelsHeader), num_levels);
    curr_run_data.AddDataWHeader(std::string(kNumBpItersHeader), num_iterations);
    curr_run_data.AddDataWHeader(std::string(kDiscCostCapHeader), (double)disc_k_bp);
    curr_run_data.AddDataWHeader(std::string(kDataCostCapHeader), (double)data_k_bp);
    curr_run_data.AddDataWHeader(std::string(kBpSettingsLambdaHeader), (double)lambda_bp);
    curr_run_data.AddDataWHeader(std::string(kBpSettingsSigmaHeader), (double)smoothing_sigma);

    return curr_run_data;
  }
};

}

#endif //BP_SETTINGS_H_