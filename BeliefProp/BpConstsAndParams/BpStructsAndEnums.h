/*
 * BpStructsAndEnums.h
 *
 *  Created on: Sep 22, 2019
 *      Author: scott
 */

#ifndef BPSTRUCTSANDENUMS_H_
#define BPSTRUCTSANDENUMS_H_

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <string>
#include "BpStereoParameters.h"
#include "BpTypeConstraints.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"

//parameters type requires AsRunData() function to return the parameters as a
//RunData object
template <typename T>
concept Params_t =
  requires(T t) {
    { t.AsRunData() } -> std::same_as<RunData>;
  };

namespace beliefprop {

//structure to store the settings for the number of levels and iterations
struct BpSettings
{
  //initally set to default values
  unsigned int num_levels{bp_params::kLevelsBp};
  unsigned int num_iterations{bp_params::kItersBp};
  float smoothing_sigma{bp_params::kSigmaBp};
  float lambda_bp{bp_params::kLambdaBp};
  float data_k_bp{bp_params::kDataKBp};
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
    curr_run_data.AddDataWHeader("kDataKBp", (double)data_k_bp);
    curr_run_data.AddDataWHeader("kLambdaBp", (double)lambda_bp);
    curr_run_data.AddDataWHeader("kSigmaBp", (double)smoothing_sigma);

    return curr_run_data;
  }

  //declare friend function to output bp settings to stream
  friend std::ostream& operator<<(std::ostream& results_stream, const BpSettings& bp_settings);
};

//function to output bp settings to stream
inline std::ostream& operator<<(std::ostream& results_stream, const BpSettings& bp_settings) {
  //get settings as RunData object and then use overloaded << operator for RunData
  results_stream << bp_settings.AsRunData();
  return results_stream;  
}

//used to define the two checkerboard "parts" that the image is divided into
enum class Checkerboard_Part {kCheckerboardPart0, kCheckerboardPart1 };
enum class Message_Arrays : unsigned int { 
  kMessagesUCheckerboard0, kMessagesDCheckerboard0, kMessagesLCheckerboard0, kMessagesRCheckerboard0,
  kMessagesUCheckerboard1, kMessagesDCheckerboard1, kMessagesLCheckerboard1, kMessagesRCheckerboard1 };
enum class MessageComp { kUMessage, kDMessage, kLMessage, kRMessage };

//each checkerboard messages element corresponds to an array of message values that can be mapped to
//a unique value within Message_Arrays enum
//could use a map/unordered map to map Message_Arrays enum to corresponding message array but using array structure is likely faster
template <RunData_ptr T>
using CheckerboardMessages = std::array<T, 8>;

//belief propagation checkerboard messages and data costs must be pointers to a bp data type
//define alias for two-element array with data costs for each bp processing checkerboard
template <RunData_ptr T>
using DataCostsCheckerboards = std::array<T, 2>;

//enum corresponding to each kernel in belief propagation that can be run in parallel
enum class BpKernel : unsigned int { 
  kBlurImages,
  kDataCostsAtLevel,
  kInitMessageVals,
  kBpAtLevel,
  kCopyAtLevel,
  kOutputDisp };
constexpr unsigned int kNumKernels{6};

};

#endif /* BPSTRUCTSANDENUMS_H_ */
