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
struct BPsettings
{
  //initally set to default values
  unsigned int num_levels_{bp_params::kLevelsBp};
  unsigned int num_iterations_{bp_params::kItersBp};
  float smoothing_sigma_{bp_params::kSigmaBp};
  float lambda_bp_{bp_params::kLambdaBp};
  float data_k_bp_{bp_params::kDataKBp};
  //discontinuity cost cap set to infinity by default but is
  //expected to be dependent on number of disparity values and set when
  //number of disparity values is set
  float disc_k_bp_{bp_consts::kInfBp};
  //number of disparity values must be set for each stereo set
  unsigned int num_disp_vals_{0};

  //retrieve bp settings as RunData object containing description headers with corresponding values
  //for each setting
  RunData AsRunData() const {
    RunData curr_run_data;
    curr_run_data.AddDataWHeader("Num Possible Disparity Values", num_disp_vals_);
    curr_run_data.AddDataWHeader("Num BP Levels", num_levels_);
    curr_run_data.AddDataWHeader("Num BP Iterations", num_iterations_);
    curr_run_data.AddDataWHeader("DISC_K_BP", (double)disc_k_bp_);
    curr_run_data.AddDataWHeader("kDataKBp", (double)data_k_bp_);
    curr_run_data.AddDataWHeader("kLambdaBp", (double)lambda_bp_);
    curr_run_data.AddDataWHeader("kSigmaBp", (double)smoothing_sigma_);

    return curr_run_data;
  }

  //declare friend function to output bp settings to stream
  friend std::ostream& operator<<(std::ostream& results_stream, const BPsettings& bpSettings);
};

//function to output bp settings to stream
inline std::ostream& operator<<(std::ostream& results_stream, const BPsettings& bpSettings) {
  //get settings as RunData object and then use overloaded << operator for RunData
  results_stream << bpSettings.AsRunData();
  return results_stream;  
}

//structure to store properties of a bp processing level
struct LevelProperties
{
  LevelProperties(const std::array<unsigned int, 2>& widthHeight, unsigned long offsetIntoArrays, unsigned int levelNum,
    run_environment::AccSetting accSetting) :
    width_level_(widthHeight[0]), height_level_(widthHeight[1]),
    bytes_align_memory_(run_environment::getBytesAlignMemory(accSetting)),
    num_data_align_width_(run_environment::getNumDataAlignWidth(accSetting)),
    width_checkerboard_level_(CheckerboardWidthTargetDevice(width_level_)),
    padded_width_checkerboard_level_(PaddedCheckerboardWidth(width_checkerboard_level_)),
    offset_into_arrays_(offsetIntoArrays), level_num_(levelNum),
    div_padded_checkerboard_w_align_{(accSetting == run_environment::AccSetting::kAVX512) ? 16u : 8u} {}
  
  LevelProperties(const std::array<unsigned int, 2>& widthHeight, unsigned long offsetIntoArrays, unsigned int levelNum,
    unsigned int bytesAlignMemory, unsigned int numDataAlignWidth, unsigned int divPaddedChBoardWAlign) :
    width_level_(widthHeight[0]), height_level_(widthHeight[1]),
    bytes_align_memory_(bytesAlignMemory),
    num_data_align_width_(numDataAlignWidth),
    width_checkerboard_level_(CheckerboardWidthTargetDevice(width_level_)),
    padded_width_checkerboard_level_(PaddedCheckerboardWidth(width_checkerboard_level_)),
    offset_into_arrays_(offsetIntoArrays), level_num_(levelNum), div_padded_checkerboard_w_align_(divPaddedChBoardWAlign) {}

  //get bp level properties for next (higher) level in hierarchy that processes data with half width/height of current level
  template <RunData_t T>
  beliefprop::LevelProperties getNextLevelProperties(unsigned int num_disparity_values) const {
    const auto offset_next_level = offset_into_arrays_ + getNumDataInBpArrays<T>(num_disparity_values);
    return LevelProperties({(unsigned int)ceil((float)width_level_ / 2.0f), (unsigned int)ceil((float)height_level_ / 2.0f)},
      offset_next_level, (level_num_ + 1), bytes_align_memory_, num_data_align_width_, div_padded_checkerboard_w_align_);
  }

  //get the amount of data in each BP array (data cost/messages for each checkerboard) at the current level
  //with the specified number of possible disparity values
  template <RunData_t T>
  unsigned int getNumDataInBpArrays(unsigned int num_disparity_values) const {
    return getNumDataForAlignedMemoryAtLevel<T>({width_level_, height_level_}, num_disparity_values);
  }

  unsigned int CheckerboardWidthTargetDevice(unsigned int width_level) const {
    return (unsigned int)std::ceil(((float)width_level) / 2.0f);
  }

  unsigned int PaddedCheckerboardWidth(unsigned int checkerboardWidth) const
  {
    //add "padding" to checkerboard width if necessary for alignment
    return ((checkerboardWidth % num_data_align_width_) == 0) ?
           checkerboardWidth :
           (checkerboardWidth + (num_data_align_width_ - (checkerboardWidth % num_data_align_width_)));
  }

  template <RunData_t T>
  unsigned long getNumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& width_height_level,
      unsigned int num_possible_disparities) const
  {
    const unsigned long numDataAtLevel = (unsigned long)PaddedCheckerboardWidth(CheckerboardWidthTargetDevice(width_height_level[0])) *
      ((unsigned long)width_height_level[1]) * (unsigned long)num_possible_disparities;
    unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

    if ((numBytesAtLevel % bytes_align_memory_) == 0) {
      return numDataAtLevel;
    }
    else {
      numBytesAtLevel += (bytes_align_memory_ - (numBytesAtLevel % bytes_align_memory_));
      return (numBytesAtLevel / sizeof(T));
    }
  }

  template <RunData_t T, run_environment::AccSetting ACCELERATION>
  static unsigned long getTotalDataForAlignedMemoryAllLevels(const std::array<unsigned int, 2>& width_height_bottom_level,
    unsigned int num_possible_disparities, unsigned int num_levels)
  {
    beliefprop::LevelProperties curr_level_properties(width_height_bottom_level, 0, 0, ACCELERATION);
    unsigned long total_data = curr_level_properties.getNumDataInBpArrays<T>(num_possible_disparities);
    for (unsigned int curr_level = 1; curr_level < num_levels; curr_level++) {
      curr_level_properties = curr_level_properties.getNextLevelProperties<T>(num_possible_disparities);
      total_data += curr_level_properties.getNumDataInBpArrays<T>(num_possible_disparities);
    }

    return total_data;
  }

  unsigned int width_level_;
  unsigned int height_level_;
  unsigned int bytes_align_memory_;
  unsigned int num_data_align_width_;
  unsigned int width_checkerboard_level_;
  unsigned int padded_width_checkerboard_level_;
  unsigned long offset_into_arrays_;
  unsigned int level_num_;
  //avx512 requires data to be aligned on 64 bytes (16 float values); otherwise data set to be
  //aligned on 32 bytes (8 float values)
  unsigned int div_padded_checkerboard_w_align_;
};

//used to define the two checkerboard "parts" that the image is divided into
enum class Checkerboard_Part {kCheckerboardPart0, kCheckerboardPart1 };
enum class Message_Arrays : unsigned int { 
  kMessagesUCheckerboard0, kMessagesDCheckerboard0, kMessagesLCheckerboard0, kMessagesRCheckerboard0,
  kMessagesUCheckerboard1, kMessagesDCheckerboard1, kMessagesLCheckerboard1, kMessagesRCheckerboard1 };
enum class MessageComp { kUMessage, kDMessage, kLMessage, kRMessage };

//each checkerboard messages element corresponds to separate Message_Arrays enum that go from 0 to 7 (8 total)
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
