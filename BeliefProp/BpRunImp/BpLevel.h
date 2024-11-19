/*
 * BpLevel.h
 *
 *  Created on: Nov 18, 2024
 *      Author: scott
 */

#include "BpConstsAndParams/BpStereoParameters.h"
#include "BpConstsAndParams/BpTypeConstraints.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"

namespace beliefprop {

//POD struct to store bp level data
//struct can be passed to __global__ CUDAs kernel so needs to take restrictions of what's allowed for
//passing data from the host to a CUDA kernel into account
struct BpLevelProperties {
  unsigned int width_level_;
  unsigned int height_level_;
  unsigned int bytes_align_memory_;
  unsigned int num_data_align_width_;
  unsigned int width_checkerboard_level_;
  unsigned int padded_width_checkerboard_level_;
  unsigned int level_num_;
  //avx512 requires data to be aligned on 64 bytes (16 float values); otherwise data set to be
  //aligned on 32 bytes (8 float values)
  unsigned int div_padded_checkerboard_w_align_;
  unsigned long offset_into_arrays_;
};

//class to store and retrieve properties of a bp processing level
class BpLevel
{
public:
  BpLevel(const std::array<unsigned int, 2>& width_height, unsigned long offset_into_arrays, unsigned int level_num,
    run_environment::AccSetting acc_setting)
  {
    level_properties_.width_level_ = width_height[0];
    level_properties_.height_level_ = width_height[1];
    level_properties_.bytes_align_memory_ = run_environment::GetBytesAlignMemory(acc_setting);
    level_properties_.num_data_align_width_ = run_environment::GetNumDataAlignWidth(acc_setting);
    level_properties_.width_checkerboard_level_ = CheckerboardWidthTargetDevice(level_properties_.width_level_);
    level_properties_.padded_width_checkerboard_level_ = PaddedCheckerboardWidth(level_properties_.width_checkerboard_level_);
    level_properties_.offset_into_arrays_=  offset_into_arrays;
    level_properties_.level_num_ = level_num;
    level_properties_.div_padded_checkerboard_w_align_ = (acc_setting == run_environment::AccSetting::kAVX512) ? 16u : 8u;
  }
  
  BpLevel(const std::array<unsigned int, 2>& width_height, unsigned long offset_into_arrays, unsigned int level_num,
    unsigned int bytes_align_memory, unsigned int num_data_align_width, unsigned int div_padded_ch_board_w_align)
  {
    level_properties_.width_level_ = width_height[0];
    level_properties_.height_level_ = width_height[1];
    level_properties_.bytes_align_memory_ = bytes_align_memory;
    level_properties_.num_data_align_width_ = num_data_align_width;
    level_properties_.width_checkerboard_level_ = CheckerboardWidthTargetDevice(level_properties_.width_level_);
    level_properties_.padded_width_checkerboard_level_ = PaddedCheckerboardWidth(level_properties_.width_checkerboard_level_);
    level_properties_.offset_into_arrays_ = offset_into_arrays;
    level_properties_.level_num_ = level_num;
    level_properties_.div_padded_checkerboard_w_align_ = div_padded_ch_board_w_align;
  }

  //get bp level properties for next (higher) level in hierarchy that processes data with half width/height of current level
  template <RunData_t T>
  beliefprop::BpLevel NextBpLevel(unsigned int num_disparity_values) const {
    const auto offset_next_level = level_properties_.offset_into_arrays_ + NumDataInBpArrays<T>(num_disparity_values);
    return BpLevel({(unsigned int)ceil((float)level_properties_.width_level_ / 2.0f),
      (unsigned int)ceil((float)level_properties_.height_level_ / 2.0f)},
      offset_next_level, (level_properties_.level_num_ + 1), level_properties_.bytes_align_memory_,
      level_properties_.num_data_align_width_, level_properties_.div_padded_checkerboard_w_align_);
  }

  //get the amount of data in each BP array (data cost/messages for each checkerboard) at the current level
  //with the specified number of possible disparity values
  template <RunData_t T>
  unsigned int NumDataInBpArrays(unsigned int num_disparity_values) const {
    return NumDataForAlignedMemoryAtLevel<T>({level_properties_.width_level_, level_properties_.height_level_}, num_disparity_values);
  }

  unsigned int CheckerboardWidthTargetDevice(unsigned int width_level) const {
    return (unsigned int)std::ceil(((float)width_level) / 2.0f);
  }

  unsigned int PaddedCheckerboardWidth(unsigned int checkerboard_width) const
  {
    //add "padding" to checkerboard width if necessary for alignment
    return ((checkerboard_width % level_properties_.num_data_align_width_) == 0) ?
           checkerboard_width :
           (checkerboard_width + (level_properties_.num_data_align_width_ - (checkerboard_width % level_properties_.num_data_align_width_)));
  }

  template <RunData_t T>
  unsigned long NumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& width_height_level,
      unsigned int num_possible_disparities) const
  {
    const unsigned long numDataAtLevel = (unsigned long)PaddedCheckerboardWidth(CheckerboardWidthTargetDevice(width_height_level[0])) *
      ((unsigned long)width_height_level[1]) * (unsigned long)num_possible_disparities;
    unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

    if ((numBytesAtLevel % level_properties_.bytes_align_memory_) == 0) {
      return numDataAtLevel;
    }
    else {
      numBytesAtLevel += (level_properties_.bytes_align_memory_ - (numBytesAtLevel % level_properties_.bytes_align_memory_));
      return (numBytesAtLevel / sizeof(T));
    }
  }

  template <RunData_t T, run_environment::AccSetting ACCELERATION>
  static unsigned long TotalDataForAlignedMemoryAllLevels(const std::array<unsigned int, 2>& width_height_bottom_level,
    unsigned int num_possible_disparities, unsigned int num_levels)
  {
    beliefprop::BpLevel curr_level_properties(width_height_bottom_level, 0, 0, ACCELERATION);
    unsigned long total_data = curr_level_properties.NumDataInBpArrays<T>(num_possible_disparities);
    for (unsigned int curr_level = 1; curr_level < num_levels; curr_level++) {
      curr_level_properties = curr_level_properties.NextBpLevel<T>(num_possible_disparities);
      total_data += curr_level_properties.NumDataInBpArrays<T>(num_possible_disparities);
    }

    return total_data;
  }
  
  //return level properties as const reference to avoid copying and not allow it to be modified
  const BpLevelProperties& LevelProperties() const { return level_properties_; }

private:
  BpLevelProperties level_properties_;
};

}