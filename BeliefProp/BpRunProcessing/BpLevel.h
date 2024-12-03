/*
 * BpLevel.h
 *
 *  Created on: Nov 18, 2024
 *      Author: scott
 */

#include <array>
#include <cmath>
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunTypeConstraints.h"

#ifndef BP_LEVEL_H_
#define BP_LEVEL_H_

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
  std::size_t offset_into_arrays_;
};

//class to store and retrieve properties of a bp processing level
class BpLevel
{
public:
  BpLevel(const std::array<unsigned int, 2>& width_height, std::size_t offset_into_arrays, unsigned int level_num,
    run_environment::AccSetting acc_setting);
  
  BpLevel(const std::array<unsigned int, 2>& width_height, std::size_t offset_into_arrays, unsigned int level_num,
    unsigned int bytes_align_memory, unsigned int num_data_align_width, unsigned int div_padded_ch_board_w_align);

  //get bp level properties for next (higher) level in hierarchy that processes data with half width/height of current level
  template <RunData_t T>
  beliefprop::BpLevel NextBpLevel(unsigned int num_disparity_values) const;

  //get the amount of data in each BP array (data cost/messages for each checkerboard) at the current level
  //with the specified number of possible disparity values
  template <RunData_t T>
  std::size_t NumDataInBpArrays(unsigned int num_disparity_values) const;

  unsigned int CheckerboardWidthTargetDevice(unsigned int width_level) const;

  unsigned int PaddedCheckerboardWidth(unsigned int checkerboard_width) const;

  template <RunData_t T>
  std::size_t NumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& width_height_level,
      unsigned int num_possible_disparities) const;

  template <RunData_t T, run_environment::AccSetting ACCELERATION>
  static std::size_t TotalDataForAlignedMemoryAllLevels(const std::array<unsigned int, 2>& width_height_bottom_level,
    unsigned int num_possible_disparities, unsigned int num_levels);
  
  //return level properties as const reference to avoid copying and not allow it to be modified
  const BpLevelProperties& LevelProperties() const { return level_properties_; }

private:
  BpLevelProperties level_properties_;
};

//get bp level properties for next (higher) level in hierarchy that processes data with half width/height of current level
template <RunData_t T>
beliefprop::BpLevel BpLevel::NextBpLevel(unsigned int num_disparity_values) const {
  const std::size_t offset_next_level = level_properties_.offset_into_arrays_ + NumDataInBpArrays<T>(num_disparity_values);
  return BpLevel({(unsigned int)std::ceil((float)level_properties_.width_level_ / 2.0f),
    (unsigned int)std::ceil((float)level_properties_.height_level_ / 2.0f)},
    offset_next_level, (level_properties_.level_num_ + 1), level_properties_.bytes_align_memory_,
    level_properties_.num_data_align_width_, level_properties_.div_padded_checkerboard_w_align_);
}

//get the amount of data in each BP array (data cost/messages for each checkerboard) at the current level
//with the specified number of possible disparity values
template <RunData_t T>
std::size_t BpLevel::NumDataInBpArrays(unsigned int num_disparity_values) const {
  return NumDataForAlignedMemoryAtLevel<T>({level_properties_.width_level_, level_properties_.height_level_}, num_disparity_values);
}

template <RunData_t T>
std::size_t BpLevel::NumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& width_height_level,
    unsigned int num_possible_disparities) const
{
  const std::size_t numDataAtLevel = (std::size_t)PaddedCheckerboardWidth(CheckerboardWidthTargetDevice(width_height_level[0])) *
    ((std::size_t)width_height_level[1]) * (std::size_t)num_possible_disparities;
  std::size_t numBytesAtLevel = numDataAtLevel * sizeof(T);

  if ((numBytesAtLevel % level_properties_.bytes_align_memory_) == 0) {
    return numDataAtLevel;
  }
  else {
    numBytesAtLevel += (level_properties_.bytes_align_memory_ - (numBytesAtLevel % level_properties_.bytes_align_memory_));
    return (numBytesAtLevel / sizeof(T));
  }
}

template <RunData_t T, run_environment::AccSetting ACCELERATION>
std::size_t BpLevel::TotalDataForAlignedMemoryAllLevels(const std::array<unsigned int, 2>& width_height_bottom_level,
  unsigned int num_possible_disparities, unsigned int num_levels)
{
  beliefprop::BpLevel curr_level_properties(width_height_bottom_level, 0, 0, ACCELERATION);
  std::size_t total_data = curr_level_properties.NumDataInBpArrays<T>(num_possible_disparities);
  for (unsigned int curr_level = 1; curr_level < num_levels; curr_level++) {
    curr_level_properties = curr_level_properties.NextBpLevel<T>(num_possible_disparities);
    total_data += curr_level_properties.NumDataInBpArrays<T>(num_possible_disparities);
  }

  return total_data;
}

}

#endif //BP_LEVEL_H_