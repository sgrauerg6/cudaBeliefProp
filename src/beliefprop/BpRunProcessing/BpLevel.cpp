/*
 * BpLevel.cpp
 *
 *  Created on: Nov 12, 2024
 *      Author: scott
 */

#include "BpLevel.h"

namespace beliefprop {

BpLevel::BpLevel(const std::array<unsigned int, 2>& width_height, std::size_t offset_into_arrays, unsigned int level_num,
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
  
BpLevel::BpLevel(const std::array<unsigned int, 2>& width_height, std::size_t offset_into_arrays, unsigned int level_num,
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

unsigned int BpLevel::CheckerboardWidthTargetDevice(unsigned int width_level) const {
  return (unsigned int)std::ceil(((float)width_level) / 2.0f);
}

unsigned int BpLevel::PaddedCheckerboardWidth(unsigned int checkerboard_width) const
{
  //add "padding" to checkerboard width if necessary for alignment
  return ((checkerboard_width % level_properties_.num_data_align_width_) == 0) ?
         checkerboard_width :
         (checkerboard_width + (level_properties_.num_data_align_width_ - (checkerboard_width % level_properties_.num_data_align_width_)));
}

}
