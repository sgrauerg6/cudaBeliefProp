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
 * @file BpLevel.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "BpLevel.h"

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

