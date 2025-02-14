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
 * @file BpLevel.h
 * @author Scott Grauer-Gray
 * @brief Declares class to store and retrieve properties of a bp processing level
 * 
 * @copyright Copyright (c) 2024
 */

#include <array>
#include <cmath>
#include "RunSettingsParams/RunSettings.h"
#include "RunEval/RunTypeConstraints.h"

#ifndef BP_LEVEL_H_
#define BP_LEVEL_H_

namespace beliefprop {

/**
 * @brief POD struct to store bp level data.
 * Struct can be passed to global CUDAs kernel so needs to take restrictions of what's allowed for
 * passing data from the host to a CUDA kernel into account.
 */
struct BpLevelProperties {
  unsigned int width_level_;
  unsigned int height_level_;
  unsigned int bytes_align_memory_;
  unsigned int width_checkerboard_level_;
  unsigned int padded_width_checkerboard_level_;
  unsigned int level_num_;
  std::size_t offset_into_arrays_;
};

};

/**
 * @brief Class to store and retrieve properties of a bp processing level
 * including a data type specified as a class template parameter
 */
template <RunData_t T>
class BpLevel
{
public:
  /**
   * @brief Construct a new BpLevel object with specified height/wdith, offset
   * into data/message arrays, level number, and acceleration setting
   * 
   * @param width_height
   * @param offset_into_arrays
   * @param level_num
   * @param acc_setting
   */
  explicit BpLevel(
    const std::array<unsigned int, 2>& width_height,
    std::size_t offset_into_arrays,
    unsigned int level_num,
    run_environment::AccSetting acc_setting);
  
  /**
   * @brief Construct a new BpLevel object with specified height/wdith, offset
   * into data/message arrays, level number, and byte alignment for memory
   * 
   * @param width_height
   * @param offset_into_arrays
   * @param level_num
   * @param bytes_align_memory
   */
  explicit BpLevel(
    const std::array<unsigned int, 2>& width_height,
    std::size_t offset_into_arrays,
    unsigned int level_num,
    unsigned int bytes_align_memory);

  /**
   * @brief Get bp level properties for next (higher) level in hierarchy that
   * processes data with half width/height of current level
   * 
   * @param num_disparity_values 
   * @return BpLevel object for next level in hierarchy 
   */
  BpLevel NextBpLevel(unsigned int num_disparity_values) const;

  /**
   * @brief Get the amount of data in each BP array (data cost/messages for
   * each checkerboard) at the current level with the specified number of
   * possible disparity values
   * 
   * @param num_disparity_values 
   * @return Num data needed in each "belief propagation" array at level
   */
  std::size_t NumDataInBpArrays(unsigned int num_disparity_values) const;

  /**
   * @brief Get width of checkerboard to use in bp processing at level.<br>
   * Checkerboard width is half of the width of the level.
   * 
   * @param width_level
   * @return Width of checkerboard used in bp processing at level
   */
  unsigned int CheckerboardWidth(unsigned int width_level) const;

  /**
   * @brief Get width of padded checkerboard to use in bp processing at
   * level.<br>
   * Padding can add additional width to checkerboard so that reading/writing
   * data starting at x=0 in the checkerboard is aligned for processing using
   * target acceleration (SIMD, CUDA, etc.).
   * 
   * @param width_level
   * @return Width of checkerboard used in bp processing at level
   */
  unsigned int PaddedCheckerboardWidth(unsigned int checkerboard_width) const;

  /**
   * @brief Get count of total data needed for bp processing at level
   * 
   * @param width_height_level 
   * @param num_possible_disparities 
   * @return Count of total data needed for bp processing at level
   */
  std::size_t NumDataForAlignedMemoryAtLevel(
    const std::array<unsigned int, 2>& width_height_level,
    unsigned int num_possible_disparities) const;

  /**
   * @brief Static function to get count of total data needed for bp processing
   * at all levels
   * 
   * @param width_height_bottom_level 
   * @param num_possible_disparities 
   * @param num_levels 
   * @param acceleration 
   * @return Count of total data needed for bp processing
   */
  static std::size_t TotalDataForAlignedMemoryAllLevels(
    const std::array<unsigned int, 2>& width_height_bottom_level,
    unsigned int num_possible_disparities,
    unsigned int num_levels,
    run_environment::AccSetting acceleration);
  
  /**
   * @brief Return level properties as const reference to avoid copying
   * and not allow it to be modified
   * 
   * @return BpLevelProperties "POD" struct with level properties
   */
  const beliefprop::BpLevelProperties& LevelProperties() const {
    return level_properties_;
  }

private:
  beliefprop::BpLevelProperties level_properties_;
};

template <RunData_t T>
BpLevel<T>::BpLevel(
  const std::array<unsigned int, 2>& width_height,
  std::size_t offset_into_arrays,
  unsigned int level_num,
  run_environment::AccSetting acc_setting)
{
  level_properties_.width_level_ = width_height[0];
  level_properties_.height_level_ = width_height[1];
  level_properties_.level_num_ = level_num;
  level_properties_.offset_into_arrays_=  offset_into_arrays;
  level_properties_.bytes_align_memory_ = 
    run_environment::GetBytesAlignMemory(acc_setting);
  level_properties_.width_checkerboard_level_ =
    CheckerboardWidth(level_properties_.width_level_);
  level_properties_.padded_width_checkerboard_level_ =
    PaddedCheckerboardWidth(level_properties_.width_checkerboard_level_);
}


template <RunData_t T>
BpLevel<T>::BpLevel(
  const std::array<unsigned int, 2>& width_height,
  std::size_t offset_into_arrays,
  unsigned int level_num,
  unsigned int bytes_align_memory)
{
  level_properties_.width_level_ = width_height[0];
  level_properties_.height_level_ = width_height[1];
  level_properties_.level_num_ = level_num;
  level_properties_.offset_into_arrays_ = offset_into_arrays;
  level_properties_.bytes_align_memory_ = bytes_align_memory;
  level_properties_.width_checkerboard_level_ =
    CheckerboardWidth(level_properties_.width_level_);
  level_properties_.padded_width_checkerboard_level_ =
    PaddedCheckerboardWidth(level_properties_.width_checkerboard_level_);
}

//get bp level properties for next (higher) level in hierarchy that processes
//data with half width/height of current level
template <RunData_t T>
BpLevel<T> BpLevel<T>::NextBpLevel(unsigned int num_disparity_values) const
{
  const std::size_t offset_next_level =
    level_properties_.offset_into_arrays_ +
    NumDataInBpArrays(num_disparity_values);
  return BpLevel<T>(
    {(unsigned int)std::ceil((float)level_properties_.width_level_ / 2.0f),
     (unsigned int)std::ceil((float)level_properties_.height_level_ / 2.0f)},
    offset_next_level,
    (level_properties_.level_num_ + 1),
    level_properties_.bytes_align_memory_);
}

//get the amount of data in each BP array (data cost/messages for each
//checkerboard) at the current level with the specified number of possible
//disparity values
template <RunData_t T>
std::size_t BpLevel<T>::NumDataInBpArrays(
  unsigned int num_disparity_values) const
{
  return NumDataForAlignedMemoryAtLevel(
    {level_properties_.width_level_, level_properties_.height_level_},
    num_disparity_values);
}

template <RunData_t T>
std::size_t BpLevel<T>::NumDataForAlignedMemoryAtLevel(
  const std::array<unsigned int, 2>& width_height_level,
  unsigned int num_possible_disparities) const
{
  const std::size_t numDataAtLevel =
    (std::size_t)PaddedCheckerboardWidth(CheckerboardWidth(width_height_level[0])) *
    ((std::size_t)width_height_level[1]) *
    (std::size_t)num_possible_disparities;
  std::size_t numBytesAtLevel = numDataAtLevel * sizeof(T);

  if ((numBytesAtLevel % level_properties_.bytes_align_memory_) == 0) {
    return numDataAtLevel;
  }
  else {
    numBytesAtLevel +=
      (level_properties_.bytes_align_memory_ -
        (numBytesAtLevel % level_properties_.bytes_align_memory_));
    return (numBytesAtLevel / sizeof(T));
  }
}

template <RunData_t T>
unsigned int BpLevel<T>::CheckerboardWidth(
  unsigned int width_level) const
{
  return (unsigned int)std::ceil(((float)width_level) / 2.0f);
}

template <RunData_t T>
unsigned int BpLevel<T>::PaddedCheckerboardWidth(
  unsigned int checkerboard_width) const
{
  size_t num_data_align_width = level_properties_.bytes_align_memory_ / sizeof(T); 
  //add "padding" to checkerboard width if necessary for alignment
  return ((checkerboard_width % num_data_align_width) == 0) ?
         checkerboard_width :
         (checkerboard_width +
           (num_data_align_width - (checkerboard_width % num_data_align_width)));
}

//static function
template <RunData_t T>
std::size_t BpLevel<T>::TotalDataForAlignedMemoryAllLevels(
  const std::array<unsigned int, 2>& width_height_bottom_level,
  unsigned int num_possible_disparities,
  unsigned int num_levels,
  run_environment::AccSetting acceleration)
{
  BpLevel<T> curr_level_properties(width_height_bottom_level, 0, 0, acceleration);
  std::size_t total_data =
    curr_level_properties.NumDataInBpArrays(num_possible_disparities);
  for (unsigned int curr_level = 1; curr_level < num_levels; curr_level++)
  {
    curr_level_properties = curr_level_properties.NextBpLevel(num_possible_disparities);
    total_data += curr_level_properties.NumDataInBpArrays(num_possible_disparities);
  }

  return total_data;
}

#endif //BP_LEVEL_H_