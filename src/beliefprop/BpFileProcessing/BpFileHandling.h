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
 * @file BpFileHandling.h
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BPFILEHANDLING_H_
#define BPFILEHANDLING_H_

#include <filesystem>
#include <algorithm>
#include "BpFileHandlingConsts.h"

/**
 * @brief Class to retrieve path of stereo set files for reading and for output
 * 
 */
class BpFileHandling {
public:
  /**
   * @brief Constructor takes stereo set name as input, which must match the directory name of the stereo set
   * 
   * @param stereo_set_name 
   */
  BpFileHandling(const std::string& stereo_set_name) : 
    stereo_set_path_{StereoSetsPath() / stereo_set_name}, num_out_disp_map_{1} { }

  /**
   * @brief Return path to reference image with valid extension if found, otherwise throw filesystem error
   * 
   * @return std::filesystem::path 
   */
  std::filesystem::path RefImagePath() const;

  /**
   * @brief Return path to test image with valid extension if found, otherwise throw filesystem error
   * 
   * @return std::filesystem::path 
   */
  std::filesystem::path TestImagePath() const;

  /**
   * @brief Return path to use for current output disparity and then increment (to support multiple computed output disparity maps)
   * 
   * @return const std::filesystem::path 
   */
  const std::filesystem::path GetCurrentOutputDisparityFilePathAndIncrement() {
    return stereo_set_path_ / 
      (std::string(bp_file_handling::kOutDispImageNameBase) + std::to_string(num_out_disp_map_++) + ".pgm");
  }

  /**
   * @brief Get file path to ground truth disparity map
   * 
   * @return const std::filesystem::path 
   */
  const std::filesystem::path GroundTruthDisparityFilePath() const {
    return stereo_set_path_ / (std::string(bp_file_handling::kGroundTruthDispFile));
  }

private:
  /**
   * @brief Retrieve path of stereo images to process using
   * kBeliefPropDirectoryName and kStereoSetsDirectoryName constants
   * 
   * @return std::filesystem::path 
   */
  std::filesystem::path StereoSetsPath() const;

  const std::filesystem::path stereo_set_path_;
  unsigned int num_out_disp_map_;
};

#endif /* BPFILEHANDLING_H_ */
