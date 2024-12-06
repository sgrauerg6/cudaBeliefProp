/*
 * BpFileHandling.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLING_H_
#define BPFILEHANDLING_H_

#include <filesystem>
#include <algorithm>
#include "BpFileHandlingConsts.h"

//class for retrieve path of stereo set files for reading and for output
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
