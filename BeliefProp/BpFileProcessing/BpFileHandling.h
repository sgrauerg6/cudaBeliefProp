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
  //constructor takes stereo set name as input, which must match the directory name of the stereo set
  BpFileHandling(const std::string& stereo_set_name) : stereo_set_path_{getStereoSetsPath() / stereo_set_name}, num_out_disp_map_{1} { }

  //return path to reference image with valid extension if found, otherwise throw filesystem error
  std::filesystem::path getRefImagePath() const;

  //return path to test image with valid extension if found, otherwise throw filesystem error
  std::filesystem::path getTestImagePath() const;

  //return path to use for current output disparity and then increment (to support multiple computed output disparity maps)
  const std::filesystem::path getCurrentOutputDisparityFilePathAndIncrement() {
    return stereo_set_path_ / (std::string(bp_file_handling::OUT_DISP_IMAGE_NAME_BASE) + std::to_string(num_out_disp_map_++) + ".pgm");
  }

  //get file path to ground truth disparity map
  const std::filesystem::path getGroundTruthDisparityFilePath() const {
    return stereo_set_path_ / (std::string(bp_file_handling::GROUND_TRUTH_DISP_FILE));
  }

private:
  //retrieve path of stereo images to process using BELIEF_PROP_DIRECTORY_NAME and STEREO_SETS_DIRECTORY_NAME
  //constants
  std::filesystem::path getStereoSetsPath() const;

  const std::filesystem::path stereo_set_path_;
  unsigned int num_out_disp_map_;
};

#endif /* BPFILEHANDLING_H_ */
