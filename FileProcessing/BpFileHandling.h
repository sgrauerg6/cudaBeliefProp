/*
 * BpFileHandling.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLING_H_
#define BPFILEHANDLING_H_

#include <filesystem>
#include "BpFileHandlingConsts.h"
#include <algorithm>

class BpFileHandling {
public:

  //constructor takes stereo set name as input, which must match the directory name of the stereo set
  BpFileHandling(const std::string& stereo_set_name) : num_out_disp_map_(1) {
    stereo_set_path_ = getStereoSetsPath() / stereo_set_name;
  }

  //virtual ~BpFileHandling();

  std::filesystem::path getStereoSetsPath() const
  {
    std::filesystem::path currentPath = std::filesystem::current_path();

    while (true)
    {
      //create directory iterator corresponding to current path
      std::filesystem::directory_iterator dirIt = std::filesystem::directory_iterator(currentPath);

      //check if any of the directories in the current path correspond to the stereo sets directory;
      //if so return iterator to directory; otherwise return iterator to end indicating that directory not
      //found in current path
      std::filesystem::directory_iterator it = std::find_if(std::filesystem::begin(dirIt), std::filesystem::end(dirIt), 
        [](const auto &p) { return p.path().stem() == bp_file_handling::STEREO_SETS_DIRECTORY_NAME; });

      //check if return from find_if at iterator end and therefore didn't find stereo sets directory;
      //if that's the case continue to outer directory
      //for now assuming stereo sets directory exists in some outer directory and program won't work without it
      if (it == std::filesystem::end(dirIt))
      {
        //if current path same as parent path, then can't continue and throw error
        if (currentPath == currentPath.parent_path()) {
          throw std::filesystem::filesystem_error("Stereo set directory not found", std::error_code());
        }

        currentPath = currentPath.parent_path();
      }
      else {
        std::filesystem::path stereoSetPath = it->path();
        return stereoSetPath;
      }
    }

    return std::filesystem::path();
  }

  //return path to reference image with valid extension if found, otherwise throw filesystem error
  std::filesystem::path getRefImagePath() const
  {
    //check if ref image exists for each possible extension (currently pgm and ppm) and return path if so
    for (const auto& extension : bp_file_handling::IN_IMAGE_POSS_EXTENSIONS) {
      if (std::filesystem::exists((stereo_set_path_ / (bp_file_handling::REF_IMAGE_NAME + "." + extension)))) {
        return stereo_set_path_ / (bp_file_handling::REF_IMAGE_NAME + "." + extension);
      }
    }

    throw std::filesystem::filesystem_error("Reference image not found", std::error_code());
  }

  //return path to test image with valid extension if found, otherwise throw filesystem error
  std::filesystem::path getTestImagePath() const
  {
    //check if test image exists for each possible extension (currently pgm and ppm) and return path if so
    for (const auto& extension : bp_file_handling::IN_IMAGE_POSS_EXTENSIONS) {
      if (std::filesystem::exists((stereo_set_path_ / (bp_file_handling::TEST_IMAGE_NAME + "." + extension)))) {
        return stereo_set_path_ / (bp_file_handling::TEST_IMAGE_NAME + "." + extension);
      }
    }

    throw std::filesystem::filesystem_error("Test image not found", std::error_code());
  }

  //return path to use for current output disparity and then increment (to support multiple computed output disparity maps)
  const std::filesystem::path getCurrentOutputDisparityFilePathAndIncrement() {
    return stereo_set_path_ / (bp_file_handling::OUT_DISP_IMAGE_NAME_BASE + std::to_string(num_out_disp_map_++) + ".pgm");

  }

  //get file path to ground truth disparity map
  const std::filesystem::path getGroundTruthDisparityFilePath() const {
    return stereo_set_path_ / (bp_file_handling::GROUND_TRUTH_DISP_FILE);
  }

private:
  std::filesystem::path stereo_set_path_;
  unsigned int num_out_disp_map_;
};

#endif /* BPFILEHANDLING_H_ */
