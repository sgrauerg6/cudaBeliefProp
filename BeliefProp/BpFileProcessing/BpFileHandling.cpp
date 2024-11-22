/*
 * BpFileHandling.cpp
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#include "BpFileHandling.h"

//retrieve path of stereo images to process using kBeliefPropDirectoryName and kStereoSetsDirectoryName
//constants
std::filesystem::path BpFileHandling::StereoSetsPath() const
{
  std::filesystem::path current_path = std::filesystem::current_path();
  while (true)
  {
    //create directory iterator corresponding to current path
    std::filesystem::directory_iterator dir_iter = std::filesystem::directory_iterator(current_path);

    //check if any of the directories in the current path correspond to the belief prop directory;
    //if so return iterator to directory; otherwise return iterator to end indicating that directory not
    //found in current path
    std::filesystem::directory_iterator it = std::find_if(std::filesystem::begin(dir_iter), std::filesystem::end(dir_iter), 
      [](const auto& p) { return p.path().stem() == bp_file_handling::kBeliefPropDirectoryName; });

    //check if return from find_if at iterator end and therefore didn't find stereo sets directory;
    //if that's the case continue to outer directory
    //for now assuming stereo sets directory exists in some outer directory and program won't work without it
    if (it == std::filesystem::end(dir_iter))
    {
      //if current path same as parent path, then can't continue and throw error
      if (current_path == current_path.parent_path()) {
        throw std::filesystem::filesystem_error("Belief prop directory not found", std::error_code());
      }
      current_path = current_path.parent_path();
    }
    else {
      std::filesystem::path stereo_set_path = it->path() / bp_file_handling::kStereoSetsDirectoryName;
      if (std::filesystem::is_directory(stereo_set_path)) {
        return stereo_set_path;
      }
      else {
        throw std::filesystem::filesystem_error("Stereo set directory not found in belief prop directory", std::error_code());
      }      
    }
  }
  return std::filesystem::path();
}

//return path to reference image with valid extension if found, otherwise throw filesystem error
std::filesystem::path BpFileHandling::RefImagePath() const
{
  //check if ref image exists for each possible extension (currently pgm and ppm) and return path if so
  for (const auto& extension : bp_file_handling::kInImagePossExtensions) {
    if (std::filesystem::exists((stereo_set_path_ / (std::string(bp_file_handling::kRefImageName) + "." + std::string(extension))))) {
      return stereo_set_path_ / (std::string(bp_file_handling::kRefImageName) + "." + std::string(extension));
    }
  }

  throw std::filesystem::filesystem_error("Reference image not found", std::error_code());
}

//return path to test image with valid extension if found, otherwise throw filesystem error
std::filesystem::path BpFileHandling::TestImagePath() const
{
  //check if test image exists for each possible extension (currently pgm and ppm) and return path if so
  for (const auto& extension : bp_file_handling::kInImagePossExtensions) {
    if (std::filesystem::exists((stereo_set_path_ / (std::string(bp_file_handling::kTestImageName) + "." + std::string(extension))))) {
      return stereo_set_path_ / (std::string(bp_file_handling::kTestImageName) + "." + std::string(extension));
    }
  }

  throw std::filesystem::filesystem_error("Test image not found", std::error_code());
}
