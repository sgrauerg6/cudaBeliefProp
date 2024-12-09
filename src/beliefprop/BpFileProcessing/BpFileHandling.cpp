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
 * @file BpFileHandling.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "BpFileHandling.h"

/**
 * @brief Retrieve path of stereo images to process using kBeliefPropDirectoryName
 * and kStereoSetsDirectoryName constants
 * 
 * @return std::filesystem::path 
 */
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

/**
 * @brief Return path to reference image with valid extension if found,
 * otherwise throw filesystem error
 * 
 * @return std::filesystem::path 
 */
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

/**
 * @brief Return path to test image with valid extension if found,
 * otherwise throw filesystem error
 * 
 * @return std::filesystem::path 
 */
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
