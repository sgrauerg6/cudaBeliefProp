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
 * @file EvaluateImpResultsBp.cpp
 * @author Scott Grauer-Gray
 * @brief Function definition for class to evaluate belief propagation
 * implementation results.
 * 
 * @copyright Copyright (c) 2024
 */

#include "EvaluateImpResultsBp.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include <filesystem>

//retrieve path of belief propagation implementation results
std::filesystem::path EvaluateImpResultsBp::GetImpResultsPath() const
{
  std::filesystem::path current_path = std::filesystem::current_path();
  while (true) {
    //create directory iterator corresponding to current path
    std::filesystem::directory_iterator dir_iter =
      std::filesystem::directory_iterator(current_path);
    
    //set variable corresponding to iterator past end of directory since it's
    //used multiple times
    auto dir_end_iter = std::filesystem::end(dir_iter);

    //check if any of the directories in the current path correspond to the
    //belief propagation directory; if so return iterator to directory;
    //otherwise return iterator to end indicating that directory not
    //found in current path
    std::filesystem::directory_iterator it =
      std::find_if(std::filesystem::begin(dir_iter),
                   dir_end_iter, 
                   [](const auto &p) {
                    return p.path().stem() ==
                      beliefprop::kBeliefPropDirectoryName; });
    
    //check if return from find_if at iterator end and therefore didn't find
    //belief propagation directory; if that's the case continue to outer
    //directory
    //for now assuming stereo sets directory exists in some outer directory and
    //program won't work without it
    if (it == dir_end_iter)
    {
      //if current path same as parent path, throw error
      if (current_path == current_path.parent_path()) {
        throw std::filesystem::filesystem_error(
          "Belief propagation directory not found", std::error_code());
      }
      //continue to next outer directory
      current_path = current_path.parent_path();
   }
    
    //retrieve and return path for implementation results which is a subfolder
    //inside of belief propagation directory
    if (it != dir_end_iter) {
      const std::filesystem::path impResultsPath{
        it->path() / run_eval::kImpResultsFolderName};
      if (!(std::filesystem::is_directory(impResultsPath))) {
        //create directory if it doesn't exist
        std::filesystem::create_directory(impResultsPath);
      }
      return impResultsPath;
    }
  }
}
