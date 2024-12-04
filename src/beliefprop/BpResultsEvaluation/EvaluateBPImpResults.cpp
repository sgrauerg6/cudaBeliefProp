/*
 * EvaluateBPImpResults.cpp
 *
 *  Created on: March 2, 2024
 *      Author: scott
 * 
 *  Function definition for class to evaluate belief propagation implementation results.
 */

#include "EvaluateBPImpResults.h"
#include "BpFileProcessing/BpFileHandlingConsts.h"
#include <filesystem>

//retrieve path of belief propagation implementation results
std::filesystem::path EvaluateBPImpResults::GetImpResultsPath() const
{
  std::filesystem::path current_path = std::filesystem::current_path();
  while (true) {
    //create directory iterator corresponding to current path
    std::filesystem::directory_iterator dir_iter = std::filesystem::directory_iterator(current_path);

    //check if any of the directories in the current path correspond to the belief propagation directory;
    //if so return iterator to directory; otherwise return iterator to end indicating that directory not
    //found in current path
    std::filesystem::directory_iterator it = std::find_if(std::filesystem::begin(dir_iter), std::filesystem::end(dir_iter), 
      [](const auto &p) { return p.path().stem() == bp_file_handling::kBeliefPropDirectoryName; });
    
    //check if return from find_if at iterator end and therefore didn't find belief propagation directory;
    //if that's the case continue to outer directory
    //for now assuming stereo sets directory exists in some outer directory and program won't work without it
    if (it == std::filesystem::end(dir_iter))
    {
      //if current path same as parent path, throw error
      if (current_path == current_path.parent_path()) {
        throw std::filesystem::filesystem_error("Belief propagation directory not found", std::error_code());
      }
      //continue to next outer directory
      current_path = current_path.parent_path();
   }
    
    //retrieve and return path for implementation results which is a subfolder inside of belief propagation directory
    if (it != std::filesystem::end(dir_iter)) {
      const std::filesystem::path impResultsPath{it->path() / run_eval::kImpResultsFolderName};
      if (!(std::filesystem::is_directory(impResultsPath))) {
        //create directory if it doesn't exist
        std::filesystem::create_directory(impResultsPath);
      }
      return impResultsPath;
    }
  }

  //return empty path if no path found
  return std::filesystem::path();
}
