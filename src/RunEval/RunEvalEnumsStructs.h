/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file RunEvalEnumsStructs.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with enums and structs for implementation run
 * evaluation
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_EVAL_ENUMS_STRUCTS_H_
#define RUN_EVAL_ENUMS_STRUCTS_H_

#include <array>
#include <string_view>
#include <map>
#include <filesystem>

/** 
 * @brief Namespace with enums for implementation run evaluation
 */
namespace run_eval {

  /** @brief Enum for status to indicate if error or no error */
  enum class Status { kNoError, kError };

  /** @brief Enum to specify average or median for "middle" value in data */
  enum class MiddleValData { kAverage, kMedian };

  //declare output results type and array containing all output results types
  enum class OutResults{
    kDefaultPParams, kOptPParams, kSpeedups, kOptWSpeedups
  };
    
  //structure containing directory path and description in file name for
  //each output result file
  struct OutFileInfo{
    std::filesystem::path dir_path;
    std::string_view desc_file_name;
  };
};

#endif //RUN_EVAL_ENUMS_STRUCTS_H_