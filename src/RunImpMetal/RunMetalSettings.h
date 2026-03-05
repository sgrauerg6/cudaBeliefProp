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
 * @file RunMetalSettings.h
 * @author Scott Grauer-Gray
 * @brief Contains namespace with constants and functions to get Metal device
 * properties as well as default and test parallel parameters to use in Metal
 * implementation run optimization
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef RUN_METAL_SETTINGS_H_
#define RUN_METAL_SETTINGS_H_

#include <vector>
#include <array>
#include <set>
#include "RunEval/RunData.h"

//set data type used for half-precision with Metal
/*#if defined(USE_BFLOAT16_FOR_HALF_PRECISION)
using halftype = bfloat;
#else
using halftype = half;
//#endif //USE_BFLOAT16_FOR_HALF_PRECISION*/
using halftype = half;

/**
 * @brief Namespace with constants and functions to get CUDA device properties
 * as well as default and test parallel parameters to use in CUDA
 * implementation run optimization
 */
namespace run_cuda {

constexpr std::string_view kMetalDesc{"Metal"};

/** @brief Default thread block dimensions (which is what parallel parameters
 *  corresponds to in Metal implementation) */
constexpr std::array<unsigned int, 2> kParallelParamsDefault{{32, 4}};

#if defined(DEFAULT_PARALLEL_PARAMS_ONLY)

/** @brief Empty parallel parameter alternative options since setting is to
  *  only use default parallel parameters. */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{};

#elif defined(LIMITED_ALT_PARALLEL_PARAMS)

/** @brief Parallel parameter alternative options to run to retrieve optimized parallel
 *  parameters in Metal implementation
 *  Parallel parameter corresponds to thread block dimensions in Metal implementation
 *  OK to include default parallel parameters in alternative options but not required */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{
  {16, 1}, {32, 4}, {64, 4}};

#else

/** @brief Parallel parameter alternative options to run to retrieve optimized parallel
 *  parameters in Metal implementation
 *  Parallel parameter corresponds to thread block dimensions in Metal implementation
 *  OK to include default parallel parameters in alternative options but not required */
const std::set<std::array<unsigned int, 2>> kParallelParameterAltOptions{
  {16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5},
  {32, 6}, {32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4},
  {128, 1}, {128, 2}, {256, 1}, {32, 10}, {32, 12}, {32, 14}, {32, 16},
  {64, 5}, {64, 6}, {64, 7}, {64, 8}, {128, 3}, {128, 4}, {256, 2}};

#endif

};

#endif /* RUN_METAL_SETTINGS_H_ */
