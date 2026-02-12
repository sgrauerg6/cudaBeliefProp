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
 * @file BpConstsEnumsAliases.h
 * @author Scott Grauer-Gray
 * @brief File with namespace for enums, constants, structures, and
 * functions specific to belief propagation processing
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef BP_CONSTS_ENUMS_ALIASES_H_
#define BP_CONSTS_ENUMS_ALIASES_H_

#include <array>
#include "RunEval/RunTypeConstraints.h"

/**
 * @brief Namespace for enums, constants, structures, and
 * functions specific to belief propagation processing
 */
namespace beliefprop {

/** @brief Define the two checkerboard "parts" that the image is divided into */
enum class CheckerboardPart : size_t { kCheckerboardPart0, kCheckerboardPart1 };
enum class MessageArrays : size_t { 
  kMessagesUCheckerboard, kMessagesDCheckerboard, kMessagesLCheckerboard, kMessagesRCheckerboard };
enum class MessageComp { kUMessage, kDMessage, kLMessage, kRMessage };

/** @brief Number of checkerboard parts for processing */
constexpr size_t kNumCheckerboardParts{2};

/** @brief Number of message arrays in each checkerboard part */
constexpr size_t kNumMessageArrays{4};

/**
 * @brief Define alias for array with message costs for each bp
 * processing checkerboard.<br>
 * Each checkerboard messages element corresponds to an array of message values
 * and message values must be pointers to a bp data type.
 * 
 * @tparam T 
 */
template <RunData_ptr T>
using CheckerboardMessages = std::array<std::array<T, kNumMessageArrays>, kNumCheckerboardParts>;

/**
 * @brief Define alias for two-element array with data costs for each bp
 * processing checkerboard.<br>
 * Belief propagation data costs must be pointers to a bp data type.
 * 
 * 
 * @tparam T 
 */
template <RunData_ptr T>
using DataCostsCheckerboards = std::array<T, kNumCheckerboardParts>;

constexpr size_t kNumKernels{6};

/** @brief Enum corresponding to each kernel in belief propagation that can be run in parallel */
enum class BpKernel : size_t { 
  kBlurImages,
  kDataCostsAtLevel,
  kInitMessageVals,
  kBpAtLevel,
  kCopyAtLevel,
  kOutputDisp
};

};

#endif /* BP_CONSTS_ENUMS_ALIASES_H_ */
