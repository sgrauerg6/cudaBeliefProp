/*
 * BpConstsEnumsAliases.h
 *
 *  Created on: Sep 22, 2019
 *      Author: scott
 */

#ifndef BP_CONSTS_ENUMS_ALIASES_H_
#define BP_CONSTS_ENUMS_ALIASES_H_

#include <array>
#include "RunEval/RunTypeConstraints.h"

namespace beliefprop {


/**
 * @brief Float value of "infinity" that works with half-precision
 * 
 */
constexpr float kInfBp{65504};

/**
 * @brief Define the two checkerboard "parts" that the image is divided into
 * 
 */
enum class CheckerboardPart : unsigned int { kCheckerboardPart0, kCheckerboardPart1 };
enum class MessageArrays : unsigned int { 
  kMessagesUCheckerboard, kMessagesDCheckerboard, kMessagesLCheckerboard, kMessagesRCheckerboard };
enum class MessageComp { kUMessage, kDMessage, kLMessage, kRMessage };

/**
 * @brief Number of checkerboard parts for processing
 * 
 */
constexpr unsigned int kNumCheckerboardParts{2};

/**
 * @brief Number of message arrays in each checkerboard part
 * 
 */
constexpr unsigned int kNumMessageArrays{4};

/**
 * @brief Each checkerboard messages element corresponds to an array of message values
 * could use a map/unordered map to map MessageArrays enum to corresponding message array
 * but using array structure is likely faster
 * 
 * @tparam T 
 */
template <RunData_ptr T>
using CheckerboardMessages = std::array<std::array<T, kNumMessageArrays>, kNumCheckerboardParts>;

/**
 * @brief Belief propagation checkerboard messages and data costs must be pointers to a bp data type
 * define alias for two-element array with data costs for each bp processing checkerboard
 * 
 * @tparam T 
 */
template <RunData_ptr T>
using DataCostsCheckerboards = std::array<T, kNumCheckerboardParts>;

/**
 * @brief Enum corresponding to each kernel in belief propagation that can be run in parallel
 * 
 */
constexpr unsigned int kNumKernels{6};
enum class BpKernel : unsigned int { 
  kBlurImages,
  kDataCostsAtLevel,
  kInitMessageVals,
  kBpAtLevel,
  kCopyAtLevel,
  kOutputDisp
};

};

#endif /* BP_CONSTS_ENUMS_ALIASES_H_ */
