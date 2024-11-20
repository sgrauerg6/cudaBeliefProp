/*
 * BpStructsAndEnums.h
 *
 *  Created on: Sep 22, 2019
 *      Author: scott
 */

#ifndef BPSTRUCTSANDENUMS_H_
#define BPSTRUCTSANDENUMS_H_

#include <array>
#include <vector>
#include <thread>
#include <iostream>
#include <cmath>
#include <string>
#include "BpStereoParameters.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"

namespace beliefprop {

//used to define the two checkerboard "parts" that the image is divided into
enum class Checkerboard_Part {kCheckerboardPart0, kCheckerboardPart1 };
enum class Message_Arrays : unsigned int { 
  kMessagesUCheckerboard0, kMessagesDCheckerboard0, kMessagesLCheckerboard0, kMessagesRCheckerboard0,
  kMessagesUCheckerboard1, kMessagesDCheckerboard1, kMessagesLCheckerboard1, kMessagesRCheckerboard1 };
enum class MessageComp { kUMessage, kDMessage, kLMessage, kRMessage };

//each checkerboard messages element corresponds to an array of message values that can be mapped to
//a unique value within Message_Arrays enum
//could use a map/unordered map to map Message_Arrays enum to corresponding message array but using array structure is likely faster
template <RunData_ptr T>
using CheckerboardMessages = std::array<T, 8>;

//belief propagation checkerboard messages and data costs must be pointers to a bp data type
//define alias for two-element array with data costs for each bp processing checkerboard
template <RunData_ptr T>
using DataCostsCheckerboards = std::array<T, 2>;

//enum corresponding to each kernel in belief propagation that can be run in parallel
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

#endif /* BPSTRUCTSANDENUMS_H_ */
