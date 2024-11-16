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
#include "BpTypeConstraints.h"
#include "RunSettingsEval/RunTypeConstraints.h"
#include "RunSettingsEval/RunSettings.h"
#include "RunSettingsEval/RunData.h"

//parameters type requires runData() function to return the parameters as a
//RunData object
template <typename T>
concept Params_t =
  requires(T t) {
    { t.runData() } -> std::same_as<RunData>;
  };

namespace beliefprop {

//structure to store the settings for the number of levels and iterations
struct BPsettings
{
  //initally set to default values
  unsigned int numLevels_{bp_params::LEVELS_BP};
  unsigned int numIterations_{bp_params::ITER_BP};
  float smoothingSigma_{bp_params::SIGMA_BP};
  float lambda_bp_{bp_params::LAMBDA_BP};
  float data_k_bp_{bp_params::DATA_K_BP};
  //discontinuity cost cap set to infinity by default but is
  //expected to be dependent on number of disparity values and set when
  //number of disparity values is set
  float disc_k_bp_{bp_consts::INF_BP};
  //number of disparity values must be set for each stereo set
  unsigned int numDispVals_{0};

  RunData runData() const {
    RunData currRunData;
    currRunData.addDataWHeader("Num Possible Disparity Values", numDispVals_);
    currRunData.addDataWHeader("Num BP Levels", numLevels_);
    currRunData.addDataWHeader("Num BP Iterations", numIterations_);
    currRunData.addDataWHeader("DISC_K_BP", (double)disc_k_bp_);
    currRunData.addDataWHeader("DATA_K_BP", (double)data_k_bp_);
    currRunData.addDataWHeader("LAMBDA_BP", (double)lambda_bp_);
    currRunData.addDataWHeader("SIGMA_BP", (double)smoothingSigma_);

    return currRunData;
  }

  friend std::ostream& operator<<(std::ostream& resultsStream, const BPsettings& bpSettings);
};

inline std::ostream& operator<<(std::ostream& resultsStream, const BPsettings& bpSettings) {
  resultsStream << "Num Possible Disparity Values: " << bpSettings.numDispVals_ << "\n";
  resultsStream << "Num BP Levels: " << bpSettings.numLevels_ << "\n";
  resultsStream << "Num BP Iterations: " << bpSettings.numIterations_ << "\n";
  resultsStream << "DISC_K_BP: " << bpSettings.disc_k_bp_ << "\n";
  resultsStream << "DATA_K_BP: " << bpSettings.data_k_bp_ << "\n";
  resultsStream << "LAMBDA_BP: " << bpSettings.lambda_bp_ << "\n";
  resultsStream << "SIGMA_BP: " << bpSettings.smoothingSigma_ << "\n";
  
  return resultsStream;  
}


//structure to store the properties of the current level
//
struct levelProperties
{
  levelProperties(const std::array<unsigned int, 2>& widthHeight, unsigned long offsetIntoArrays, unsigned int levelNum,
    run_environment::AccSetting accSetting) :
    widthLevel_(widthHeight[0]), heightLevel_(widthHeight[1]),
    bytesAlignMemory_(run_environment::getBytesAlignMemory(accSetting)),
    numDataAlignWidth_(run_environment::getNumDataAlignWidth(accSetting)),
    widthCheckerboardLevel_(getCheckerboardWidthTargetDevice(widthLevel_)),
    paddedWidthCheckerboardLevel_(getPaddedCheckerboardWidth(widthCheckerboardLevel_)),
    offsetIntoArrays_(offsetIntoArrays), levelNum_(levelNum),
    divPaddedChBoardWAlign_{(accSetting == run_environment::AccSetting::AVX512) ? 16u : 8u} {}
  
  levelProperties(const std::array<unsigned int, 2>& widthHeight, unsigned long offsetIntoArrays, unsigned int levelNum,
    unsigned int bytesAlignMemory, unsigned int numDataAlignWidth, unsigned int divPaddedChBoardWAlign) :
    widthLevel_(widthHeight[0]), heightLevel_(widthHeight[1]),
    bytesAlignMemory_(bytesAlignMemory),
    numDataAlignWidth_(numDataAlignWidth),
    widthCheckerboardLevel_(getCheckerboardWidthTargetDevice(widthLevel_)),
    paddedWidthCheckerboardLevel_(getPaddedCheckerboardWidth(widthCheckerboardLevel_)),
    offsetIntoArrays_(offsetIntoArrays), levelNum_(levelNum), divPaddedChBoardWAlign_(divPaddedChBoardWAlign) {}

  //get bp level properties for next (higher) level in hierarchy that processed data with half width/height of current level
  template <RunData_t T>
  beliefprop::levelProperties getNextLevelProperties(const unsigned int numDisparityValues) const {
    const auto offsetNextLevel = offsetIntoArrays_ + getNumDataInBpArrays<T>(numDisparityValues);
    return levelProperties({(unsigned int)ceil((float)widthLevel_ / 2.0f), (unsigned int)ceil((float)heightLevel_ / 2.0f)},
      offsetNextLevel, (levelNum_ + 1), bytesAlignMemory_, numDataAlignWidth_, divPaddedChBoardWAlign_);
  }

  //get the amount of data in each BP array (data cost/messages for each checkerboard) at the current level
  //with the given number of possible movements
  template <RunData_t T>
  unsigned int getNumDataInBpArrays(const unsigned int numDisparityValues) const {
    return getNumDataForAlignedMemoryAtLevel<T>({widthLevel_, heightLevel_}, numDisparityValues);
  }

  unsigned int getCheckerboardWidthTargetDevice(const unsigned int widthLevel) const {
    return (unsigned int)std::ceil(((float)widthLevel) / 2.0f);
  }

  unsigned int getPaddedCheckerboardWidth(const unsigned int checkerboardWidth) const
  {
    //add "padding" to checkerboard width if necessary for alignment
    return ((checkerboardWidth % numDataAlignWidth_) == 0) ?
           checkerboardWidth :
           (checkerboardWidth + (numDataAlignWidth_ - (checkerboardWidth % numDataAlignWidth_)));
  }

  template <RunData_t T>
  unsigned long getNumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& widthHeightLevel,
      const unsigned int totalPossibleMovements) const
  {
    const unsigned long numDataAtLevel = (unsigned long)getPaddedCheckerboardWidth(getCheckerboardWidthTargetDevice(widthHeightLevel[0])) *
      ((unsigned long)widthHeightLevel[1]) * (unsigned long)totalPossibleMovements;
    unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

    if ((numBytesAtLevel % bytesAlignMemory_) == 0) {
      return numDataAtLevel;
    }
    else {
      numBytesAtLevel += (bytesAlignMemory_ - (numBytesAtLevel % bytesAlignMemory_));
      return (numBytesAtLevel / sizeof(T));
    }
  }

  template <RunData_t T, run_environment::AccSetting ACCELERATION>
  static unsigned long getTotalDataForAlignedMemoryAllLevels(const std::array<unsigned int, 2>& widthHeightBottomLevel,
    const unsigned int totalPossibleMovements, const unsigned int numLevels)
  {
    beliefprop::levelProperties currLevelProperties(widthHeightBottomLevel, 0, 0, ACCELERATION);
    unsigned long totalData = currLevelProperties.getNumDataInBpArrays<T>(totalPossibleMovements);
    for (unsigned int currLevelNum = 1; currLevelNum < numLevels; currLevelNum++) {
      currLevelProperties = currLevelProperties.getNextLevelProperties<T>(totalPossibleMovements);
      totalData += currLevelProperties.getNumDataInBpArrays<T>(totalPossibleMovements);
    }

    return totalData;
  }


  unsigned int widthLevel_;
  unsigned int heightLevel_;
  unsigned int bytesAlignMemory_;
  unsigned int numDataAlignWidth_;
  unsigned int widthCheckerboardLevel_;
  unsigned int paddedWidthCheckerboardLevel_;
  unsigned long offsetIntoArrays_;
  unsigned int levelNum_;
  //avx512 requires data to be aligned on 64 bytes (16 float values); otherwise data set to be
  //aligned on 32 bytes (8 float values)
  unsigned int divPaddedChBoardWAlign_;
};

//used to define the two checkerboard "parts" that the image is divided into
enum class Checkerboard_Parts {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1 };
enum class Message_Arrays : unsigned int { MESSAGES_U_CHECKERBOARD_0 = 0, MESSAGES_D_CHECKERBOARD_0, MESSAGES_L_CHECKERBOARD_0, MESSAGES_R_CHECKERBOARD_0,
                      MESSAGES_U_CHECKERBOARD_1, MESSAGES_D_CHECKERBOARD_1, MESSAGES_L_CHECKERBOARD_1, MESSAGES_R_CHECKERBOARD_1 };
enum class messageComp { U_MESSAGE, D_MESSAGE, L_MESSAGE, R_MESSAGE };

template <RunData_ptr T>
struct checkerboardMessages
{
  //each checkerboard messages element corresponds to separate Message_Arrays enum that go from 0 to 7 (8 total)
  //could use a map/unordered map to map Message_Arrays enum to corresponding message array but using array structure is likely faster
  std::array<T, 8> checkerboardMessagesAtLevel_;
};

//belief propagation checkerboard messages and data costs must be pointers to a bp data type
template <RunData_ptr T>
struct dataCostData
{
  T dataCostCheckerboard0_;
  T dataCostCheckerboard1_;
};

//enum corresponding to each kernel in belief propagation that can be run in parallel
enum class BpKernel : unsigned int { BLUR_IMAGES, DATA_COSTS_AT_LEVEL, INIT_MESSAGE_VALS, BP_AT_LEVEL,
                COPY_AT_LEVEL, OUTPUT_DISP };
constexpr unsigned int NUM_KERNELS{6u};

};

#endif /* BPSTRUCTSANDENUMS_H_ */
