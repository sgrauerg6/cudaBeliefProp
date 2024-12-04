/*
 * DetailedTimingBPConsts.h
 *
 *  Created on: Sep 20, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGBPCONSTS_H_
#define DETAILEDTIMINGBPCONSTS_H_

#include <array>
#include <string>
#include <unordered_map>

namespace beliefprop {

constexpr std::array<std::array<std::string_view, 3>, 10> kLevelDCostBpTimeCTimeNames{
  std::array<std::string_view, 3>{"Level 0 Data Costs", "Level 0 BP Runtime", "Level 0 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 1 Data Costs", "Level 1 BP Runtime", "Level 1 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 2 Data Costs", "Level 2 BP Runtime", "Level 2 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 3 Data Costs", "Level 3 BP Runtime", "Level 3 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 4 Data Costs", "Level 4 BP Runtime", "Level 4 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 5 Data Costs", "Level 5 BP Runtime", "Level 5 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 6 Data Costs", "Level 6 BP Runtime", "Level 6 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 7 Data Costs", "Level 7 BP Runtime", "Level 7 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 8 Data Costs", "Level 8 BP Runtime", "Level 8 Copy Runtime"},
  std::array<std::string_view, 3>{"Level 9 Data Costs", "Level 9 BP Runtime", "Level 9 Copy Runtime"}
};

enum class Runtime_Type { kInitSettingsMalloc, kLevel0DataCosts, kLevel1DataCosts, kLevel2DataCosts,
  kLevel3DataCosts, kLevel4DataCosts, kLevel5DataCosts, kLevel6DataCosts,
  kLevel7DataCosts, kLevel8DataCosts, kLevel9DataCosts, kDataCostsHigherLevel, kInitMessages,
  kInitMessagesKernel, kBpIters, kCopyData, kCopyDataKernel, kCopyDataMemManagement, kOutputDisparity, kFinalFree,
  kTotalTimed, kSmoothing, kLevel0Bp, kLevel1Bp, kLevel2Bp, kLevel3Bp, kLevel4Bp, kLevel5Bp, kLevel6Bp, kLevel7Bp,
  kLevel8Bp, kLevel9Bp, kLevel0Copy, kLevel1Copy, kLevel2Copy, kLevel3Copy, kLevel4Copy, kLevel5Copy, kLevel6Copy, kLevel7Copy,
  kLevel8Copy, kLevel9Copy, kTotalBp, kTotalNoTransfer, kTotalWithTransfer };

const std::unordered_map<Runtime_Type, std::string_view> kTimingNames{
  {Runtime_Type::kInitSettingsMalloc, "Time init settings malloc"}, 
  {Runtime_Type::kLevel0DataCosts, kLevelDCostBpTimeCTimeNames[0][0]}, 
  {Runtime_Type::kLevel1DataCosts, kLevelDCostBpTimeCTimeNames[1][0]}, 
  {Runtime_Type::kLevel2DataCosts, kLevelDCostBpTimeCTimeNames[2][0]}, 
  {Runtime_Type::kLevel3DataCosts, kLevelDCostBpTimeCTimeNames[3][0]}, 
  {Runtime_Type::kLevel4DataCosts, kLevelDCostBpTimeCTimeNames[4][0]},
  {Runtime_Type::kLevel5DataCosts, kLevelDCostBpTimeCTimeNames[5][0]},
  {Runtime_Type::kLevel6DataCosts, kLevelDCostBpTimeCTimeNames[6][0]},
  {Runtime_Type::kLevel7DataCosts, kLevelDCostBpTimeCTimeNames[7][0]},
  {Runtime_Type::kLevel8DataCosts, kLevelDCostBpTimeCTimeNames[8][0]},
  {Runtime_Type::kLevel9DataCosts, kLevelDCostBpTimeCTimeNames[9][0]},
  {Runtime_Type::kDataCostsHigherLevel, "Total time get data costs after bottom level"},
  {Runtime_Type::kInitMessages, "Time to init message values"}, 
  {Runtime_Type::kInitMessagesKernel, "Time to init message values (kernel portion only)"}, 
  {Runtime_Type::kBpIters, "Total time BP Iters"},
  {Runtime_Type::kCopyData, "Total time Copy Data"},
  {Runtime_Type::kCopyDataKernel, "Total time Copy Data (kernel portion only)"}, 
  {Runtime_Type::kCopyDataMemManagement, "Total time Copy Data (memory management portion only)"}, 
  {Runtime_Type::kOutputDisparity, "Time get output disparity"},
  {Runtime_Type::kFinalFree, "Time final free"}, 
  {Runtime_Type::kTotalTimed, "Total timed"}, 
  {Runtime_Type::kSmoothing, "Smoothing Runtime"}, 
  {Runtime_Type::kLevel0Bp, kLevelDCostBpTimeCTimeNames[0][1]}, 
  {Runtime_Type::kLevel1Bp, kLevelDCostBpTimeCTimeNames[1][1]}, 
  {Runtime_Type::kLevel2Bp, kLevelDCostBpTimeCTimeNames[2][1]}, 
  {Runtime_Type::kLevel3Bp, kLevelDCostBpTimeCTimeNames[3][1]},
  {Runtime_Type::kLevel4Bp, kLevelDCostBpTimeCTimeNames[4][1]}, 
  {Runtime_Type::kLevel5Bp, kLevelDCostBpTimeCTimeNames[5][1]}, 
  {Runtime_Type::kLevel6Bp, kLevelDCostBpTimeCTimeNames[6][1]}, 
  {Runtime_Type::kLevel7Bp, kLevelDCostBpTimeCTimeNames[7][1]}, 
  {Runtime_Type::kLevel8Bp, kLevelDCostBpTimeCTimeNames[8][1]}, 
  {Runtime_Type::kLevel9Bp, kLevelDCostBpTimeCTimeNames[9][1]}, 
  {Runtime_Type::kLevel0Copy, kLevelDCostBpTimeCTimeNames[0][2]}, 
  {Runtime_Type::kLevel1Copy, kLevelDCostBpTimeCTimeNames[1][2]}, 
  {Runtime_Type::kLevel2Copy, kLevelDCostBpTimeCTimeNames[2][2]}, 
  {Runtime_Type::kLevel3Copy, kLevelDCostBpTimeCTimeNames[3][2]}, 
  {Runtime_Type::kLevel4Copy, kLevelDCostBpTimeCTimeNames[4][2]}, 
  {Runtime_Type::kLevel5Copy, kLevelDCostBpTimeCTimeNames[5][2]}, 
  {Runtime_Type::kLevel6Copy, kLevelDCostBpTimeCTimeNames[6][2]}, 
  {Runtime_Type::kLevel7Copy, kLevelDCostBpTimeCTimeNames[7][2]}, 
  {Runtime_Type::kLevel8Copy, kLevelDCostBpTimeCTimeNames[8][2]}, 
  {Runtime_Type::kLevel9Copy, kLevelDCostBpTimeCTimeNames[9][2]}, 
  {Runtime_Type::kTotalBp, "Total BP Runtime"}, 
  {Runtime_Type::kTotalNoTransfer, "Total Runtime not including data transfer time"},
  {Runtime_Type::kTotalWithTransfer, "Total runtime including data transfer time"}};
};

#endif /* DETAILEDTIMINGBPCONSTS_H_ */
