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

namespace beliefprop {

const std::array<std::array<std::string, 3>, 10> LEVEL_DCOST_BPTIME_CTIME_NAMES{
    std::array<std::string, 3>{"Level 0 Data Costs", "Level 0 BP Runtime", "Level 0 Copy Runtime"},
    std::array<std::string, 3>{"Level 1 Data Costs", "Level 1 BP Runtime", "Level 1 Copy Runtime"},
    std::array<std::string, 3>{"Level 2 Data Costs", "Level 2 BP Runtime", "Level 2 Copy Runtime"},
    std::array<std::string, 3>{"Level 3 Data Costs", "Level 3 BP Runtime", "Level 3 Copy Runtime"},
    std::array<std::string, 3>{"Level 4 Data Costs", "Level 4 BP Runtime", "Level 4 Copy Runtime"},
    std::array<std::string, 3>{"Level 5 Data Costs", "Level 5 BP Runtime", "Level 5 Copy Runtime"},
    std::array<std::string, 3>{"Level 6 Data Costs", "Level 6 BP Runtime", "Level 6 Copy Runtime"},
    std::array<std::string, 3>{"Level 7 Data Costs", "Level 7 BP Runtime", "Level 7 Copy Runtime"},
    std::array<std::string, 3>{"Level 8 Data Costs", "Level 8 BP Runtime", "Level 8 Copy Runtime"},
    std::array<std::string, 3>{"Level 9 Data Costs", "Level 9 BP Runtime", "Level 9 Copy Runtime"}};

enum class Runtime_Type { INIT_SETTINGS_MALLOC, LEVEL_0_DATA_COSTS, LEVEL_1_DATA_COSTS, LEVEL_2_DATA_COSTS,
  LEVEL_3_DATA_COSTS, LEVEL_4_DATA_COSTS, LEVEL_5_DATA_COSTS, LEVEL_6_DATA_COSTS,
  LEVEL_7_DATA_COSTS, LEVEL_8_DATA_COSTS, LEVEL_9_DATA_COSTS, DATA_COSTS_HIGHER_LEVEL, INIT_MESSAGES,
  INIT_MESSAGES_KERNEL, BP_ITERS, COPY_DATA, COPY_DATA_KERNEL, COPY_DATA_MEM_MANAGEMENT, OUTPUT_DISPARITY, FINAL_FREE,
  TOTAL_TIMED, SMOOTHING, LEVEL_0_BP, LEVEL_1_BP, LEVEL_2_BP, LEVEL_3_BP, LEVEL_4_BP, LEVEL_5_BP, LEVEL_6_BP, LEVEL_7_BP,
  LEVEL_8_BP, LEVEL_9_BP, LEVEL_0_COPY, LEVEL_1_COPY, LEVEL_2_COPY, LEVEL_3_COPY, LEVEL_4_COPY, LEVEL_5_COPY, LEVEL_6_COPY, LEVEL_7_COPY,
  LEVEL_8_COPY, LEVEL_9_COPY, TOTAL_BP, TOTAL_NO_TRANSFER, TOTAL_WITH_TRANSFER };

const std::unordered_map<Runtime_Type, std::string> timingNames = {
  {Runtime_Type::INIT_SETTINGS_MALLOC, "Time init settings malloc"}, 
  {Runtime_Type::LEVEL_0_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[0][0]}, 
  {Runtime_Type::LEVEL_1_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[1][0]}, 
  {Runtime_Type::LEVEL_2_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[2][0]}, 
  {Runtime_Type::LEVEL_3_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[3][0]}, 
  {Runtime_Type::LEVEL_4_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[4][0]},
  {Runtime_Type::LEVEL_5_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[5][0]},
  {Runtime_Type::LEVEL_6_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[6][0]},
  {Runtime_Type::LEVEL_7_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[7][0]},
  {Runtime_Type::LEVEL_8_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[8][0]},
  {Runtime_Type::LEVEL_9_DATA_COSTS, LEVEL_DCOST_BPTIME_CTIME_NAMES[9][0]},
  {Runtime_Type::DATA_COSTS_HIGHER_LEVEL, "Total time get data costs after bottom level"},
  {Runtime_Type::INIT_MESSAGES, "Time to init message values"}, 
  {Runtime_Type::INIT_MESSAGES_KERNEL, "Time to init message values (kernel portion only)"}, 
  {Runtime_Type::BP_ITERS, "Total time BP Iters"},
  {Runtime_Type::COPY_DATA, "Total time Copy Data"},
  {Runtime_Type::COPY_DATA_KERNEL, "Total time Copy Data (kernel portion only)"}, 
  {Runtime_Type::COPY_DATA_MEM_MANAGEMENT, "Total time Copy Data (memory management portion only)"}, 
  {Runtime_Type::OUTPUT_DISPARITY, "Time get output disparity"},
  {Runtime_Type::FINAL_FREE, "Time final free"}, 
  {Runtime_Type::TOTAL_TIMED, "Total timed"}, 
  {Runtime_Type::SMOOTHING, "Smoothing Runtime"}, 
  {Runtime_Type::LEVEL_0_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[0][1]}, 
  {Runtime_Type::LEVEL_1_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[1][1]}, 
  {Runtime_Type::LEVEL_2_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[2][1]}, 
  {Runtime_Type::LEVEL_3_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[3][1]},
  {Runtime_Type::LEVEL_4_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[4][1]}, 
  {Runtime_Type::LEVEL_5_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[5][1]}, 
  {Runtime_Type::LEVEL_6_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[6][1]}, 
  {Runtime_Type::LEVEL_7_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[7][1]}, 
  {Runtime_Type::LEVEL_8_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[8][1]}, 
  {Runtime_Type::LEVEL_9_BP, LEVEL_DCOST_BPTIME_CTIME_NAMES[9][1]}, 
  {Runtime_Type::LEVEL_0_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[0][2]}, 
  {Runtime_Type::LEVEL_1_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[1][2]}, 
  {Runtime_Type::LEVEL_2_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[2][2]}, 
  {Runtime_Type::LEVEL_3_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[3][2]}, 
  {Runtime_Type::LEVEL_4_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[4][2]}, 
  {Runtime_Type::LEVEL_5_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[5][2]}, 
  {Runtime_Type::LEVEL_6_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[6][2]}, 
  {Runtime_Type::LEVEL_7_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[7][2]}, 
  {Runtime_Type::LEVEL_8_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[8][2]}, 
  {Runtime_Type::LEVEL_9_COPY, LEVEL_DCOST_BPTIME_CTIME_NAMES[9][2]}, 
  {Runtime_Type::TOTAL_BP, "Total BP Runtime"}, 
  {Runtime_Type::TOTAL_NO_TRANSFER, "Total Runtime not including data transfer time"},
  {Runtime_Type::TOTAL_WITH_TRANSFER, "Total runtime including data transfer time"}};
}

#endif /* DETAILEDTIMINGBPCONSTS_H_ */
