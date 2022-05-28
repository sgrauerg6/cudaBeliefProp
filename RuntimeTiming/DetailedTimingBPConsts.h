/*
 * DetailedTimingBPConsts.h
 *
 *  Created on: Sep 20, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGBPCONSTS_H_
#define DETAILEDTIMINGBPCONSTS_H_

enum class Runtime_Type_BP { INIT_SETTINGS_MALLOC, LEVEL_0_DATA_COSTS, LEVEL_1_DATA_COSTS, LEVEL_2_DATA_COSTS,
    LEVEL_3_DATA_COSTS, LEVEL_4_DATA_COSTS, LEVEL_5_DATA_COSTS, LEVEL_6_DATA_COSTS,
	LEVEL_7_DATA_COSTS, LEVEL_8_DATA_COSTS, LEVEL_9_DATA_COSTS, DATA_COSTS_HIGHER_LEVEL, INIT_MESSAGES,
    INIT_MESSAGES_KERNEL, BP_ITERS, COPY_DATA, COPY_DATA_KERNEL, COPY_DATA_MEM_MANAGEMENT, OUTPUT_DISPARITY, FINAL_FREE,
	TOTAL_TIMED, SMOOTHING, LEVEL_0_BP, LEVEL_1_BP, LEVEL_2_BP, LEVEL_3_BP, LEVEL_4_BP, LEVEL_5_BP, LEVEL_6_BP, LEVEL_7_BP,
	LEVEL_8_BP, LEVEL_9_BP, LEVEL_0_COPY, LEVEL_1_COPY, LEVEL_2_COPY, LEVEL_3_COPY, LEVEL_4_COPY, LEVEL_5_COPY, LEVEL_6_COPY, LEVEL_7_COPY,
	LEVEL_8_COPY, LEVEL_9_COPY, TOTAL_BP, TOTAL_NO_TRANSFER, TOTAL_WITH_TRANSFER };

const std::unordered_map<Runtime_Type_BP, std::string> timingNames_BP = {
	{Runtime_Type_BP::INIT_SETTINGS_MALLOC, "Time init settings malloc"}, 
	{Runtime_Type_BP::LEVEL_0_DATA_COSTS, "Level 0 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_1_DATA_COSTS, "Level 1 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_2_DATA_COSTS, "Level 2 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_3_DATA_COSTS, "Level 3 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_4_DATA_COSTS, "Level 4 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_5_DATA_COSTS, "Level 5 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_6_DATA_COSTS, "Level 6 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_7_DATA_COSTS, "Level 7 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_8_DATA_COSTS, "Level 8 Data Costs"}, 
	{Runtime_Type_BP::LEVEL_9_DATA_COSTS, "Level 9 Data Costs"}, 
	{Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL, "Total time get data costs after bottom level"},
	{Runtime_Type_BP::INIT_MESSAGES, "Time to init message values"}, 
	{Runtime_Type_BP::INIT_MESSAGES_KERNEL, "Time to init message values (kernel portion only)"}, 
	{Runtime_Type_BP::BP_ITERS, "Total time BP Iters"}, {Runtime_Type_BP::COPY_DATA, "Total time Copy Data"},
	{Runtime_Type_BP::COPY_DATA_KERNEL, "Total time Copy Data (kernel portion only)"}, 
	{Runtime_Type_BP::COPY_DATA_MEM_MANAGEMENT, "Total time Copy Data (memory management portion only)"}, 
	{Runtime_Type_BP::OUTPUT_DISPARITY, "Time get output disparity"},
	{Runtime_Type_BP::FINAL_FREE, "Time final free"}, 
	{Runtime_Type_BP::TOTAL_TIMED, "Total timed"}, 
	{Runtime_Type_BP::SMOOTHING, "Smoothing Runtime"}, 
	{Runtime_Type_BP::LEVEL_0_BP, "Level 0 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_1_BP, "Level 1 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_2_BP, "Level 2 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_3_BP, "Level 3 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_4_BP, "Level 4 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_5_BP, "Level 5 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_6_BP, "Level 6 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_7_BP, "Level 7 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_8_BP, "Level 8 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_9_BP, "Level 9 BP Runtime"}, 
	{Runtime_Type_BP::LEVEL_0_COPY, "Level 0 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_1_COPY, "Level 1 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_2_COPY, "Level 2 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_3_COPY, "Level 3 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_4_COPY, "Level 4 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_5_COPY, "Level 5 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_6_COPY, "Level 6 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_7_COPY, "Level 7 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_8_COPY, "Level 8 Copy Runtime"}, 
	{Runtime_Type_BP::LEVEL_9_COPY, "Level 9 Copy Runtime"}, 
	{Runtime_Type_BP::TOTAL_BP, "Total BP Runtime"}, 
	{Runtime_Type_BP::TOTAL_NO_TRANSFER, "Total Runtime not including data transfer time"},
	{Runtime_Type_BP::TOTAL_WITH_TRANSFER, "Total runtime including data transfer time"}};


#endif /* DETAILEDTIMINGBPCONSTS_H_ */
