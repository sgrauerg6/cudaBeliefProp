/*
 * DetailedTimingBPConsts.h
 *
 *  Created on: Sep 20, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGBPCONSTS_H_
#define DETAILEDTIMINGBPCONSTS_H_

enum class Runtime_Type_BP { INIT_SETTINGS_MALLOC, DATA_COSTS_BOTTOM_LEVEL, DATA_COSTS_HIGHER_LEVEL, INIT_MESSAGES,
    INIT_MESSAGES_KERNEL, BP_ITERS, COPY_DATA, COPY_DATA_KERNEL, COPY_DATA_MEM_MANAGEMENT, OUTPUT_DISPARITY, FINAL_FREE,
	TOTAL_TIMED, SMOOTHING, LEVEL_0, LEVEL_1, LEVEL_2, LEVEL_3, LEVEL_4, LEVEL_5, LEVEL_6, LEVEL_7, LEVEL_8, LEVEL_9,
	TOTAL_BP, TOTAL_NO_TRANSFER, TOTAL_WITH_TRANSFER };

const std::unordered_map<Runtime_Type_BP, std::string> timingNames_BP = {
	{Runtime_Type_BP::INIT_SETTINGS_MALLOC, "Time init settings malloc"}, 
	{Runtime_Type_BP::DATA_COSTS_BOTTOM_LEVEL, "Time get data costs bottom level"}, 
	{Runtime_Type_BP::DATA_COSTS_HIGHER_LEVEL, "Time get data costs higher levels"},
	{Runtime_Type_BP::INIT_MESSAGES, "Time to init message values"}, 
	{Runtime_Type_BP::INIT_MESSAGES_KERNEL, "Time to init message values (kernel portion only)"}, 
	{Runtime_Type_BP::BP_ITERS, "Total time BP Iters"}, {Runtime_Type_BP::COPY_DATA, "Total time Copy Data"},
	{Runtime_Type_BP::COPY_DATA_KERNEL, "Total time Copy Data (kernel portion only)"}, 
	{Runtime_Type_BP::COPY_DATA_MEM_MANAGEMENT, "Total time Copy Data (memory management portion only)"}, 
	{Runtime_Type_BP::OUTPUT_DISPARITY, "Time get output disparity"},
	{Runtime_Type_BP::FINAL_FREE, "Time final free"}, 
	{Runtime_Type_BP::TOTAL_TIMED, "Total timed"}, 
	{Runtime_Type_BP::SMOOTHING, "Smoothing Runtime"}, 
	{Runtime_Type_BP::LEVEL_0, "Level 0 Runtime"}, 
	{Runtime_Type_BP::LEVEL_1, "Level 1 Runtime"}, 
	{Runtime_Type_BP::LEVEL_2, "Level 2 Runtime"}, 
	{Runtime_Type_BP::LEVEL_3, "Level 3 Runtime"}, 
	{Runtime_Type_BP::LEVEL_4, "Level 4 Runtime"}, 
	{Runtime_Type_BP::LEVEL_5, "Level 5 Runtime"}, 
	{Runtime_Type_BP::LEVEL_6, "Level 6 Runtime"}, 
	{Runtime_Type_BP::LEVEL_7, "Level 7 Runtime"}, 
	{Runtime_Type_BP::LEVEL_8, "Level 8 Runtime"}, 
	{Runtime_Type_BP::LEVEL_9, "Level 9 Runtime"}, 
	{Runtime_Type_BP::TOTAL_BP, "Total BP Runtime"}, 
	{Runtime_Type_BP::TOTAL_NO_TRANSFER, "Total Runtime not including data transfer time"},
	{Runtime_Type_BP::TOTAL_WITH_TRANSFER, "Total runtime including data transfer time"}};


#endif /* DETAILEDTIMINGBPCONSTS_H_ */
