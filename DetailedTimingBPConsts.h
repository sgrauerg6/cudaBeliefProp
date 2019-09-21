/*
 * DetailedTimingBPConsts.h
 *
 *  Created on: Sep 20, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGBPCONSTS_H_
#define DETAILEDTIMINGBPCONSTS_H_

enum Runtime_Type_BP { INIT_SETTINGS_MALLOC, DATA_COSTS_BOTTOM_LEVEL, DATA_COSTS_HIGHER_LEVEL, INIT_MESSAGES, INIT_MESSAGES_KERNEL, BP_ITERS, COPY_DATA, COPY_DATA_KERNEL, COPY_DATA_MEM_MANAGEMENT, OUTPUT_DISPARITY, FINAL_FREE, TOTAL_TIMED, SMOOTHING, TOTAL_BP, TOTAL_NO_TRANSFER, TOTAL_WITH_TRANSFER };
const std::unordered_map<Runtime_Type_BP, std::string> timingNames_BP = {{INIT_SETTINGS_MALLOC, "Time init settings malloc"}, {DATA_COSTS_BOTTOM_LEVEL, "Time get data costs bottom level"}, {DATA_COSTS_HIGHER_LEVEL, "Time get data costs higher levels"},
			{INIT_MESSAGES, "Time to init message values"}, {INIT_MESSAGES_KERNEL, "Time to init message values (kernel portion only)"}, {BP_ITERS, "Total time BP Iters"}, {COPY_DATA, "Total time Copy Data"},
			{COPY_DATA_KERNEL, "Total time Copy Data (kernel portion only)"}, {COPY_DATA_MEM_MANAGEMENT, "Total time Copy Data (memory management portion only)"}, {OUTPUT_DISPARITY, "Time get output disparity"},
			{FINAL_FREE, "Time final free:"}, {TOTAL_TIMED, "Total timed"}, {SMOOTHING, "Smoothing Runtime"}, {TOTAL_BP, "Total BP Runtime"}, {TOTAL_NO_TRANSFER, "Total Runtime not including data transfer time)"},
			{TOTAL_WITH_TRANSFER, "Total runtime including data transfer time"}};


#endif /* DETAILEDTIMINGBPCONSTS_H_ */
