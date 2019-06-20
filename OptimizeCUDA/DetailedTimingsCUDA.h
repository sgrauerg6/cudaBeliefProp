/*
 * DetailedTimingsCUDA.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef DETAILEDTIMINGSCUDA_H_
#define DETAILEDTIMINGSCUDA_H_

#include "DetailedTimings.h"
#include <algorithm>

class DetailedTimingsCUDA : public DetailedTimings
{
public:

	std::vector<double> totalTimeInitSettingsMallocStart;
	std::vector<double> totalTimeGetDataCostsBottomLevel;
	std::vector<double> totalTimeGetDataCostsHigherLevels;
	std::vector<double> totalTimeInitMessageVals;
	std::vector<double> totalTimeBpIters;
	std::vector<double> timeBpItersKernelTotalTime;
	std::vector<double> totalTimeCopyData;
	std::vector<double> timeCopyDataKernelTotalTime;
	std::vector<double> totalTimeGetOutputDisparity;
	std::vector<double> totalTimeFinalUnbindFree;
	std::vector<double> totalTimeFinalFree;
	std::vector<double> totalTimed;
	std::vector<double> totalTimeInitMessageValuesKernelTime;
	std::vector<double> totalMemoryProcessingTime;
	std::vector<double> totalComputationProcessing;

	void addTimings(DetailedTimings* timingsToAdd)
	{
		//Can assume that timingsToAdd is of same type as current class
		DetailedTimingsCUDA* timingsToAddCUDA = static_cast<DetailedTimingsCUDA*>(timingsToAdd);
		totalTimeInitSettingsMallocStart.push_back(timingsToAddCUDA->totalTimeInitSettingsMallocStart.at(0));
		totalTimeGetDataCostsBottomLevel.push_back(timingsToAddCUDA->totalTimeGetDataCostsBottomLevel.at(0));
		totalTimeGetDataCostsHigherLevels.push_back(timingsToAddCUDA->totalTimeGetDataCostsHigherLevels.at(0));
		totalTimeInitMessageVals.push_back(timingsToAddCUDA->totalTimeInitMessageVals.at(0));
		totalTimeBpIters.push_back(timingsToAddCUDA->totalTimeBpIters.at(0));
		timeBpItersKernelTotalTime.push_back(timingsToAddCUDA->timeBpItersKernelTotalTime.at(0));
		totalTimeCopyData.push_back(timingsToAddCUDA->totalTimeCopyData.at(0));
		timeCopyDataKernelTotalTime.push_back(timingsToAddCUDA->timeCopyDataKernelTotalTime.at(0));
		totalTimeGetOutputDisparity.push_back(timingsToAddCUDA->totalTimeGetOutputDisparity.at(0));
		totalTimeFinalUnbindFree.push_back(timingsToAddCUDA->totalTimeFinalUnbindFree.at(0));
		totalTimeFinalFree.push_back(timingsToAddCUDA->totalTimeFinalFree.at(0));
		totalTimed.push_back(timingsToAddCUDA->totalTimed.at(0));
		totalTimeInitMessageValuesKernelTime.push_back(timingsToAddCUDA->totalTimeInitMessageValuesKernelTime.at(0));
		totalMemoryProcessingTime.push_back(timingsToAddCUDA->totalMemoryProcessingTime.at(0));
		totalComputationProcessing.push_back(timingsToAddCUDA->totalComputationProcessing.at(0));
		totNumTimings = totalTimeInitSettingsMallocStart.size();
	}

	void SortTimings()
	{
		std::sort(totalTimeInitSettingsMallocStart.begin(), totalTimeInitSettingsMallocStart.end());
		std::sort(totalTimeGetDataCostsBottomLevel.begin(), totalTimeGetDataCostsBottomLevel.end());
		std::sort(totalTimeGetDataCostsHigherLevels.begin(), totalTimeGetDataCostsHigherLevels.end());
		std::sort(totalTimeInitMessageVals.begin(), totalTimeInitMessageVals.end());
		std::sort(totalTimeBpIters.begin(), totalTimeBpIters.end());
		std::sort(timeBpItersKernelTotalTime.begin(), timeBpItersKernelTotalTime.end());
		std::sort(totalTimeCopyData.begin(), totalTimeCopyData.end());
		std::sort(timeCopyDataKernelTotalTime.begin(), timeCopyDataKernelTotalTime.end());
		std::sort(totalTimeGetOutputDisparity.begin(), totalTimeGetOutputDisparity.end());
		std::sort(totalTimeFinalUnbindFree.begin(), totalTimeFinalUnbindFree.end());
		std::sort(totalTimeFinalFree.begin(), totalTimeFinalFree.end());
		std::sort(totalTimed.begin(), totalTimed.end());
		std::sort(totalTimeInitMessageValuesKernelTime.begin(), totalTimeInitMessageValuesKernelTime.end());
		std::sort(totalMemoryProcessingTime.begin(), totalMemoryProcessingTime.end());
		std::sort(totalComputationProcessing.begin(), totalComputationProcessing.end());
	}

	void PrintMedianTimings()
	{
		SortTimings();
		printf("Median Timings\n");
		printf("Time init settings malloc: %f\n", totalTimeInitSettingsMallocStart.at(totNumTimings/2));
		printf("Time get data costs bottom level: %f\n", totalTimeGetDataCostsBottomLevel.at(totNumTimings/2));
		printf("Time get data costs higher levels: %f\n", totalTimeGetDataCostsHigherLevels.at(totNumTimings/2));
		printf("Time to init message values: %f\n", totalTimeInitMessageVals.at(totNumTimings/2));
		printf("Time to init message values (kernel portion only): %f\n", totalTimeInitMessageValuesKernelTime.at(totNumTimings/2));
		printf("Total time BP Iters: %f\n", totalTimeBpIters.at(totNumTimings/2));
		printf("Total time BP Iters (kernel portion only): %f\n", timeBpItersKernelTotalTime.at(totNumTimings/2));
		printf("Total time Copy Data: %f\n", totalTimeCopyData.at(totNumTimings/2));
		printf("Total time Copy Data (kernel portion only): %f\n", timeCopyDataKernelTotalTime.at(totNumTimings/2));
		printf("Time get output disparity: %f\n", totalTimeGetOutputDisparity.at(totNumTimings/2));
		printf("Time final unbind free: %f\n", totalTimeFinalUnbindFree.at(totNumTimings/2));
		printf("Time final free: %f\n", totalTimeFinalFree.at(totNumTimings/2));
		printf("Total timed: %f\n", totalTimed.at(totNumTimings/2));
		printf("Total memory processing time: %f\n", totalMemoryProcessingTime.at(totNumTimings/2));
		printf("Total computation processing time: %f\n", totalComputationProcessing.at(totNumTimings/2));
	}

	void PrintMedianTimingsToFile(FILE* pFile) {
		SortTimings();
		fprintf(pFile, "Median Timings\n");
		fprintf(pFile, "Time init settings malloc: %f\n",
				totalTimeInitSettingsMallocStart.at(totNumTimings / 2));
		fprintf(pFile, "Time get data costs bottom level: %f\n",
				totalTimeGetDataCostsBottomLevel.at(totNumTimings / 2));
		fprintf(pFile, "Time get data costs higher levels: %f\n",
				totalTimeGetDataCostsHigherLevels.at(totNumTimings / 2));
		fprintf(pFile, "Time to init message values: %f\n",
				totalTimeInitMessageVals.at(totNumTimings / 2));
		fprintf(pFile,
				"Time to init message values (kernel portion only): %f\n",
				totalTimeInitMessageValuesKernelTime.at(totNumTimings / 2));
		fprintf(pFile, "Total time BP Iters: %f\n",
				totalTimeBpIters.at(totNumTimings / 2));
		fprintf(pFile, "Total time BP Iters (kernel portion only): %f\n",
				timeBpItersKernelTotalTime.at(totNumTimings / 2));
		fprintf(pFile, "Total time Copy Data: %f\n",
				totalTimeCopyData.at(totNumTimings / 2));
		fprintf(pFile, "Total time Copy Data (kernel portion only): %f\n",
				timeCopyDataKernelTotalTime.at(totNumTimings / 2));
		fprintf(pFile, "Time get output disparity: %f\n",
				totalTimeGetOutputDisparity.at(totNumTimings / 2));
		fprintf(pFile, "Time final unbind free: %f\n",
				totalTimeFinalUnbindFree.at(totNumTimings / 2));
		fprintf(pFile, "Time final free: %f\n",
				totalTimeFinalFree.at(totNumTimings / 2));
		fprintf(pFile, "Total timed: %f\n", totalTimed.at(totNumTimings / 2));
		fprintf(pFile, "Total memory processing time: %f\n",
				totalMemoryProcessingTime.at(totNumTimings / 2));
		fprintf(pFile, "Total computation processing time: %f\n",
				totalComputationProcessing.at(totNumTimings / 2));
	}
};

#endif /* DETAILEDTIMINGSCUDA_H_ */
