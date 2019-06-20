/*
 * ProcessBPOnTargetDevice.h
 *
 *  Created on: Jun 20, 2019
 *      Author: scott
 */

#ifndef PROCESSBPONTARGETDEVICE_H_
#define PROCESSBPONTARGETDEVICE_H_

#include "DetailedTimings.h"
#include "bpStereoParameters.h"
#include <math.h>
#include <chrono>

#define RUN_DETAILED_TIMING


template<typename T>
class ProcessBPOnTargetDevice {
public:
	ProcessBPOnTargetDevice() { }
	virtual ~ProcessBPOnTargetDevice() { }

		virtual int getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) = 0;

		virtual void allocateMemoryOnTargetDevice(void** arrayToAllocate, int numBytesAllocate) = 0;

		virtual void freeMemoryOnTargetDevice(void* arrayToFree) = 0;

		virtual void initializeDataCosts(float* image1PixelsCompDevice,
				float* image2PixelsCompDevice, T* dataCostDeviceCheckerboard1,
				T* dataCostDeviceCheckerboard2, BPsettings& algSettings) = 0;

		virtual void initializeDataCurrentLevel(T* dataCostStereoCheckerboard1,
				T* dataCostStereoCheckerboard2,
				T* dataCostDeviceToWriteToCheckerboard1,
				T* dataCostDeviceToWriteToCheckerboard2,
				int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
				int prevWidthLevelActualIntegerSize,
				int prevHeightLevelActualIntegerSize) = 0;

		virtual void initializeMessageValsToDefault(
				T* messageUDeviceSet0Checkerboard1,
				T* messageDDeviceSet0Checkerboard1,
				T* messageLDeviceSet0Checkerboard1,
				T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2,
				T* messageDDeviceSet0Checkerboard2,
				T* messageLDeviceSet0Checkerboard2,
				T* messageRDeviceSet0Checkerboard2, int widthLevelActualIntegerSize,
				int heightLevelActualIntegerSize, int totalPossibleMovements) = 0;

		virtual void runBPAtCurrentLevel(BPsettings& algSettings,
				int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
				T* messageUDeviceSet0Checkerboard1,
				T* messageDDeviceSet0Checkerboard1,
				T* messageLDeviceSet0Checkerboard1,
				T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2,
				T* messageDDeviceSet0Checkerboard2,
				T* messageLDeviceSet0Checkerboard2,
				T* messageRDeviceSet0Checkerboard2,
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2) = 0;

		virtual void copyMessageValuesToNextLevelDown(
				int prevWidthLevelActualIntegerSize,
				int prevHeightLevelActualIntegerSize,
				int widthLevelActualIntegerSize, int heightLevelActualIntegerSize,
				T* messageUDeviceSet0Checkerboard1,
				T* messageDDeviceSet0Checkerboard1,
				T* messageLDeviceSet0Checkerboard1,
				T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2,
				T* messageDDeviceSet0Checkerboard2,
				T* messageLDeviceSet0Checkerboard2,
				T* messageRDeviceSet0Checkerboard2,
				T** messageUDeviceSet1Checkerboard1,
				T** messageDDeviceSet1Checkerboard1,
				T** messageLDeviceSet1Checkerboard1,
				T** messageRDeviceSet1Checkerboard1,
				T** messageUDeviceSet1Checkerboard2,
				T** messageDDeviceSet1Checkerboard2,
				T** messageLDeviceSet1Checkerboard2,
				T** messageRDeviceSet1Checkerboard2) = 0;

		virtual void retrieveOutputDisparity(
				T* dataCostDeviceCurrentLevelCheckerboard1,
				T* dataCostDeviceCurrentLevelCheckerboard2,
				T* messageUDeviceSet0Checkerboard1,
				T* messageDDeviceSet0Checkerboard1,
				T* messageLDeviceSet0Checkerboard1,
				T* messageRDeviceSet0Checkerboard1,
				T* messageUDeviceSet0Checkerboard2,
				T* messageDDeviceSet0Checkerboard2,
				T* messageLDeviceSet0Checkerboard2,
				T* messageRDeviceSet0Checkerboard2,
				T* messageUDeviceSet1Checkerboard1,
				T* messageDDeviceSet1Checkerboard1,
				T* messageLDeviceSet1Checkerboard1,
				T* messageRDeviceSet1Checkerboard1,
				T* messageUDeviceSet1Checkerboard2,
				T* messageDDeviceSet1Checkerboard2,
				T* messageLDeviceSet1Checkerboard2,
				T* messageRDeviceSet1Checkerboard2,
				float* resultingDisparityMapCompDevice, int widthLevel,
				int heightLevel, int currentCheckerboardSet) = 0;

	//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
	//input is images image1Pixels and image1Pixels
	//output is resultingDisparityMap
		DetailedTimings* operator()(float* image1PixelsCompDevice, float* image2PixelsCompDevice, float* resultingDisparityMapCompDevice, BPsettings& algSettings)
		{

#ifdef RUN_DETAILED_TIMING

	DetailedTimings* timingsPointer = new DetailedTimings;
	//timeCopyDataKernelTotalTime = 0.0;
	//timeBpItersKernelTotalTime = 0.0;
	std::chrono::duration<double> diff;

#endif

			//printf("Start opt CPU\n");
			//retrieve the total number of possible movements; this is equal to the number of disparity values
			int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

			//start at the "bottom level" and work way up to determine amount of space needed to store data costs
			float widthLevel = (float)algSettings.widthImages;
			float heightLevel = (float)algSettings.heightImages;

			//store the "actual" integer size of the width and height of the level since it's not actually
			//possible to work with level with a decimal sizes...the portion of the last row/column is truncated
			//if the width/level size has a decimal
			int widthLevelActualIntegerSize = (int)roundf(widthLevel);
			int heightLevelActualIntegerSize = (int)roundf(heightLevel);

			int halfTotalDataAllLevels = 0;

			//compute "half" the total number of pixels in including every level of the "pyramid"
			//using "half" because the data is split in two using the checkerboard scheme
			for (int levelNum = 0; levelNum < algSettings.numLevels; levelNum++)
			{
				halfTotalDataAllLevels += (getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * (totalPossibleMovements);
				widthLevel /= 2.0f;
				heightLevel /= 2.0f;

				widthLevelActualIntegerSize = (int)ceil(widthLevel);
				heightLevelActualIntegerSize = (int)ceil(heightLevel);
			}

			//declare and then allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
			//each checkboard holds half of the data
			T* dataCostDeviceCheckerboard1; //checkerboard 1 includes the pixel in slot (0, 0)
			T* dataCostDeviceCheckerboard2;

			T* messageUDeviceCheckerboard1;
			T* messageDDeviceCheckerboard1;
			T* messageLDeviceCheckerboard1;
			T* messageRDeviceCheckerboard1;

			T* messageUDeviceCheckerboard2;
			T* messageDDeviceCheckerboard2;
			T* messageLDeviceCheckerboard2;
			T* messageRDeviceCheckerboard2;

#ifdef RUN_DETAILED_TIMING

	auto timeInitSettingsMallocStart = std::chrono::system_clock::now();

#endif

		#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			//printf("ALLOC ALL MEMORY\n");
			allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard1, 10*halfTotalDataAllLevels*sizeof(T));
			dataCostDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[1*(halfTotalDataAllLevels)]);

			messageUDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[2*(halfTotalDataAllLevels)]);
			messageDDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[3*(halfTotalDataAllLevels)]);
			messageLDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[4*(halfTotalDataAllLevels)]);
			messageRDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[5*(halfTotalDataAllLevels)]);

			messageUDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[6*(halfTotalDataAllLevels)]);
			messageDDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[7*(halfTotalDataAllLevels)]);
			messageLDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[8*(halfTotalDataAllLevels)]);
			messageRDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[9*(halfTotalDataAllLevels)]);

		#else

			allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard1, halfTotalDataAllLevels*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&dataCostDeviceCheckerboard2, halfTotalDataAllLevels*sizeof(T));

		#endif

#ifdef RUN_DETAILED_TIMING

	auto timeInitSettingsMallocEnd = std::chrono::system_clock::now();

	diff = timeInitSettingsMallocEnd-timeInitSettingsMallocStart;
	double totalTimeInitSettingsMallocStart = diff.count();

	auto timeInitDataCostsStart = std::chrono::system_clock::now();

#endif


			//now go "back to" the bottom level to initialize the data costs starting at the bottom level and going up the pyramid
			widthLevel = (float)algSettings.widthImages;
			heightLevel = (float)algSettings.heightImages;

			widthLevelActualIntegerSize = (int)roundf(widthLevel);
			heightLevelActualIntegerSize = (int)roundf(heightLevel);

			//printf("INIT DATA COSTS\n");
			//initialize the data cost at the bottom level
			initializeDataCosts(
					image1PixelsCompDevice, image2PixelsCompDevice,
					dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
					algSettings);
			//printf("DONE INIT DATA COSTS\n");

#ifdef RUN_DETAILED_TIMING

	auto timeInitDataCostsEnd = std::chrono::system_clock::now();
	diff = timeInitDataCostsEnd-timeInitDataCostsStart;

	double totalTimeGetDataCostsBottomLevel = diff.count();
	auto timeInitDataCostsHigherLevelsStart = std::chrono::system_clock::now();

#endif

			int offsetLevel = 0;

			//set the data costs at each level from the bottom level "up"
			for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
			{
				int prev_level_offset_level = offsetLevel;

				//width is half since each part of the checkboard contains half the values going across
				//retrieve offset where the data starts at the "current level"
				offsetLevel += (getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize)) * (heightLevelActualIntegerSize) * totalPossibleMovements;

				widthLevel /= 2.0f;
				heightLevel /= 2.0f;

				int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
				int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

				widthLevelActualIntegerSize = (int)ceil(widthLevel);
				heightLevelActualIntegerSize = (int)ceil(heightLevel);
				int widthCheckerboard = getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize);

				T* dataCostStereoCheckerboard1 =
						&dataCostDeviceCheckerboard1[prev_level_offset_level];
				T* dataCostStereoCheckerboard2 =
						&dataCostDeviceCheckerboard2[prev_level_offset_level];
				T* dataCostDeviceToWriteToCheckerboard1 =
						&dataCostDeviceCheckerboard1[offsetLevel];
				T* dataCostDeviceToWriteToCheckerboard2 =
						&dataCostDeviceCheckerboard2[offsetLevel];

				//printf("INIT DATA COSTS CURRENT LEVEL\n");
				initializeDataCurrentLevel(dataCostStereoCheckerboard1,
						dataCostStereoCheckerboard2, dataCostDeviceToWriteToCheckerboard1,
						dataCostDeviceToWriteToCheckerboard2,
						widthLevelActualIntegerSize,
						heightLevelActualIntegerSize, prevWidthLevelActualIntegerSize,
						prevHeightLevelActualIntegerSize);
				//printf("DONE INIT DATA COSTS CURRENT LEVEL\n");
			}

#ifdef RUN_DETAILED_TIMING

	auto timeInitDataCostsHigherLevelsEnd = std::chrono::system_clock::now();
	diff = timeInitDataCostsHigherLevelsEnd-timeInitDataCostsHigherLevelsStart;

	double totalTimeGetDataCostsHigherLevels = diff.count();

#endif

			//declare the space to pass the BP messages
			//need to have two "sets" of checkerboards because
			//the message values at the "higher" level in the image
			//pyramid need copied to a lower level without overwriting
			//values
			T* dataCostDeviceCurrentLevelCheckerboard1;
			T* dataCostDeviceCurrentLevelCheckerboard2;
			T* messageUDeviceSet0Checkerboard1;
			T* messageDDeviceSet0Checkerboard1;
			T* messageLDeviceSet0Checkerboard1;
			T* messageRDeviceSet0Checkerboard1;

			T* messageUDeviceSet0Checkerboard2;
			T* messageDDeviceSet0Checkerboard2;
			T* messageLDeviceSet0Checkerboard2;
			T* messageRDeviceSet0Checkerboard2;

			T* messageUDeviceSet1Checkerboard1;
			T* messageDDeviceSet1Checkerboard1;
			T* messageLDeviceSet1Checkerboard1;
			T* messageRDeviceSet1Checkerboard1;

			T* messageUDeviceSet1Checkerboard2;
			T* messageDDeviceSet1Checkerboard2;
			T* messageLDeviceSet1Checkerboard2;
			T* messageRDeviceSet1Checkerboard2;

#ifdef RUN_DETAILED_TIMING

	auto timeInitMessageValuesStart = std::chrono::system_clock::now();

#endif

			dataCostDeviceCurrentLevelCheckerboard1 = &dataCostDeviceCheckerboard1[offsetLevel];
			dataCostDeviceCurrentLevelCheckerboard2 = &dataCostDeviceCheckerboard2[offsetLevel];

		#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
			messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
			messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
			messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

			messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
			messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
			messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
			messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

		#else

			//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
			int numDataAndMessageSetInCheckerboardAtLevel = (getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize)) * heightLevelActualIntegerSize * totalPossibleMovements;

			//allocate the space for the message values in the first checkboard set at the current level
			allocateMemoryOnTargetDevice((void**)&messageUDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&messageDDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&messageLDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&messageRDeviceSet0Checkerboard1, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));

			allocateMemoryOnTargetDevice((void**)&messageUDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&messageDDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&messageLDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));
			allocateMemoryOnTargetDevice((void**)&messageRDeviceSet0Checkerboard2, numDataAndMessageSetInCheckerboardAtLevel*sizeof(T));

		#endif

#ifdef RUN_DETAILED_TIMING

	auto timeInitMessageValuesKernelTimeStart = std::chrono::system_clock::now();

#endif

			//printf("initializeMessageValsToDefault\n");
			//initialize all the BP message values at every pixel for every disparity to 0
			initializeMessageValsToDefault(messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
					messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
					widthLevelActualIntegerSize, heightLevelActualIntegerSize, totalPossibleMovements);
			//printf("DONE initializeMessageValsToDefault\n");

#ifdef RUN_DETAILED_TIMING

	auto timeInitMessageValuesKernelTimeEnd = std::chrono::system_clock::now();
	diff = timeInitMessageValuesKernelTimeEnd-timeInitMessageValuesKernelTimeStart;

	double totalTimeInitMessageValuesKernelTime = diff.count();

	auto timeInitMessageValuesEnd = std::chrono::system_clock::now();
	diff = timeInitMessageValuesEnd-timeInitMessageValuesStart;

	double totalTimeInitMessageVals = diff.count();

#endif

			//alternate between checkerboard sets 0 and 1
			int currentCheckerboardSet = 0;

#ifdef RUN_DETAILED_TIMING

	double totalTimeBpIters = 0.0;
	double totalTimeCopyData = 0.0;

#endif

			//run BP at each level in the "pyramid" starting on top and continuing to the bottom
			//where the final movement values are computed...the message values are passed from
			//the upper level to the lower levels; this pyramid methods causes the BP message values
			//to converge more quickly
			for (int levelNum = algSettings.numLevels - 1; levelNum >= 0; levelNum--)
			{

#ifdef RUN_DETAILED_TIMING

		auto timeBpIterStart = std::chrono::system_clock::now();

#endif
				//printf("LEVEL: %d\n", levelNum);
				//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
				if (currentCheckerboardSet == 0)
				{
					runBPAtCurrentLevel(algSettings,
							widthLevelActualIntegerSize, heightLevelActualIntegerSize,
							messageUDeviceSet0Checkerboard1,
							messageDDeviceSet0Checkerboard1,
							messageLDeviceSet0Checkerboard1,
							messageRDeviceSet0Checkerboard1,
							messageUDeviceSet0Checkerboard2,
							messageDDeviceSet0Checkerboard2,
							messageLDeviceSet0Checkerboard2,
							messageRDeviceSet0Checkerboard2,
							dataCostDeviceCurrentLevelCheckerboard1,
							dataCostDeviceCurrentLevelCheckerboard2);
				}
				else
				{
					runBPAtCurrentLevel(algSettings,
							widthLevelActualIntegerSize, heightLevelActualIntegerSize,
							messageUDeviceSet1Checkerboard1,
							messageDDeviceSet1Checkerboard1,
							messageLDeviceSet1Checkerboard1,
							messageRDeviceSet1Checkerboard1,
							messageUDeviceSet1Checkerboard2,
							messageDDeviceSet1Checkerboard2,
							messageLDeviceSet1Checkerboard2,
							messageRDeviceSet1Checkerboard2,
							dataCostDeviceCurrentLevelCheckerboard1,
							dataCostDeviceCurrentLevelCheckerboard2);
				}

#ifdef RUN_DETAILED_TIMING

		auto timeBpIterEnd = std::chrono::system_clock::now();
		diff = timeBpIterEnd-timeBpIterStart;

		totalTimeBpIters += diff.count();

		auto timeCopyMessageValuesStart = std::chrono::system_clock::now();

#endif

				//printf("DONE BP RUN\n");

				//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level
				if (levelNum > 0)
				{
					int prevWidthLevelActualIntegerSize = widthLevelActualIntegerSize;
					int prevHeightLevelActualIntegerSize = heightLevelActualIntegerSize;

					//the "next level" down has double the width and height of the current level
					widthLevel *= 2.0f;
					heightLevel *= 2.0f;

					widthLevelActualIntegerSize = (int)ceil(widthLevel);
					heightLevelActualIntegerSize = (int)ceil(heightLevel);
					int widthCheckerboard = getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize);

					offsetLevel -= widthCheckerboard * heightLevelActualIntegerSize * totalPossibleMovements;

					dataCostDeviceCurrentLevelCheckerboard1 = &dataCostDeviceCheckerboard1[offsetLevel];
					dataCostDeviceCurrentLevelCheckerboard2 = &dataCostDeviceCheckerboard2[offsetLevel];

					//bind messages in the current checkerboard set to the texture to copy to the "other" checkerboard set at the next level
					if (currentCheckerboardSet == 0)
					{

		#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

						messageUDeviceSet1Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
						messageDDeviceSet1Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
						messageLDeviceSet1Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
						messageRDeviceSet1Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

						messageUDeviceSet1Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
						messageDDeviceSet1Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
						messageLDeviceSet1Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
						messageRDeviceSet1Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

		#endif

						copyMessageValuesToNextLevelDown(
								prevWidthLevelActualIntegerSize,
								prevHeightLevelActualIntegerSize,
								widthLevelActualIntegerSize,
								heightLevelActualIntegerSize,
								messageUDeviceSet0Checkerboard1,
								messageDDeviceSet0Checkerboard1,
								messageLDeviceSet0Checkerboard1,
								messageRDeviceSet0Checkerboard1,
								messageUDeviceSet0Checkerboard2,
								messageDDeviceSet0Checkerboard2,
								messageLDeviceSet0Checkerboard2,
								messageRDeviceSet0Checkerboard2,
								(T**)&messageUDeviceSet1Checkerboard1,
								(T**)&messageDDeviceSet1Checkerboard1,
								(T**)&messageLDeviceSet1Checkerboard1,
								(T**)&messageRDeviceSet1Checkerboard1,
								(T**)&messageUDeviceSet1Checkerboard2,
								(T**)&messageDDeviceSet1Checkerboard2,
								(T**)&messageLDeviceSet1Checkerboard2,
								(T**)&messageRDeviceSet1Checkerboard2);


						currentCheckerboardSet = 1;
					}
					else
					{

		#ifdef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

						messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
						messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
						messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
						messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

						messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
						messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
						messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
						messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

		#endif

						copyMessageValuesToNextLevelDown(
								prevWidthLevelActualIntegerSize,
								prevHeightLevelActualIntegerSize,
								widthLevelActualIntegerSize,
								heightLevelActualIntegerSize,
								messageUDeviceSet1Checkerboard1,
								messageDDeviceSet1Checkerboard1,
								messageLDeviceSet1Checkerboard1,
								messageRDeviceSet1Checkerboard1,
								messageUDeviceSet1Checkerboard2,
								messageDDeviceSet1Checkerboard2,
								messageLDeviceSet1Checkerboard2,
								messageRDeviceSet1Checkerboard2,
								(T**)&messageUDeviceSet0Checkerboard1,
								(T**)&messageDDeviceSet0Checkerboard1,
								(T**)&messageLDeviceSet0Checkerboard1,
								(T**)&messageRDeviceSet0Checkerboard1,
								(T**)&messageUDeviceSet0Checkerboard2,
								(T**)&messageDDeviceSet0Checkerboard2,
								(T**)&messageLDeviceSet0Checkerboard2,
								(T**)&messageRDeviceSet0Checkerboard2);

						currentCheckerboardSet = 0;
					}
				}

#ifdef RUN_DETAILED_TIMING

		auto timeCopyMessageValuesEnd = std::chrono::system_clock::now();
		diff = timeCopyMessageValuesEnd-timeCopyMessageValuesStart;

		totalTimeCopyData += diff.count();

#endif

			}

#ifdef RUN_DETAILED_TIMING

	auto timeGetOutputDisparityStart = std::chrono::system_clock::now();

#endif

			retrieveOutputDisparity(dataCostDeviceCurrentLevelCheckerboard1, dataCostDeviceCurrentLevelCheckerboard2,
					messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
					messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
					messageUDeviceSet1Checkerboard1, messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, messageRDeviceSet1Checkerboard1,
					messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, messageLDeviceSet1Checkerboard2, messageRDeviceSet1Checkerboard2,
					resultingDisparityMapCompDevice, widthLevel, heightLevel, currentCheckerboardSet);

#ifdef RUN_DETAILED_TIMING

	auto timeGetOutputDisparityEnd = std::chrono::system_clock::now();
	diff = timeGetOutputDisparityEnd-timeGetOutputDisparityStart;

	double totalTimeGetOutputDisparity = diff.count();

	auto timeFinalUnbindFreeStart = std::chrono::system_clock::now();
	auto timeFinalFreeStart = std::chrono::system_clock::now();

#endif


		#ifndef USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT

			//printf("ALLOC MULT MEM SEGMENTS\n");

			//free the device storage for the message values used to retrieve the output movement values
			if (currentCheckerboardSet == 0)
			{
				//free device space allocated to message values
				freeMemoryOnTargetDevice((void*)messageUDeviceSet0Checkerboard1);
				freeMemoryOnTargetDevice((void*)messageDDeviceSet0Checkerboard1);
				freeMemoryOnTargetDevice((void*)messageLDeviceSet0Checkerboard1);
				freeMemoryOnTargetDevice((void*)messageRDeviceSet0Checkerboard1);

				freeMemoryOnTargetDevice((void*)messageUDeviceSet0Checkerboard2);
				freeMemoryOnTargetDevice((void*)messageDDeviceSet0Checkerboard2);
				freeMemoryOnTargetDevice((void*)messageLDeviceSet0Checkerboard2);
				freeMemoryOnTargetDevice((void*)messageRDeviceSet0Checkerboard2);
			}
			else
			{
				//free device space allocated to message values
				freeMemoryOnTargetDevice((void*)messageUDeviceSet1Checkerboard1);
				freeMemoryOnTargetDevice((void*)messageDDeviceSet1Checkerboard1);
				freeMemoryOnTargetDevice((void*)messageLDeviceSet1Checkerboard1);
				freeMemoryOnTargetDevice((void*)messageRDeviceSet1Checkerboard1);

				freeMemoryOnTargetDevice((void*)messageUDeviceSet1Checkerboard2);
				freeMemoryOnTargetDevice((void*)messageDDeviceSet1Checkerboard2);
				freeMemoryOnTargetDevice((void*)messageLDeviceSet1Checkerboard2);
				freeMemoryOnTargetDevice((void*)messageRDeviceSet1Checkerboard2);
			}

			//now free the allocated data space
			freeMemoryOnTargetDevice((void*)dataCostDeviceCheckerboard1);
			freeMemoryOnTargetDevice((void*)dataCostDeviceCheckerboard2);


		#else

			//now free the allocated data space
			freeMemoryOnTargetDevice(dataCostDeviceCheckerboard1);

		#endif

#ifdef RUN_DETAILED_TIMING

			double timeCopyDataKernelTotalTime = 0.0;
					double timeBpItersKernelTotalTime = 0.0;
	auto timeFinalUnbindFreeEnd = std::chrono::system_clock::now();
	auto timeFinalFreeEnd = std::chrono::system_clock::now();

	diff = timeFinalUnbindFreeEnd-timeFinalUnbindFreeStart;
	double totalTimeFinalUnbindFree = diff.count();

	diff = timeFinalFreeEnd-timeFinalFreeStart;
	double totalTimeFinalFree = diff.count();

	double totalMemoryProcessingTime =  0.0;/*totalTimeInitSettingsMallocStart + totalTimeFinalUnbindFree
			+ (totalTimeInitMessageVals - totalTimeInitMessageValuesKernelTime);
			//+ (totalTimeCopyData - timeCopyDataKernelTotalTime)
			//+ (timeBpItersKernelTotalTime - timeBpItersKernelTotalTime);*/
	double totalComputationProcessing = 0.0;/*totalTimeGetDataCostsBottomLevel
			+ totalTimeGetDataCostsHigherLevels
			+ totalTimeInitMessageValuesKernelTime + totalTimeCopyData
			+ timeBpItersKernelTotalTime + totalTimeGetOutputDisparity;*/
	double totalTimed = 0.0;/*totalTimeInitSettingsMallocStart
			+ totalTimeGetDataCostsBottomLevel
			+ totalTimeGetDataCostsHigherLevels + totalTimeInitMessageVals
			+ totalTimeBpIters + totalTimeCopyData + totalTimeGetOutputDisparity
			+ totalTimeFinalUnbindFree;*/
	timingsPointer->totalTimeInitSettingsMallocStart.push_back(
			totalTimeInitSettingsMallocStart);
	timingsPointer->totalTimeGetDataCostsBottomLevel.push_back(
			totalTimeGetDataCostsBottomLevel);
	timingsPointer->totalTimeGetDataCostsHigherLevels.push_back(
			totalTimeGetDataCostsHigherLevels);
	timingsPointer->totalTimeInitMessageVals.push_back(totalTimeInitMessageVals);
	timingsPointer->totalTimeInitMessageValuesKernelTime.push_back(totalTimeInitMessageValuesKernelTime);
	timingsPointer->totalTimeBpIters.push_back(totalTimeBpIters);
	timingsPointer->timeBpItersKernelTotalTime.push_back(timeBpItersKernelTotalTime);
	timingsPointer->totalTimeCopyData.push_back(totalTimeCopyData);
	timingsPointer->timeCopyDataKernelTotalTime.push_back(timeCopyDataKernelTotalTime);
	timingsPointer->totalTimeGetOutputDisparity.push_back(totalTimeGetOutputDisparity);
	timingsPointer->totalTimeFinalUnbindFree.push_back(totalTimeFinalUnbindFree);
	timingsPointer->totalTimeFinalFree.push_back(totalTimeFinalFree);
	timingsPointer->totalTimed.push_back(totalTimed);
	timingsPointer->totalMemoryProcessingTime.push_back(totalMemoryProcessingTime);
	timingsPointer->totalComputationProcessing.push_back(totalComputationProcessing);
	return timingsPointer;

#else

	return nullptr;

#endif
		}
};

#endif /* PROCESSBPONTARGETDEVICE_H_ */
