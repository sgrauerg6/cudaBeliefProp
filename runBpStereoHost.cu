/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

//Defines the functions to run the CUDA implementation of 2-D Stereo estimation using BP

#include "runBpStereoHostHeader.cuh"
#include <sys/time.h>

#define USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS
#define RUN_DETAILED_TIMING
#define USE_SAME_ARRAY_FOR_ALL_ALLOC

struct timeval timeCopyDataKernelStart;
		struct timeval timeCopyDataKernelEnd;
		double timeCopyDataKernelTotalTime = 0.0;

		struct timeval timeBpItersKernelStart;
				struct timeval timeBpItersKernelEnd;
				double timeBpItersKernelTotalTime = 0.0;

//functions directed related to running BP to retrieve the movement between the images

//set the current BP settings in the host in constant memory on the device
__host__ void setBPSettingInConstMem(BPsettings& currentBPSettings)
{
	//write BP settings to constant memory on the GPU
	(cudaMemcpyToSymbol(BPSettingsConstMemStereo, &currentBPSettings, sizeof(BPsettings)));
}

//run the given number of iterations of BP at the current level using the given message values in global device memory
__host__ void runBPAtCurrentLevel(int& numIterationsAtLevel, int& widthLevelActualIntegerSize, int& heightLevelActualIntegerSize, size_t& dataTexOffset,
	float*& messageUDeviceCheckerboard1, float*& messageDDeviceCheckerboard1, float*& messageLDeviceCheckerboard1, 
	float*& messageRDeviceCheckerboard1, float*& messageUDeviceCheckerboard2, float*& messageDDeviceCheckerboard2, float*& messageLDeviceCheckerboard2, 
	float*& messageRDeviceCheckerboard2, dim3& grid, dim3& threads, int& numBytesDataAndMessageSetInCheckerboardAtLevel, float* dataCostDeviceCheckerboard1,
	float* dataCostDeviceCheckerboard2)

{
	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < numIterationsAtLevel; iterationNum++)
	{
		if ((iterationNum % 2) == 0)
		{
			(cudaThreadSynchronize());

#ifdef RUN_DETAILED_TIMING
			gettimeofday(&timeBpItersKernelStart, NULL);
#endif

#ifdef USE_TEXTURES

			runBPIterationUsingCheckerboardUpdates <<<  grid, threads >>> (messageUDeviceCheckerboard2, messageDDeviceCheckerboard2, 
					messageLDeviceCheckerboard2, messageRDeviceCheckerboard2, widthLevelActualIntegerSize, heightLevelActualIntegerSize, iterationNum, ((int)dataTexOffset / sizeof(float)));

#else

			runBPIterationUsingCheckerboardUpdatesNoTextures <<<  grid, threads >>> (dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
					messageUDeviceCheckerboard1, messageDDeviceCheckerboard1, messageLDeviceCheckerboard1, messageRDeviceCheckerboard1,
					messageUDeviceCheckerboard2, messageDDeviceCheckerboard2,
					messageLDeviceCheckerboard2, messageRDeviceCheckerboard2, widthLevelActualIntegerSize, heightLevelActualIntegerSize, iterationNum, ((int)dataTexOffset / sizeof(float)));

#endif

			(cudaThreadSynchronize());

#ifdef RUN_DETAILED_TIMING

			gettimeofday(&timeBpItersKernelEnd, NULL);

							double timeStart = timeBpItersKernelStart.tv_sec
																		+ (timeBpItersKernelStart.tv_usec / 1000000.0);
														double timeEnd = timeBpItersKernelEnd.tv_sec
																		+ (timeBpItersKernelEnd.tv_usec / 1000000.0);

							timeBpItersKernelTotalTime += (timeEnd - timeStart);

#endif

#ifdef USE_TEXTURES
			cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
#endif
		}
		else
		{
			(cudaThreadSynchronize());

#ifdef RUN_DETAILED_TIMING

			gettimeofday(&timeBpItersKernelStart, NULL);

#endif

#ifdef USE_TEXTURES

			runBPIterationUsingCheckerboardUpdates <<<  grid, threads >>> (messageUDeviceCheckerboard1, messageDDeviceCheckerboard1, 
					messageLDeviceCheckerboard1, messageRDeviceCheckerboard1, widthLevelActualIntegerSize, heightLevelActualIntegerSize, iterationNum, ((int)dataTexOffset / sizeof(float)));

#else
			runBPIterationUsingCheckerboardUpdatesNoTextures <<<  grid, threads >>> (dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
					messageUDeviceCheckerboard2, messageDDeviceCheckerboard2, messageLDeviceCheckerboard2, messageRDeviceCheckerboard2,
					messageUDeviceCheckerboard1, messageDDeviceCheckerboard1,
					messageLDeviceCheckerboard1, messageRDeviceCheckerboard1, widthLevelActualIntegerSize, heightLevelActualIntegerSize, iterationNum, ((int)dataTexOffset / sizeof(float)));
#endif
			(cudaThreadSynchronize());

#ifdef RUN_DETAILED_TIMING

			gettimeofday(&timeBpItersKernelEnd, NULL);

										double timeStart = timeBpItersKernelStart.tv_sec
																					+ (timeBpItersKernelStart.tv_usec / 1000000.0);
																	double timeEnd = timeBpItersKernelEnd.tv_sec
																					+ (timeBpItersKernelEnd.tv_usec / 1000000.0);

										timeBpItersKernelTotalTime += (timeEnd - timeStart);

#endif

#ifdef USE_TEXTURES

			cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
			cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);

#endif
		}
	}
}



//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
__host__ void copyMessageValuesToNextLevelDown(int& widthLevelActualIntegerSize, int& heightLevelActualIntegerSize,
	float*& messageUDeviceCheckerboard1CopyFrom, float*& messageDDeviceCheckerboard1CopyFrom, float*& messageLDeviceCheckerboard1CopyFrom, 
	float*& messageRDeviceCheckerboard1CopyFrom, float*& messageUDeviceCheckerboard2CopyFrom, float*& messageDDeviceCheckerboard2CopyFrom, 
	float*& messageLDeviceCheckerboard2CopyFrom, float*& messageRDeviceCheckerboard2CopyFrom, float*& messageUDeviceCheckerboard1CopyTo, 
	float*& messageDDeviceCheckerboard1CopyTo, float*& messageLDeviceCheckerboard1CopyTo, float*& messageRDeviceCheckerboard1CopyTo, 
	float*& messageUDeviceCheckerboard2CopyTo, float*& messageDDeviceCheckerboard2CopyTo, float*& messageLDeviceCheckerboard2CopyTo, 
	float*& messageRDeviceCheckerboard2CopyTo, int& numBytesDataAndMessageSetInCheckerboardAtLevel, dim3& grid, dim3& threads)
{
#ifdef USE_TEXTURES

	//bind the linear memory storing to computed message values to copy from to a texture
	cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceCheckerboard1CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceCheckerboard1CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceCheckerboard1CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceCheckerboard1CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);

	cudaBindTexture(0, messageUPrevTexStereoCheckerboard2, messageUDeviceCheckerboard2CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDPrevTexStereoCheckerboard2, messageDDeviceCheckerboard2CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLPrevTexStereoCheckerboard2, messageLDeviceCheckerboard2CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRPrevTexStereoCheckerboard2, messageRDeviceCheckerboard2CopyFrom, numBytesDataAndMessageSetInCheckerboardAtLevel);
#endif

#ifndef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

	//allocate space in the GPU for the message values in the checkerboard set to copy to
	(cudaMalloc((void**) &messageUDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageDDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageLDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageRDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));

	(cudaMalloc((void**) &messageUDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageDDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageLDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageRDeviceCheckerboard2CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel));

#endif

	( cudaThreadSynchronize() );

#ifdef USE_TEXTURES

	gettimeofday(&timeCopyDataKernelStart, NULL);

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	copyPrevLevelToNextLevelBPCheckerboardStereo <<< grid, threads >>> (messageUDeviceCheckerboard1CopyTo, messageDDeviceCheckerboard1CopyTo, messageLDeviceCheckerboard1CopyTo, 
			messageRDeviceCheckerboard1CopyTo, messageUDeviceCheckerboard2CopyTo, messageDDeviceCheckerboard2CopyTo, messageLDeviceCheckerboard2CopyTo, 
			messageRDeviceCheckerboard2CopyTo, (widthLevelActualIntegerSize), (heightLevelActualIntegerSize), CHECKERBOARD_PART_1);

	( cudaThreadSynchronize() );

	copyPrevLevelToNextLevelBPCheckerboardStereo <<< grid, threads >>> (messageUDeviceCheckerboard1CopyTo, messageDDeviceCheckerboard1CopyTo, messageLDeviceCheckerboard1CopyTo, 
			messageRDeviceCheckerboard1CopyTo, messageUDeviceCheckerboard2CopyTo, messageDDeviceCheckerboard2CopyTo, messageLDeviceCheckerboard2CopyTo, 
			messageRDeviceCheckerboard2CopyTo, (widthLevelActualIntegerSize), (heightLevelActualIntegerSize), CHECKERBOARD_PART_2);

#else

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeCopyDataKernelStart, NULL);

#endif

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
		//storing the current message values
		copyPrevLevelToNextLevelBPCheckerboardStereoNoTextures <<< grid, threads >>> (messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
				messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
				messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
				messageUDeviceCheckerboard1CopyTo, messageDDeviceCheckerboard1CopyTo, messageLDeviceCheckerboard1CopyTo,
				messageRDeviceCheckerboard1CopyTo, messageUDeviceCheckerboard2CopyTo, messageDDeviceCheckerboard2CopyTo, messageLDeviceCheckerboard2CopyTo,
				messageRDeviceCheckerboard2CopyTo, (widthLevelActualIntegerSize), (heightLevelActualIntegerSize), CHECKERBOARD_PART_1);

		( cudaThreadSynchronize() );

		copyPrevLevelToNextLevelBPCheckerboardStereoNoTextures <<< grid, threads >>> (messageUDeviceCheckerboard1CopyFrom, messageDDeviceCheckerboard1CopyFrom,
				messageLDeviceCheckerboard1CopyFrom, messageRDeviceCheckerboard1CopyFrom, messageUDeviceCheckerboard2CopyFrom,
				messageDDeviceCheckerboard2CopyFrom, messageLDeviceCheckerboard2CopyFrom, messageRDeviceCheckerboard2CopyFrom,
				messageUDeviceCheckerboard1CopyTo, messageDDeviceCheckerboard1CopyTo, messageLDeviceCheckerboard1CopyTo,
				messageRDeviceCheckerboard1CopyTo, messageUDeviceCheckerboard2CopyTo, messageDDeviceCheckerboard2CopyTo, messageLDeviceCheckerboard2CopyTo,
				messageRDeviceCheckerboard2CopyTo, (widthLevelActualIntegerSize), (heightLevelActualIntegerSize), CHECKERBOARD_PART_2);

#endif
	( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeCopyDataKernelEnd, NULL);

				double timeStart = timeCopyDataKernelStart.tv_sec
															+ (timeCopyDataKernelStart.tv_usec / 1000000.0);
											double timeEnd = timeCopyDataKernelEnd.tv_sec
															+ (timeCopyDataKernelEnd.tv_usec / 1000000.0);

				timeCopyDataKernelTotalTime += (timeEnd - timeStart);

#endif

#ifndef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

	//free the now-copied from computed data of the completed level
	cudaFree(messageUDeviceCheckerboard1CopyFrom);
	cudaFree(messageDDeviceCheckerboard1CopyFrom);
	cudaFree(messageLDeviceCheckerboard1CopyFrom);
	cudaFree(messageRDeviceCheckerboard1CopyFrom);

	cudaFree(messageUDeviceCheckerboard2CopyFrom);
	cudaFree(messageDDeviceCheckerboard2CopyFrom);
	cudaFree(messageLDeviceCheckerboard2CopyFrom);
	cudaFree(messageRDeviceCheckerboard2CopyFrom);

#endif

#ifdef USE_TEXTURES

	//bind the newly written message data to the appropriate texture
	cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceCheckerboard1CopyTo, numBytesDataAndMessageSetInCheckerboardAtLevel);
#endif
}

//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
__host__ void initializeDataCosts(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& dataCostDeviceCheckerboard1, float*& dataCostDeviceCheckerboard2, BPsettings& algSettings)
{
	//allocate array and copy image data
	//data is in the single-float value format
	cudaChannelFormatDesc channelDescImages = cudaCreateChannelDesc<float>();

	//store the two image pixels in the GPU in a CUDA array
	cudaArray* cu_arrayImage1BP;
	cudaArray* cu_arrayImage2BP;

	//allocate and then copy the image pixel data for the two images on the GPU
	( cudaMallocArray( &cu_arrayImage1BP, &channelDescImages, algSettings.widthImages, algSettings.heightImages )); 
	( cudaMallocArray( &cu_arrayImage2BP, &channelDescImages, algSettings.widthImages, algSettings.heightImages )); 

	( cudaMemcpyToArray( cu_arrayImage1BP, 0, 0, image1PixelsDevice, algSettings.widthImages*algSettings.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));
	( cudaMemcpyToArray( cu_arrayImage2BP, 0, 0, image2PixelsDevice, algSettings.widthImages*algSettings.heightImages*sizeof(float), cudaMemcpyDeviceToDevice));

	// set texture parameters for the CUDA arrays to hold the input images
	image1PixelsTextureBPStereo.addressMode[0] = cudaAddressModeClamp;
	image1PixelsTextureBPStereo.addressMode[1] = cudaAddressModeClamp;
	image1PixelsTextureBPStereo.filterMode = cudaFilterModePoint;
	image1PixelsTextureBPStereo.normalized = false;    // access with normalized texture coordinates

	image2PixelsTextureBPStereo.addressMode[0] = cudaAddressModeClamp;
	image2PixelsTextureBPStereo.addressMode[1] = cudaAddressModeClamp;
	image2PixelsTextureBPStereo.filterMode = cudaFilterModePoint;
	image2PixelsTextureBPStereo.normalized = false;    // access with normalized texture coordinates

	//Bind the CUDA Arrays holding the input image pixel arrays to the appropriate texture
	( cudaBindTextureToArray( image1PixelsTextureBPStereo, cu_arrayImage1BP, channelDescImages));
	( cudaBindTextureToArray( image2PixelsTextureBPStereo, cu_arrayImage2BP, channelDescImages));

	//setup execution parameters
	//the thread size remains constant throughout but the grid size is adjusted based on the current level/kernal to run
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;


	//kernal run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
	grid.x = (unsigned int)ceil((float)algSettings.widthImages / (float)threads.x);
	grid.y = (unsigned int)ceil((float)algSettings.heightImages / (float)threads.y);

	//initialize the data the the "bottom" of the image pyramid
	initializeBottomLevelDataStereo <<< grid, threads >>> (dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2);

	( cudaThreadSynchronize() );

	//unbind the texture attached to the image pixel values
	cudaUnbindTexture( image1PixelsTextureBPStereo);
	cudaUnbindTexture( image2PixelsTextureBPStereo);

	//image data no longer needed after data costs are computed
	(cudaFreeArray(cu_arrayImage1BP));
	(cudaFreeArray(cu_arrayImage2BP));
}



//initialize the message values with no previous message values...all message values are set to 0
__host__ void initializeMessageValsToDefault(float*& messageUDeviceSet0Checkerboard1, float*& messageDDeviceSet0Checkerboard1, float*& messageLDeviceSet0Checkerboard1, float*& messageRDeviceSet0Checkerboard1,
												  float*& messageUDeviceSet0Checkerboard2, float*& messageDDeviceSet0Checkerboard2, float*& messageLDeviceSet0Checkerboard2, float*& messageRDeviceSet0Checkerboard2,
												  int widthOfCheckerboard, int heightOfCheckerboard, int numPossibleMovements)
{
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)widthOfCheckerboard / (float)threads.x), (unsigned int)ceil((float)heightOfCheckerboard / (float)threads.y));

	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	initializeMessageValsToDefault <<< grid, threads >>> (messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, 
												messageRDeviceSet0Checkerboard1, messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, 
												messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2, widthOfCheckerboard, heightOfCheckerboard);

	cudaThreadSynchronize();
}




//run the belief propagation algorithm with on a set of stereo images to generate a disparity map
//the input images image1PixelsDevice and image2PixelsDevice are stored in the global memory of the GPU
//the output movements resultingDisparityMapDevice is stored in the global memory of the GPU
__host__ void runBeliefPropStereoCUDA(float*& image1PixelsDevice, float*& image2PixelsDevice, float*& resultingDisparityMapDevice, BPsettings& algSettings)
{	
#ifdef RUN_DETAILED_TIMING

	timeCopyDataKernelTotalTime = 0.0;
	timeBpItersKernelTotalTime = 0.0;
	double timeStart;
	double timeEnd;
#endif

	//retrieve the total number of possible movements; this is equal to the number of disparity values 
	int totalPossibleMovements = NUM_POSSIBLE_DISPARITY_VALUES;

#ifdef RUN_DETAILED_TIMING

				struct timeval timeInitSettingsConstMemStart;
								struct timeval timeInitSettingsConstMemEnd;

								gettimeofday(&timeInitSettingsConstMemStart, NULL);
#endif

	//set the BP algorithm and extension settings on the device
	setBPSettingInConstMem(algSettings);

	( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeInitSettingsConstMemEnd, NULL);

	timeStart = timeInitSettingsConstMemStart.tv_sec
													+ (timeInitSettingsConstMemStart.tv_usec / 1000000.0);
									timeEnd = timeInitSettingsConstMemEnd.tv_sec
													+ (timeInitSettingsConstMemEnd.tv_usec / 1000000.0);


		double totalTimeInitSettingsConstMem = timeEnd - timeStart;

#endif


	//setup execution parameters
	//the thread size remains constant throughout but the grid size is adjusted based on the current level/kernal to run
	dim3 threads(BLOCK_SIZE_WIDTH_BP, BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//start at the "bottom level" and word way up to determine amount of space needed to store data costs
	float widthLevel = (float)algSettings.widthImages;
	float heightLevel = (float)algSettings.heightImages;

	//store the "actual" integer size of the width and height of the level since it's not actually
	//possible to work with level with a decimal sizes...the portion of the last row/column is truncated
	//if the width/level size has a decimal
	int widthLevelActualIntegerSize = (int)floor(widthLevel);
	int heightLevelActualIntegerSize = (int)floor(heightLevel);

	int halfTotalDataAllLevels = 0;

	//compute "half" the total number of pixels in including every level of the "pyramid"
	//using "half" because the data is split in two using the checkerboard scheme
	for (int levelNum = 0; levelNum < algSettings.numLevels; levelNum++)
	{
		halfTotalDataAllLevels += (widthLevelActualIntegerSize/2)*(heightLevelActualIntegerSize);
		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		widthLevelActualIntegerSize = (int)floor(widthLevel);
		heightLevelActualIntegerSize = (int)floor(heightLevel);
	}

	//declare and then allocate the space on the device to store the data cost component at each possible movement at each level of the "pyramid"
	//each checkboard holds half of the data
	float* dataCostDeviceCheckerboard1; //checkerboard 1 includes the pixel in slot (0, 0)
	float* dataCostDeviceCheckerboard2;

#ifdef RUN_DETAILED_TIMING

	struct timeval timeInitSettingsMallocStart;
				struct timeval timeInitSettingsMallocEnd;
				gettimeofday(&timeInitSettingsMallocStart, NULL);
#endif

#ifdef USE_SAME_ARRAY_FOR_ALL_ALLOC

				printf("ALLOC ALL MEMORY\n");
	(cudaMalloc((void**) &dataCostDeviceCheckerboard1, 10*(halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	dataCostDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[1*(halfTotalDataAllLevels)*totalPossibleMovements]);

	float* messageUDeviceCheckerboard1;
		float* messageDDeviceCheckerboard1;
		float* messageLDeviceCheckerboard1;
		float* messageRDeviceCheckerboard1;

		float* messageUDeviceCheckerboard2;
		float* messageDDeviceCheckerboard2;
		float* messageLDeviceCheckerboard2;
		float* messageRDeviceCheckerboard2;

		messageUDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[2*(halfTotalDataAllLevels)*totalPossibleMovements]);
		messageDDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[3*(halfTotalDataAllLevels)*totalPossibleMovements]);
		messageLDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[4*(halfTotalDataAllLevels)*totalPossibleMovements]);
		messageRDeviceCheckerboard1 = &(dataCostDeviceCheckerboard1[5*(halfTotalDataAllLevels)*totalPossibleMovements]);

		messageUDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[6*(halfTotalDataAllLevels)*totalPossibleMovements]);
		messageDDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[7*(halfTotalDataAllLevels)*totalPossibleMovements]);
		messageLDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[8*(halfTotalDataAllLevels)*totalPossibleMovements]);
		messageRDeviceCheckerboard2 = &(dataCostDeviceCheckerboard1[9*(halfTotalDataAllLevels)*totalPossibleMovements]);

#else

	(cudaMalloc((void**) &dataCostDeviceCheckerboard1, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &dataCostDeviceCheckerboard2, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));

#ifdef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

	float* messageUDeviceCheckerboard1;
	float* messageDDeviceCheckerboard1;
	float* messageLDeviceCheckerboard1;
	float* messageRDeviceCheckerboard1;

	float* messageUDeviceCheckerboard2;
	float* messageDDeviceCheckerboard2;
	float* messageLDeviceCheckerboard2;
	float* messageRDeviceCheckerboard2;

	(cudaMalloc((void**) &messageUDeviceCheckerboard1, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &messageDDeviceCheckerboard1, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &messageLDeviceCheckerboard1, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &messageRDeviceCheckerboard1, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));

	(cudaMalloc((void**) &messageUDeviceCheckerboard2, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &messageDDeviceCheckerboard2, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &messageLDeviceCheckerboard2, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));
	(cudaMalloc((void**) &messageRDeviceCheckerboard2, (halfTotalDataAllLevels)*totalPossibleMovements*sizeof(float)));

#endif

#endif

	( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeInitSettingsMallocEnd, NULL);

	timeStart = timeInitSettingsMallocStart.tv_sec
												+ (timeInitSettingsMallocStart.tv_usec / 1000000.0);
								timeEnd = timeInitSettingsMallocEnd.tv_sec
												+ (timeInitSettingsMallocEnd.tv_usec / 1000000.0);


	double totalTimeInitSettingsMallocStart = timeEnd - timeStart;

	struct timeval timeInitDataCostsStart;
			struct timeval timeInitDataCostsEnd;


	gettimeofday(&timeInitDataCostsStart, NULL);

#endif

	//now go "back to" the bottom level to initialize the data costs starting at the bottom level and going up the pyramid
		widthLevel = (float)algSettings.widthImages;
		heightLevel = (float)algSettings.heightImages;

		widthLevelActualIntegerSize = (int)floor(widthLevel);
		heightLevelActualIntegerSize = (int)floor(heightLevel);


	//initialize the data cost at the bottom level 
	initializeDataCosts(image1PixelsDevice, image2PixelsDevice, dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2, algSettings);

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeInitDataCostsEnd, NULL);

	timeStart = timeInitDataCostsStart.tv_sec
							+ (timeInitDataCostsStart.tv_usec / 1000000.0);
			timeEnd = timeInitDataCostsEnd.tv_sec
							+ (timeInitDataCostsEnd.tv_usec / 1000000.0);

	double totalTimeGetDataCostsBottomLevel = timeEnd - timeStart;

#endif

		
	int offsetLevel = 0;

	//stores the number of bytes for the data costs and one set of message values in each of the two "checkerboards" at the current level
	//this is half the total number of bytes for the data/message info at the level, since there are two equal-sized checkerboards
	//initially at "bottom level" of width widthImages and height heightImages
	int numBytesDataAndMessageSetInCheckerboardAtLevel = (widthLevelActualIntegerSize/2)*(heightLevelActualIntegerSize)*totalPossibleMovements*sizeof(float);

#ifdef RUN_DETAILED_TIMING

	struct timeval timeInitDataCostsHigherLevelsStart;
		struct timeval timeInitDataCostsHigherLevelsEnd;

	gettimeofday(&timeInitDataCostsHigherLevelsStart, NULL);

#endif

	//set the data costs at each level from the bottom level "up"
	for (int levelNum = 1; levelNum < algSettings.numLevels; levelNum++)
	{
#ifdef USE_TEXTURES

		//need to have "offset" to deal with hardware alignment requirement for textures
		size_t offsetNum; 
		cudaBindTexture(&offsetNum, dataCostTexStereoCheckerboard1, &dataCostDeviceCheckerboard1[offsetLevel], numBytesDataAndMessageSetInCheckerboardAtLevel);
		cudaBindTexture(&offsetNum, dataCostTexStereoCheckerboard2, &dataCostDeviceCheckerboard2[offsetLevel], numBytesDataAndMessageSetInCheckerboardAtLevel);
#endif
		int prev_level_offset_level = offsetLevel;

		//width is half since each part of the checkboard contains half the values going across
		//retrieve offset where the data starts at the "current level"
		offsetLevel += (widthLevelActualIntegerSize / 2) *(heightLevelActualIntegerSize)*totalPossibleMovements; 

		widthLevel /= 2.0f;
		heightLevel /= 2.0f;

		widthLevelActualIntegerSize = (int)floor(widthLevel);
		heightLevelActualIntegerSize = (int)floor(heightLevel);

		//printf("LevelNum: %d  Width: %d  Height: %d \n", levelNum, widthLevelActualIntegerSize, heightLevelActualIntegerSize);

		//each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
		//the four-connected neighbors are in the other checkerboard
		grid.x = (unsigned int)ceil((float)(widthLevel / 2.0f) / (float)threads.x);
		grid.y = (unsigned int)ceil((float)heightLevel / (float)threads.y);

#ifdef USE_TEXTURES

		//initialize the data costs for the "next level" of the pyramid
		//the "next level" starts at the calculated offsetLevel
		initializeCurrentLevelDataStereo <<< grid, threads >>> (&dataCostDeviceCheckerboard1[offsetLevel], widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_1, ((int)offsetNum/sizeof(float)));
		
		( cudaThreadSynchronize() );
		
		initializeCurrentLevelDataStereo <<< grid, threads >>> (&dataCostDeviceCheckerboard2[offsetLevel], widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_2, ((int)offsetNum/sizeof(float)));

#else
		size_t offsetNum = 0;

		initializeCurrentLevelDataStereoNoTextures <<< grid, threads >>> (&dataCostDeviceCheckerboard1[prev_level_offset_level], &dataCostDeviceCheckerboard2[prev_level_offset_level], &dataCostDeviceCheckerboard1[offsetLevel], widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_1, ((int)offsetNum/sizeof(float)));

		( cudaThreadSynchronize() );

		initializeCurrentLevelDataStereoNoTextures <<< grid, threads >>> (&dataCostDeviceCheckerboard1[prev_level_offset_level], &dataCostDeviceCheckerboard2[prev_level_offset_level], &dataCostDeviceCheckerboard2[offsetLevel], widthLevelActualIntegerSize, heightLevelActualIntegerSize, CHECKERBOARD_PART_2, ((int)offsetNum/sizeof(float)));

#endif
		( cudaThreadSynchronize() );

#ifdef USE_TEXTURES

		( cudaUnbindTexture( dataCostTexStereoCheckerboard1));	
		( cudaUnbindTexture( dataCostTexStereoCheckerboard2));	

#endif

		//update number of bytes of data and message cost if not at bottom level
		if (levelNum < (algSettings.numLevels - 1))
		{
			//each "checkerboard" where the computation alternates contains half the data
			numBytesDataAndMessageSetInCheckerboardAtLevel = (widthLevelActualIntegerSize/2)*(heightLevelActualIntegerSize)*totalPossibleMovements*sizeof(float);
		}
	}

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeInitDataCostsHigherLevelsEnd, NULL);
	
	timeStart = timeInitDataCostsHigherLevelsStart.tv_sec
						+ (timeInitDataCostsHigherLevelsStart.tv_usec / 1000000.0);
		timeEnd = timeInitDataCostsHigherLevelsEnd.tv_sec
						+ (timeInitDataCostsHigherLevelsEnd.tv_usec / 1000000.0);

		double totalTimeGetDataCostsHigherLevels = timeEnd - timeStart;

#endif

		( cudaThreadSynchronize() );

	//declare the space to pass the BP messages
	//need to have two "sets" of checkerboards because
	//the message values at the "higher" level in the image
	//pyramid need copied to a lower level without overwriting
	//values
	float* messageUDeviceSet0Checkerboard1;
	float* messageDDeviceSet0Checkerboard1;
	float* messageLDeviceSet0Checkerboard1;
	float* messageRDeviceSet0Checkerboard1;

	float* messageUDeviceSet0Checkerboard2;
	float* messageDDeviceSet0Checkerboard2;
	float* messageLDeviceSet0Checkerboard2;
	float* messageRDeviceSet0Checkerboard2;

	float* messageUDeviceSet1Checkerboard1;
	float* messageDDeviceSet1Checkerboard1;
	float* messageLDeviceSet1Checkerboard1;
	float* messageRDeviceSet1Checkerboard1;

	float* messageUDeviceSet1Checkerboard2;
	float* messageDDeviceSet1Checkerboard2;
	float* messageLDeviceSet1Checkerboard2;
	float* messageRDeviceSet1Checkerboard2;

#ifdef RUN_DETAILED_TIMING

	struct timeval timeInitMessageValuesStart;
			struct timeval timeInitMessageValuesEnd;

		gettimeofday(&timeInitMessageValuesStart, NULL);

#endif

#ifdef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

	messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
	messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
	messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
	messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

	messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
	messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
	messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
	messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#else

	//allocate the space for the message values in the first checkboard set at the current level
	(cudaMalloc((void**) &messageUDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageDDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageLDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageRDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel));

	(cudaMalloc((void**) &messageUDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageDDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageLDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));
	(cudaMalloc((void**) &messageRDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel));

#endif

	//retrieve the number of bytes needed to store the data cost/each set of messages in the checkerboard
	numBytesDataAndMessageSetInCheckerboardAtLevel = (widthLevelActualIntegerSize/2)*(heightLevelActualIntegerSize)*totalPossibleMovements*sizeof(float);

	//initialize all the BP message values at every pixel for every disparity to 0
	initializeMessageValsToDefault(messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
											messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
											widthLevelActualIntegerSize / 2, heightLevelActualIntegerSize, totalPossibleMovements);

#ifdef USE_TEXTURES

		cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
		cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
		cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
		cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
#endif
		( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

		gettimeofday(&timeInitMessageValuesEnd, NULL);

		timeStart = timeInitMessageValuesStart.tv_sec
										+ (timeInitMessageValuesStart.tv_usec / 1000000.0);
						timeEnd = timeInitMessageValuesEnd.tv_sec
										+ (timeInitMessageValuesEnd.tv_usec / 1000000.0);

						double totalTimeInitMessageVals = (timeEnd - timeStart);

#endif

	//alternate between checkerboard sets 0 and 1
	int currentCheckerboardSet = 0;

#ifdef RUN_DETAILED_TIMING

	double totalTimeBpIters = 0.0;
	double totalTimeCopyData = 0.0;
	struct timeval timeBpIterStart;
	struct timeval timeBpIterEnd;
	struct timeval timeCopyMessageValuesStart;
	struct timeval timeCopyMessageValuesEnd;

#endif


	//run BP at each level in the "pyramid" starting on top and continuing to the bottom
	//where the final movement values are computed...the message values are passed from
	//the upper level to the lower levels; this pyramid methods causes the BP message values
	//to converge more quickly
	for (int levelNum = algSettings.numLevels - 1; levelNum >= 0; levelNum--)
	{
		//offset needed because of alignment requirement for textures
		size_t offset = 0;
		( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

		gettimeofday(&timeBpIterStart, NULL);

#endif

#ifdef USE_TEXTURES

		//bind the portion of the data cost "pyramid" for the current level to a texture
		cudaBindTexture(&offset, dataCostTexStereoCheckerboard1, &dataCostDeviceCheckerboard1[offsetLevel], numBytesDataAndMessageSetInCheckerboardAtLevel);
		cudaBindTexture(&offset, dataCostTexStereoCheckerboard2, &dataCostDeviceCheckerboard2[offsetLevel], numBytesDataAndMessageSetInCheckerboardAtLevel);

#endif

		//printf("LevelNumBP: %d  Width: %f  Height: %f \n", levelNum, widthLevel, heightLevel);

		if (levelNum >= 0)
		{
			grid.x = (unsigned int)ceil((float)(widthLevel/2.0f) / (float)threads.x); //only updating half at a time
			grid.y = (unsigned int)ceil((float)heightLevel / (float)threads.y);

			//need to alternate which checkerboard set to work on since copying from one to the other...need to avoid read-write conflict when copying in parallel
			if (currentCheckerboardSet == 0)
			{
				runBPAtCurrentLevel(algSettings.numIterations, widthLevelActualIntegerSize, heightLevelActualIntegerSize, offset,
					messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, 
					messageRDeviceSet0Checkerboard1, messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, 
					messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2, grid, threads, numBytesDataAndMessageSetInCheckerboardAtLevel,
					&dataCostDeviceCheckerboard1[offsetLevel], &dataCostDeviceCheckerboard2[offsetLevel]);
			}
			else
			{
				runBPAtCurrentLevel(algSettings.numIterations, widthLevelActualIntegerSize, heightLevelActualIntegerSize, offset,
					messageUDeviceSet1Checkerboard1, messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, 
					messageRDeviceSet1Checkerboard1, messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, 
					messageLDeviceSet1Checkerboard2, messageRDeviceSet1Checkerboard2, grid, threads, numBytesDataAndMessageSetInCheckerboardAtLevel,
					&dataCostDeviceCheckerboard1[offsetLevel], &dataCostDeviceCheckerboard2[offsetLevel]);
			}

			( cudaThreadSynchronize() );
		}

#ifdef RUN_DETAILED_TIMING

		gettimeofday(&timeBpIterEnd, NULL);

		timeStart = timeBpIterStart.tv_sec
								+ (timeBpIterStart.tv_usec / 1000000.0);
				timeEnd = timeBpIterEnd.tv_sec
								+ (timeBpIterEnd.tv_usec / 1000000.0);

				totalTimeBpIters += (timeEnd - timeStart);

		
		gettimeofday(&timeCopyMessageValuesStart, NULL);

#endif

		//if not at the "bottom level" copy the current message values at the current level to the corresponding slots next level 
		if (levelNum > 0)
		{	
			//the "next level" down has double the width and height of the current level
			widthLevel *= 2.0f;
			heightLevel *= 2.0f;

			widthLevelActualIntegerSize = (int)floor(widthLevel);
			heightLevelActualIntegerSize = (int)floor(heightLevel);

			offsetLevel -= ((widthLevelActualIntegerSize/2)*(heightLevelActualIntegerSize)*totalPossibleMovements);

			//update the number of bytes needed to store each set
			numBytesDataAndMessageSetInCheckerboardAtLevel = (widthLevelActualIntegerSize/2)*(heightLevelActualIntegerSize)*totalPossibleMovements*sizeof(float);

			grid.x = (unsigned int)ceil((float)(widthLevel / 4.0f) / (float)threads.x);
			grid.y = (unsigned int)ceil((float)(heightLevel / 2.0f) / (float)threads.y);

			//bind messages in the current checkerboard set to the texture to copy to the "other" checkerboard set at the next level 
			if (currentCheckerboardSet == 0)
			{

#ifdef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

				messageUDeviceSet1Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
					messageDDeviceSet1Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
					messageLDeviceSet1Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
					messageRDeviceSet1Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

					messageUDeviceSet1Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
					messageDDeviceSet1Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
					messageLDeviceSet1Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
					messageRDeviceSet1Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#endif

				copyMessageValuesToNextLevelDown(widthLevelActualIntegerSize, heightLevelActualIntegerSize,
					messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, 
					messageRDeviceSet0Checkerboard1, messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, 
					messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2, messageUDeviceSet1Checkerboard1, 
					messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, messageRDeviceSet1Checkerboard1, 
					messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, messageLDeviceSet1Checkerboard2, 
					messageRDeviceSet1Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel, grid, threads);

				currentCheckerboardSet = 1;
			}
			else
			{

#ifdef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

				messageUDeviceSet0Checkerboard1 = &messageUDeviceCheckerboard1[offsetLevel];
					messageDDeviceSet0Checkerboard1 = &messageDDeviceCheckerboard1[offsetLevel];
					messageLDeviceSet0Checkerboard1 = &messageLDeviceCheckerboard1[offsetLevel];
					messageRDeviceSet0Checkerboard1 = &messageRDeviceCheckerboard1[offsetLevel];

					messageUDeviceSet0Checkerboard2 = &messageUDeviceCheckerboard2[offsetLevel];
					messageDDeviceSet0Checkerboard2 = &messageDDeviceCheckerboard2[offsetLevel];
					messageLDeviceSet0Checkerboard2 = &messageLDeviceCheckerboard2[offsetLevel];
					messageRDeviceSet0Checkerboard2 = &messageRDeviceCheckerboard2[offsetLevel];

#endif

				copyMessageValuesToNextLevelDown(widthLevelActualIntegerSize, heightLevelActualIntegerSize,
					messageUDeviceSet1Checkerboard1, messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, 
					messageRDeviceSet1Checkerboard1, messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, 
					messageLDeviceSet1Checkerboard2, messageRDeviceSet1Checkerboard2, messageUDeviceSet0Checkerboard1, 
					messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1, 
					messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, 
					messageRDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel, grid, threads);

				currentCheckerboardSet = 0;
			}
		}
		//otherwise in "bottom level"; use message values and data costs to retrieve final movement values
		else
		{
#ifdef USE_TEXTURES
			if (currentCheckerboardSet == 0)
			{
				cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceSet0Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);

				cudaBindTexture(0, messageUPrevTexStereoCheckerboard2, messageUDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageDPrevTexStereoCheckerboard2, messageDDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageLPrevTexStereoCheckerboard2, messageLDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageRPrevTexStereoCheckerboard2, messageRDeviceSet0Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
			}
			else
			{
				cudaBindTexture(0, messageUPrevTexStereoCheckerboard1, messageUDeviceSet1Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageDPrevTexStereoCheckerboard1, messageDDeviceSet1Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageLPrevTexStereoCheckerboard1, messageLDeviceSet1Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageRPrevTexStereoCheckerboard1, messageRDeviceSet1Checkerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);

				cudaBindTexture(0, messageUPrevTexStereoCheckerboard2, messageUDeviceSet1Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageDPrevTexStereoCheckerboard2, messageDDeviceSet1Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageLPrevTexStereoCheckerboard2, messageLDeviceSet1Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
				cudaBindTexture(0, messageRPrevTexStereoCheckerboard2, messageRDeviceSet1Checkerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
			}
#endif
		}
		( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

		gettimeofday(&timeCopyMessageValuesEnd, NULL);

		timeStart = timeCopyMessageValuesStart.tv_sec
										+ (timeCopyMessageValuesStart.tv_usec / 1000000.0);
						timeEnd = timeCopyMessageValuesEnd.tv_sec
										+ (timeCopyMessageValuesEnd.tv_usec / 1000000.0);

						totalTimeCopyData += (timeEnd - timeStart);

#endif

	}


#ifdef USE_TEXTURES
		
	cudaUnbindTexture(dataCostTexStereoCheckerboard1);
	cudaUnbindTexture(dataCostTexStereoCheckerboard2);

	//bind the data costs at the "bottom level" of the image pyramid
	cudaBindTexture(0, dataCostTexStereoCheckerboard1, dataCostDeviceCheckerboard1, numBytesDataAndMessageSetInCheckerboardAtLevel);
	cudaBindTexture(0, dataCostTexStereoCheckerboard2, dataCostDeviceCheckerboard2, numBytesDataAndMessageSetInCheckerboardAtLevel);
#endif
	//printf("Final  Width: %d  Height: %d \n", widthLevelActualIntegerSize, heightLevelActualIntegerSize);

#ifdef RUN_DETAILED_TIMING

	struct timeval timeGetOutputDisparityStart;
		struct timeval timeGetOutputDisparityEnd;

	gettimeofday(&timeGetOutputDisparityStart, NULL);

#endif

	grid.x = (unsigned int)ceil((float)widthLevel / (float)threads.x);
	grid.y = (unsigned int)ceil((float)heightLevel / (float)threads.y);
#ifdef USE_TEXTURES

	//retrieve the output disparity/movement values
	retrieveOutputDisparityCheckerboardStereo <<< grid, threads >>> (resultingDisparityMapDevice, widthLevelActualIntegerSize, heightLevelActualIntegerSize);

#else

	if (currentCheckerboardSet == 0)
	{
	retrieveOutputDisparityCheckerboardStereoNoTextures <<< grid, threads >>> (dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
			messageUDeviceSet0Checkerboard1, messageDDeviceSet0Checkerboard1, messageLDeviceSet0Checkerboard1, messageRDeviceSet0Checkerboard1,
			messageUDeviceSet0Checkerboard2, messageDDeviceSet0Checkerboard2, messageLDeviceSet0Checkerboard2, messageRDeviceSet0Checkerboard2,
			resultingDisparityMapDevice, widthLevelActualIntegerSize, heightLevelActualIntegerSize);
	}
	else
	{
		retrieveOutputDisparityCheckerboardStereoNoTextures <<< grid, threads >>> (dataCostDeviceCheckerboard1, dataCostDeviceCheckerboard2,
				messageUDeviceSet1Checkerboard1, messageDDeviceSet1Checkerboard1, messageLDeviceSet1Checkerboard1, messageRDeviceSet1Checkerboard1,
				messageUDeviceSet1Checkerboard2, messageDDeviceSet1Checkerboard2, messageLDeviceSet1Checkerboard2, messageRDeviceSet1Checkerboard2,
				resultingDisparityMapDevice, widthLevelActualIntegerSize, heightLevelActualIntegerSize);

	}

#endif
	( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeGetOutputDisparityEnd, NULL);

	timeStart = timeGetOutputDisparityStart.tv_sec
											+ (timeGetOutputDisparityStart.tv_usec / 1000000.0);
							timeEnd = timeGetOutputDisparityEnd.tv_sec
											+ (timeGetOutputDisparityEnd.tv_usec / 1000000.0);

	double totalTimeGetOutputDisparity = timeEnd - timeStart;

	struct timeval timeFinalUnbindFreeStart;
			struct timeval timeFinalUnbindFreeEnd;
			struct timeval timeFinalUnbindStart;
			struct timeval timeFinalUnbindEnd;
			struct timeval timeFinalFreeStart;
			struct timeval timeFinalFreeEnd;
			gettimeofday(&timeFinalUnbindFreeStart, NULL);

#endif

#ifdef USE_TEXTURES

#ifdef RUN_DETAILED_TIMING

			gettimeofday(&timeFinalUnbindStart, NULL);

#endif

	//textures for message values no longer needed after output disparity/movement found
	cudaUnbindTexture( messageUPrevTexStereoCheckerboard1);
	cudaUnbindTexture( messageDPrevTexStereoCheckerboard1);
	cudaUnbindTexture( messageLPrevTexStereoCheckerboard1);
	cudaUnbindTexture( messageRPrevTexStereoCheckerboard1);

	cudaUnbindTexture( messageUPrevTexStereoCheckerboard2);
	cudaUnbindTexture( messageDPrevTexStereoCheckerboard2);
	cudaUnbindTexture( messageLPrevTexStereoCheckerboard2);
	cudaUnbindTexture( messageRPrevTexStereoCheckerboard2);

	//unbind the texture attached to the data costs
	cudaUnbindTexture(dataCostTexStereoCheckerboard1);
	cudaUnbindTexture(dataCostTexStereoCheckerboard2);
	( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeFinalUnbindEnd, NULL);

#endif

#endif

#ifdef RUN_DETAILED_TIMING

	timeStart = timeFinalUnbindStart.tv_sec
						+ (timeFinalUnbindStart.tv_usec / 1000000.0);
			timeEnd = timeFinalUnbindEnd.tv_sec
						+ (timeFinalUnbindEnd.tv_usec / 1000000.0);

	double totalTimeFinalUnbind = timeEnd - timeStart;

	gettimeofday(&timeFinalFreeStart, NULL);

#endif

#ifdef USE_SAME_ARRAY_FOR_ALL_ALLOC

#else

#ifdef USE_SAME_ARRAY_FOR_ALL_LEVEL_MESSAGE_VALS

	cudaFree(messageUDeviceCheckerboard1);
	cudaFree(messageDDeviceCheckerboard1);
	cudaFree(messageLDeviceCheckerboard1);
	cudaFree(messageRDeviceCheckerboard1);

	cudaFree(messageUDeviceCheckerboard2);
	cudaFree(messageDDeviceCheckerboard2);
	cudaFree(messageLDeviceCheckerboard2);
	cudaFree(messageRDeviceCheckerboard2);

#else

	//free the device storage for the message values used to retrieve the output movement values
	if (currentCheckerboardSet == 0)
	{
		//free device space allocated to message values
		cudaFree(messageUDeviceSet0Checkerboard1);
		cudaFree(messageDDeviceSet0Checkerboard1);
		cudaFree(messageLDeviceSet0Checkerboard1);
		cudaFree(messageRDeviceSet0Checkerboard1);

		cudaFree(messageUDeviceSet0Checkerboard2);
		cudaFree(messageDDeviceSet0Checkerboard2);
		cudaFree(messageLDeviceSet0Checkerboard2);
		cudaFree(messageRDeviceSet0Checkerboard2);
	}
	else
	{
		//free device space allocated to message values
		cudaFree(messageUDeviceSet1Checkerboard1);
		cudaFree(messageDDeviceSet1Checkerboard1);
		cudaFree(messageLDeviceSet1Checkerboard1);
		cudaFree(messageRDeviceSet1Checkerboard1);

		cudaFree(messageUDeviceSet1Checkerboard2);
		cudaFree(messageDDeviceSet1Checkerboard2);
		cudaFree(messageLDeviceSet1Checkerboard2);
		cudaFree(messageRDeviceSet1Checkerboard2);
	}

#endif
#endif

#ifdef USE_SAME_ARRAY_FOR_ALL_ALLOC
	printf("FREE ALL MEMORY\n");

	cudaFree(dataCostDeviceCheckerboard1);

#else
	printf("ALLOC MULT MEM SEGMENTS\n");

	//now free the allocated data space
	cudaFree(dataCostDeviceCheckerboard1);
	cudaFree(dataCostDeviceCheckerboard2);

#endif

	( cudaThreadSynchronize() );

#ifdef RUN_DETAILED_TIMING

	gettimeofday(&timeFinalUnbindFreeEnd, NULL);
	gettimeofday(&timeFinalFreeEnd, NULL);

	timeStart = timeFinalUnbindFreeStart.tv_sec
												+ (timeFinalUnbindFreeStart.tv_usec / 1000000.0);
								timeEnd = timeFinalUnbindFreeEnd.tv_sec
												+ (timeFinalUnbindFreeEnd.tv_usec / 1000000.0);

		double totalTimeFinalUnbindFree = timeEnd - timeStart;

		timeStart = timeFinalFreeStart.tv_sec
					+ (timeFinalFreeStart.tv_usec / 1000000.0);
		timeEnd = timeFinalFreeEnd.tv_sec
					+ (timeFinalFreeEnd.tv_usec / 1000000.0);

		double totalTimeFinalFree = timeEnd - timeStart;

		printf("Time const mem in init settings: %f\n", totalTimeInitSettingsConstMem);
		printf("Time init settings malloc: %f\n", totalTimeInitSettingsMallocStart);
	printf("Time get data costs bottom level: %f\n", totalTimeGetDataCostsBottomLevel);
	printf("Time get data costs higher levels: %f\n", totalTimeGetDataCostsHigherLevels);
	printf("Time to init message values: %f\n", totalTimeInitMessageVals);
	printf("Total time BP Iters: %f\n", totalTimeBpIters);
	printf("Total time BP Iters (kernel portion only): %f\n", timeBpItersKernelTotalTime);
		printf("Total time Copy Data: %f\n", totalTimeCopyData);
		printf("Total time Copy Data (kernel portion only): %f\n", timeCopyDataKernelTotalTime);
	printf("Time get output disparity: %f\n", totalTimeGetOutputDisparity);
	printf("Time final unbind free: %f\n", totalTimeFinalUnbindFree);
	printf("Time final unbind: %f\n", totalTimeFinalUnbind);
	printf("Time final free: %f\n", totalTimeFinalFree);
	double totalTimed = totalTimeInitSettingsMallocStart + totalTimeGetDataCostsBottomLevel + totalTimeGetDataCostsHigherLevels + totalTimeInitMessageVals + totalTimeBpIters + totalTimeCopyData + totalTimeGetOutputDisparity + totalTimeFinalUnbindFree;
	printf("Total timed: %f\n", totalTimed);
#endif
}

