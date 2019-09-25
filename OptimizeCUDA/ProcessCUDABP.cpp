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

#include "ProcessCUDABP.h"
#include "kernalBpStereo.cu"

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

/* May be needed if using half2
#if ((CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF) || (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO))

template<>
int ProcessCUDABP<half2>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
			return (int)ceil(((ceil(((float)widthLevelActualIntegerSize) / 2.0)) / 2.0));
}

template<>
int ProcessCUDABP<half>::getCheckerboardWidthTargetDevice(int widthLevelActualIntegerSize) {
	ProcessCUDABP<half2> processCUDABPHalf;
	return processCUDABPHalf.getCheckerboardWidthTargetDevice(widthLevelActualIntegerSize) * 2;
}

#endif
*/


//functions directed related to running BP to retrieve the movement between the images

//cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
//cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
//run the given number of iterations of BP at the current level using the given message values in global device memory
template<typename T, typename U>
void ProcessCUDABP<T, U>::runBPAtCurrentLevel(const BPsettings& algSettings,
		const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDevice)
{
	cudaDeviceSetCacheConfig(cudaFuncCachePreferShared);
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int) ceil(
			(float) (currentLevelProperties.widthCheckerboardLevel) / (float) threads.x); //only updating half at a time
	grid.y = (unsigned int) ceil((float) currentLevelProperties.heightLevel / (float) threads.y);

	//in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
	//still is a check if start of row is aligned
	bool dataAligned = MemoryAlignedAtDataStart(0, 1);

	//at each level, run BP for numIterations, alternating between updating the messages between the two "checkerboards"
	for (int iterationNum = 0; iterationNum < algSettings.numIterations; iterationNum++)
	{
		Checkerboard_Parts checkboardPartUpdate = ((iterationNum % 2) == 0) ? CHECKERBOARD_PART_1 : CHECKERBOARD_PART_0;

		(cudaDeviceSynchronize());

#if (((USE_SHARED_MEMORY == 3) || (USE_SHARED_MEMORY == 4))  && (DISP_INDEX_START_REG_LOCAL_MEM > 0))
		int numDataSharedMemory = BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
					* (DISP_INDEX_START_REG_LOCAL_MEM);
		int numBytesSharedMemory = numDataSharedMemory * sizeof(T);

#if (USE_SHARED_MEMORY == 4)

		numBytesSharedMemory *= 5;

#endif //(USE_SHARED_MEMORY == 4)

		int maxbytes = numBytesSharedMemory; // 96 KB
		cudaFuncSetAttribute(runBPIterationUsingCheckerboardUpdates<T>, cudaFuncAttributeMaxDynamicSharedMemorySize, maxbytes);

		//std::cout << "numDataSharedMemory: " << numDataSharedMemory << std::endl;
		runBPIterationUsingCheckerboardUpdates<T><<<grid, threads, maxbytes>>>(checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDevice.messagesU_Checkerboard0, messagesDevice.messagesD_Checkerboard0,
				messagesDevice.messagesL_Checkerboard0, messagesDevice.messagesR_Checkerboard0,
				messagesDevice.messagesU_Checkerboard1, messagesDevice.messagesD_Checkerboard1,
				messagesDevice.messagesL_Checkerboard1, messagesDevice.messagesR_Checkerboard1,
				algSettings.disc_k_bp, dataAligned);

#else
		runBPIterationUsingCheckerboardUpdates<T><<<grid, threads>>>(checkboardPartUpdate, currentLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				messagesDevice.messagesU_Checkerboard0, messagesDevice.messagesD_Checkerboard0,
				messagesDevice.messagesL_Checkerboard0, messagesDevice.messagesR_Checkerboard0,
				messagesDevice.messagesU_Checkerboard1, messagesDevice.messagesD_Checkerboard1,
				messagesDevice.messagesL_Checkerboard1, messagesDevice.messagesR_Checkerboard1,
				algSettings.disc_k_bp, dataAligned);
#endif

		(cudaDeviceSynchronize());
		gpuErrchk( cudaPeekAtLastError() );

		cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);
	}
}

//copy the computed BP message values from the current now-completed level to the corresponding slots in the next level "down" in the computation
//pyramid; the next level down is double the width and height of the current level so each message in the current level is copied into four "slots"
//in the next level down
//need two different "sets" of message values to avoid read-write conflicts
template<typename T, typename U>
void ProcessCUDABP<T, U>::copyMessageValuesToNextLevelDown(
		const levelProperties& currentLevelProperties,
		const levelProperties& nextlevelProperties,
		const checkerboardMessages<U>& messagesDeviceCopyFrom,
		const checkerboardMessages<U>& messagesDeviceCopyTo)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int)ceil((float)(currentLevelProperties.widthCheckerboardLevel) / (float)threads.x);
	grid.y = (unsigned int)ceil((float)(currentLevelProperties.heightLevel) / (float)threads.y);

	( cudaDeviceSynchronize() );

	gpuErrchk( cudaPeekAtLastError() );

	for (const auto& checkerboard_part : {CHECKERBOARD_PART_0, CHECKERBOARD_PART_1})
		{

	//call the kernal to copy the computed BP message data to the next level down in parallel in each of the two "checkerboards"
	//storing the current message values
	copyPrevLevelToNextLevelBPCheckerboardStereo<T> <<< grid, threads >>> (checkerboard_part, currentLevelProperties, nextlevelProperties, messagesDeviceCopyFrom.messagesU_Checkerboard0, messagesDeviceCopyFrom.messagesD_Checkerboard0,
			messagesDeviceCopyFrom.messagesL_Checkerboard0, messagesDeviceCopyFrom.messagesR_Checkerboard0, messagesDeviceCopyFrom.messagesU_Checkerboard1, messagesDeviceCopyFrom.messagesD_Checkerboard1,
			messagesDeviceCopyFrom.messagesL_Checkerboard1, messagesDeviceCopyFrom.messagesR_Checkerboard1, messagesDeviceCopyTo.messagesU_Checkerboard0, messagesDeviceCopyTo.messagesD_Checkerboard0,
			messagesDeviceCopyTo.messagesL_Checkerboard0, messagesDeviceCopyTo.messagesR_Checkerboard0, messagesDeviceCopyTo.messagesU_Checkerboard1, messagesDeviceCopyTo.messagesD_Checkerboard1,
			messagesDeviceCopyTo.messagesL_Checkerboard1, messagesDeviceCopyTo.messagesR_Checkerboard1);

(cudaDeviceSynchronize());
gpuErrchk(cudaPeekAtLastError());
		}
}




//initialize the data cost at each pixel with no estimated Stereo values...only the data and discontinuity costs are used
template<typename T, typename U>
void ProcessCUDABP<T, U>::initializeDataCosts(const BPsettings& algSettings, const levelProperties& currentLevelProperties,
		const std::array<float*, 2>& imagesOnTargetDevice, const dataCostData<U>& dataCostDeviceCheckerboard)
{
	//since this is first kernel run in BP, set to prefer L1 cache for now since no shared memory is used by default
	cudaDeviceSetCacheConfig(cudaFuncCachePreferL1);

	//setup execution parameters
	//the thread size remains constant throughout but the grid size is adjusted based on the current level/kernal to run
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//kernal run on full-sized image to retrieve data costs at the "bottom" level of the pyramid
	grid.x = (unsigned int)ceil((float)currentLevelProperties.widthLevel / (float)threads.x);
	grid.y = (unsigned int)ceil((float)currentLevelProperties.heightLevel / (float)threads.y);

	//initialize the data the the "bottom" of the image pyramid
	initializeBottomLevelDataStereo<T><<<grid, threads>>>(currentLevelProperties, imagesOnTargetDevice[0],
			imagesOnTargetDevice[1], dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1, algSettings.lambda_bp, algSettings.data_k_bp);

	( cudaDeviceSynchronize() );
}

//initialize the message values with no previous message values...all message values are set to 0
template<typename T, typename U>
void ProcessCUDABP<T, U>::initializeMessageValsToDefault(
		const levelProperties& currentLevelProperties,
		const checkerboardMessages<U>& messagesDevice)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid((unsigned int)ceil((float)currentLevelProperties.widthCheckerboardLevel / (float)threads.x), (unsigned int)ceil((float)currentLevelProperties.heightLevel / (float)threads.y));

	//initialize all the message values for each pixel at each possible movement to the default value in the kernal
	initializeMessageValsToDefaultKernel<T> <<< grid, threads >>> (currentLevelProperties, messagesDevice.messagesU_Checkerboard0, messagesDevice.messagesD_Checkerboard0, messagesDevice.messagesL_Checkerboard0,
			messagesDevice.messagesR_Checkerboard0, messagesDevice.messagesU_Checkerboard1, messagesDevice.messagesD_Checkerboard1, messagesDevice.messagesL_Checkerboard1,
			messagesDevice.messagesR_Checkerboard1);

	cudaDeviceSynchronize();
}


template<typename T, typename U>
void ProcessCUDABP<T, U>::initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const dataCostData<U>& dataCostDeviceCheckerboardWriteTo)
{
	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	//each pixel "checkerboard" is half the width of the level and there are two of them; each "pixel/point" at the level belongs to one checkerboard and
	//the four-connected neighbors are in the other checkerboard
	grid.x = (unsigned int) ceil(
			((float) currentLevelProperties.widthCheckerboardLevel) / (float) threads.x);
	grid.y = (unsigned int) ceil(
			(float) currentLevelProperties.heightLevel / (float) threads.y);

	gpuErrchk( cudaPeekAtLastError() );

	size_t offsetNum = 0;
	for (const auto& checkerboardAndDataCost : { std::make_pair(
				CHECKERBOARD_PART_0,
				dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard0),
				std::make_pair(CHECKERBOARD_PART_1,
					dataCostDeviceCheckerboardWriteTo.dataCostCheckerboard1) })
	{
		initializeCurrentLevelDataStereo<T> <<<grid, threads>>>(checkerboardAndDataCost.first,
				currentLevelProperties, prevLevelProperties,
				dataCostDeviceCheckerboard.dataCostCheckerboard0,
				dataCostDeviceCheckerboard.dataCostCheckerboard1,
				checkerboardAndDataCost.second,
				((int) offsetNum / sizeof(float)));

		cudaDeviceSynchronize();
		gpuErrchk(cudaPeekAtLastError());
	}
}

#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF_TWO)

//due to the checkerboard indexing, half2 must be converted to half with the half function used for copying to the next level
template<>
void ProcessCUDABP<half2>::copyMessageValuesToNextLevelDown(
		const levelProperties& currentLevelProperties,
		const levelProperties& nextlevelProperties,
		const checkerboardMessages<half2>& messagesDeviceCopyFrom,
		const checkerboardMessages<half2>& messagesDeviceCopyTo)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.copyMessageValuesToNextLevelDown(
			prevlevelProperties,
			currentLevelProperties,
			(half*)messagesDeviceCopyFrom.messagesU_Checkerboard0,
			(half*)messagesDeviceCopyFrom.messagesD_Checkerboard0,
			(half*)messagesDeviceCopyFrom.messagesL_Checkerboard0,
			(half*)messagesDeviceCopyFrom.messagesR_Checkerboard0,
			(half*)messagesDeviceCopyFrom.messagesU_Checkerboard1,
			(half*)messagesDeviceCopyFrom.messagesD_Checkerboard1,
			(half*)messagesDeviceCopyFrom.messagesL_Checkerboard1,
			(half*)messagesDeviceCopyFrom.messagesR_Checkerboard1,
			(half*)messagesDeviceCopyTo.messagesU_Checkerboard0,
			(half*)messagesDeviceCopyTo.messagesD_Checkerboard0,
			(half*)messagesDeviceCopyTo.messagesL_Checkerboard0,
			(half*)messagesDeviceCopyTo.messagesR_Checkerboard0,
			(half*)messagesDeviceCopyTo.messagesU_Checkerboard1,
			(half*)messagesDeviceCopyTo.messagesD_Checkerboard1,
			(half*)messagesDeviceCopyTo.messagesL_Checkerboard1,
			(half*)messagesDeviceCopyTo.messagesR_Checkerboard1);
}

//due to indexing, need to convert to half* and use half arrays for this function
template<>
void ProcessCUDABP<half2>::initializeDataCurrentLevel(const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		half2* dataCostStereoCheckerboard1,
		half2* dataCostStereoCheckerboard2,
		half2* dataCostDeviceToWriteToCheckerboard1,
		half2* dataCostDeviceToWriteToCheckerboard2)
{
	ProcessCUDABP<half> processCUDABPHalf;
	processCUDABPHalf.initializeDataCurrentLevel(currentLevelProperties,
			prevLevelProperties,
			(half*)dataCostStereoCheckerboard1,
			(half*)dataCostStereoCheckerboard2,
			(half*)dataCostDeviceToWriteToCheckerboard1,
			(half*)dataCostDeviceToWriteToCheckerboard2);
}

#endif

template<typename T, typename U>
float* ProcessCUDABP<T, U>::retrieveOutputDisparity(
		const levelProperties& currentLevelProperties,
		const dataCostData<U>& dataCostDeviceCheckerboard,
		const checkerboardMessages<U>& messagesDevice)
{
	float* resultingDisparityMapCompDevice;
	cudaMalloc((void**)&resultingDisparityMapCompDevice, currentLevelProperties.widthLevel * currentLevelProperties.heightLevel * sizeof(float));

	dim3 threads(bp_cuda_params::BLOCK_SIZE_WIDTH_BP, bp_cuda_params::BLOCK_SIZE_HEIGHT_BP);
	dim3 grid;

	grid.x = (unsigned int) ceil((float) currentLevelProperties.widthCheckerboardLevel / (float) threads.x);
	grid.y = (unsigned int) ceil((float) currentLevelProperties.heightLevel / (float) threads.y);

	retrieveOutputDisparityCheckerboardStereoOptimized<T> <<<grid, threads>>>(currentLevelProperties,
			dataCostDeviceCheckerboard.dataCostCheckerboard0,
			dataCostDeviceCheckerboard.dataCostCheckerboard1,
			messagesDevice.messagesU_Checkerboard0,
			messagesDevice.messagesD_Checkerboard0,
			messagesDevice.messagesL_Checkerboard0,
			messagesDevice.messagesR_Checkerboard0,
			messagesDevice.messagesU_Checkerboard1,
			messagesDevice.messagesD_Checkerboard1,
			messagesDevice.messagesL_Checkerboard1,
			messagesDevice.messagesR_Checkerboard1,
			resultingDisparityMapCompDevice);

	(cudaDeviceSynchronize());
	gpuErrchk(cudaPeekAtLastError());

	return resultingDisparityMapCompDevice;
}



#if CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT
template class ProcessCUDABP<float, float*>;
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE
template class ProcessCUDABP<double, double*>;
#elif CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF
template class ProcessCUDABP<half, half*>;
#endif //CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT

//not currently supporting half2 data type
//template class ProcessCUDABP<half2>;
