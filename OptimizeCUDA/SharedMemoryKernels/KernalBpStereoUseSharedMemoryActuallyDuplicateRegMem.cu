//code for using shared memory in the belief prop function; seems to work but is generally slower than not using shared memory,
//so currently not using except for testing

#define PROCESSING_ON_GPU
#include "../../SharedFuncts/SharedBPProcessingFuncts.h"
#include "../../bpStereoCudaParameters.h"
#undef PROCESSING_ON_GPU

template<typename T>
__device__ inline void dtStereoSharedActuallyRegAndRegLocalMemory(T* dstSharedMemActuallyReg, T* dst) {
	T prev;
	T lastVal;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	//int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
	//int indexIndexDstShared = startIndexDstShared;
	lastVal = dstSharedMemActuallyReg[0];
#else
	lastVal= dst[0];
#endif
#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 1;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		prev = lastVal + (T) 1.0;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstSharedMemActuallyReg[currentDisparity]) {
			dstSharedMemActuallyReg[currentDisparity] = prev;
		}
		lastVal = dstSharedMemActuallyReg[currentDisparity];
	}
#endif
#pragma unroll
	for (int currentDisparity = getMax(1, DISP_INDEX_START_REG_LOCAL_MEM);
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		prev = lastVal + (T) 1.0;
		if (prev < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = prev;
		}
		lastVal = dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
	}

//#pragma unroll 64
	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity--) {
		prev = lastVal + (T) 1.0;
		if (prev < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = prev;
		}
		lastVal = dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
	}
#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = getMin(NUM_POSSIBLE_DISPARITY_VALUES - 2,
			DISP_INDEX_START_REG_LOCAL_MEM - 1); currentDisparity >= 0;
			currentDisparity--) {
		prev = lastVal + (T) 1.0;
		if (prev < dstSharedMemActuallyReg[currentDisparity]) {
			dstSharedMemActuallyReg[currentDisparity] = prev;
		}
		lastVal = dstSharedMemActuallyReg[currentDisparity];
		//indexIndexDstShared -= BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif
}

template<>
__device__ inline void msgStereo<float, float>(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		float messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		float messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		float messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		float dataCosts[NUM_POSSIBLE_DISPARITY_VALUES], float* dstMessageArray,
		float disc_k_bp, bool dataAligned)
{
	//int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
	//int indexIntervalNextHalfIndexSharedVals = 0;
	//printf("USED SHARED MEMORY\n");
	// aggregate and find min
	float minimum = INF_BP;

#if DISP_INDEX_START_REG_LOCAL_MEM == 0
	float* dstSharedMemActuallyReg = nullptr;
#else
	//__shared__ float dstSharedMem[BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
	//		* (NUM_POSSIBLE_DISPARITY_VALUES - (NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM) + ((NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM) % 2))];
	float dstSharedMemActuallyReg[DISP_INDEX_START_REG_LOCAL_MEM];
#endif

	//int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
	//int indexIndexDstShared = startIndexDstShared;

	float dst[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dstSharedMemActuallyReg[currentDisparity] =
				messageValsNeighbor1[currentDisparity]
						+ messageValsNeighbor2[currentDisparity]
						+ messageValsNeighbor3[currentDisparity]
						+ dataCosts[currentDisparity];
		if (dstSharedMemActuallyReg[currentDisparity] < minimum) {
			minimum = dstSharedMemActuallyReg[currentDisparity];
		}
		//indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		//indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif

	//indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
	for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] =
				messageValsNeighbor1[currentDisparity]
						+ messageValsNeighbor2[currentDisparity]
						+ messageValsNeighbor3[currentDisparity]
						+ dataCosts[currentDisparity];
		if (dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] < minimum) {
			minimum = dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
		}
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//	dtStereo<float>(dst);
//#else
	//dtStereoSharedAndRegLocalMemory<float>(dstSharedMem, dst);
	dtStereoSharedActuallyRegAndRegLocalMemory<float>(dstSharedMemActuallyReg, dst);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	float valToNormalize = (float)0.0f;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	//indexIndexDstShared = startIndexDstShared;
	//indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		if (minimum < dstSharedMemActuallyReg[currentDisparity]) {
			dstSharedMemActuallyReg[currentDisparity] = minimum;
		}

		valToNormalize += dstSharedMemActuallyReg[currentDisparity];
		//indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		//indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif
//#pragma unroll 64
	for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		if (minimum < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = minimum;
		}

		valToNormalize +=
				dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
	}
		valToNormalize /= ((float) NUM_POSSIBLE_DISPARITY_VALUES);

		int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
		//indexIndexDstShared = startIndexDstShared;
		//indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
		for (int currentDisparity = 0;
				currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
				currentDisparity++) {
			dstSharedMemActuallyReg[currentDisparity] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] =
					dstSharedMemActuallyReg[currentDisparity];
			//indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
			//indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
			//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
#endif
//#pragma unroll 64
		for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] -=
					valToNormalize;
			dstMessageArray[destMessageArrayIndex] = dst[currentDisparity
					- DISP_INDEX_START_REG_LOCAL_MEM];

#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
}

template<>
__device__ inline void msgStereo<half, half>(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		half messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		half messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		half messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		half dataCosts[NUM_POSSIBLE_DISPARITY_VALUES], half* dstMessageArray,
		half disc_k_bp, bool dataAligned)
{
	//int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
	//int indexIntervalNextHalfIndexSharedVals = 0;
	//printf("USED SHARED MEMORY\n");
	// aggregate and find min
	half minimum = INF_BP;

#if DISP_INDEX_START_REG_LOCAL_MEM == 0
	half* dstSharedMemActuallyReg = nullptr;
#else
	//__shared__ half dstSharedMem[BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
	//		* (NUM_POSSIBLE_DISPARITY_VALUES - (NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM) + ((NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM) % 2))];
	half dstSharedMemActuallyReg[DISP_INDEX_START_REG_LOCAL_MEM];
#endif

	//int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
	//int indexIndexDstShared = startIndexDstShared;

	half dst[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dstSharedMemActuallyReg[currentDisparity] =
				messageValsNeighbor1[currentDisparity]
						+ messageValsNeighbor2[currentDisparity]
						+ messageValsNeighbor3[currentDisparity]
						+ dataCosts[currentDisparity];
		if (dstSharedMemActuallyReg[currentDisparity] < minimum) {
			minimum = dstSharedMemActuallyReg[currentDisparity];
		}
		//indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		//indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif

	//indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
	for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] =
				messageValsNeighbor1[currentDisparity]
						+ messageValsNeighbor2[currentDisparity]
						+ messageValsNeighbor3[currentDisparity]
						+ dataCosts[currentDisparity];
		if (dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] < minimum) {
			minimum = dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
		}
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//	dtStereo<float>(dst);
//#else
	//dtStereoSharedAndRegLocalMemory<half>(dstSharedMem, dst);
	dtStereoSharedActuallyRegAndRegLocalMemory<half>(dstSharedMemActuallyReg, dst);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	half valToNormalize = (half)0.0f;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	//indexIndexDstShared = startIndexDstShared;
	//indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		if (minimum < dstSharedMemActuallyReg[currentDisparity]) {
			dstSharedMemActuallyReg[currentDisparity] = minimum;
		}

		valToNormalize += dstSharedMemActuallyReg[currentDisparity];
		//indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		//indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif
//#pragma unroll 64
	for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		if (minimum < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = minimum;
		}

		valToNormalize +=
				dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
	}

	if (__hisnan(valToNormalize) || ((__hisinf(valToNormalize)) != 0))
	{
		int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			dstMessageArray[destMessageArrayIndex] = (half) 0.0;
#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
	}
	else
	{
		valToNormalize /= ((half) NUM_POSSIBLE_DISPARITY_VALUES);

		int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
		//indexIndexDstShared = startIndexDstShared;
		//indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
		for (int currentDisparity = 0;
				currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
				currentDisparity++) {
			dstSharedMemActuallyReg[currentDisparity] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] =
					dstSharedMemActuallyReg[currentDisparity];
			//indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
			//indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
			//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
#endif
//#pragma unroll 64
		for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] -=
					valToNormalize;
			dstMessageArray[destMessageArrayIndex] = dst[currentDisparity
					- DISP_INDEX_START_REG_LOCAL_MEM];

#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
					currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
	}
}




