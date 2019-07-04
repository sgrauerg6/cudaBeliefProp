//code for using shared memory in the belief prop function; seems to work but is generally slower than not using shared memory,
//so currently not using except for testing

#define PROCESSING_ON_GPU
#include "../../SharedFuncts/SharedBPProcessingFuncts.h"
#include "../../bpStereoCudaParameters.h"
#undef PROCESSING_ON_GPU


//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T>
__device__ inline void dtStereoSharedMemory(T* dstShared) {
	T prev;
	int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
	int indexIndexDstShared = startIndexDstShared;

//#pragma unroll 64
	for (int currentDisparity = 1;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		prev = dstShared[indexIndexDstShared] + (T) 1.0;
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
	}

//#pragma unroll 64
	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= 0; currentDisparity--) {
		prev = dstShared[indexIndexDstShared] + (T) 1.0;
		indexIndexDstShared -= BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
	}
}

template<>
__device__ inline void dtStereoSharedMemory<half>(half* dstShared) {

	int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
	int indexIntervalNextHalfIndexSharedVals = 0;

	half prev;
	int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
	int indexIndexDstShared = startIndexDstShared;

//#pragma unroll 64
	for (int currentDisparity = 1;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		prev = dstShared[indexIndexDstShared] + (half) 1.0;
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
	}

//#pragma unroll 64
	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= 0; currentDisparity--) {
		prev = dstShared[indexIndexDstShared] + (half) 1.0;
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		indexIndexDstShared -= halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		//indexIndexDstShared -= BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
	}
}

template<typename T>
__device__ inline void dtStereoSharedAndRegLocalMemory(T* dstShared, T* dst) {
	T prev;
	T lastVal;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
	int indexIndexDstShared = startIndexDstShared;
	lastVal = dstShared[indexIndexDstShared];
#else
	lastVal= dst[0];
#endif
#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 1;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		prev = lastVal + (T) 1.0;
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
		lastVal = dstShared[indexIndexDstShared];
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
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
		lastVal = dstShared[indexIndexDstShared];
		indexIndexDstShared -= BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif
}

template<>
__device__ inline void dtStereoSharedAndRegLocalMemory<half>(half* dstShared, half* dst)
{
	int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
	int indexIntervalNextHalfIndexSharedVals = 0;

	half prev;
	half lastVal;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
	int indexIndexDstShared = startIndexDstShared;
	lastVal = dstShared[indexIndexDstShared];
#else
	lastVal= dst[0];
#endif
#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 1;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		prev = lastVal + (half) 1.0;
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
		lastVal = dstShared[indexIndexDstShared];
	}
#endif
#pragma unroll
	for (int currentDisparity = getMax(1, DISP_INDEX_START_REG_LOCAL_MEM);
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		prev = lastVal + (half) 1.0;
		if (prev < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
			dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = prev;
		}
		lastVal = dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
	}

//#pragma unroll 64
	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity--) {
		prev = lastVal + (half) 1.0;
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
		prev = lastVal + (half) 1.0;
		if (prev < dstShared[indexIndexDstShared]) {
			dstShared[indexIndexDstShared] = prev;
		}
		lastVal = dstShared[indexIndexDstShared];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		indexIndexDstShared -= halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
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
	//printf("USED SHARED MEMORY\n");
	// aggregate and find min
	float minimum = INF_BP;

#if DISP_INDEX_START_REG_LOCAL_MEM == 0
	float* dstSharedMem = nullptr;
#else
	__shared__ float dstSharedMem[BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
			* (NUM_POSSIBLE_DISPARITY_VALUES - (NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM))];
#endif

	int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
	int indexIndexDstShared = startIndexDstShared;

#if DISP_INDEX_START_REG_LOCAL_MEM >= NUM_POSSIBLE_DISPARITY_VALUES
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		dstSharedMem[indexIndexDstShared] =
		messageValsNeighbor1[currentDisparity]
		+ messageValsNeighbor2[currentDisparity]
		+ messageValsNeighbor3[currentDisparity]
		+ dataCosts[currentDisparity];
		if (dstSharedMem[indexIndexDstShared] < minimum)
		{
			minimum = dstSharedMem[indexIndexDstShared];
		}
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}

	indexIndexDstShared = startIndexDstShared;

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		dstSharedMem[indexIndexDstShared] =
		messageValsNeighbor1[currentDisparity]
		+ messageValsNeighbor2[currentDisparity]
		+ messageValsNeighbor3[currentDisparity]
		+ dataCosts[currentDisparity];
		if (dstSharedMem[indexIndexDstShared] < minimum)
		{
			minimum = dstSharedMem[indexIndexDstShared];
		}
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//	dtStereo<float>(dst);
//#else
	dtStereoSharedMemory<float>(dstSharedMem);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	float valToNormalize = 0.0f;
	indexIndexDstShared = startIndexDstShared;

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		if (minimum < dstSharedMem[indexIndexDstShared])
		{
			dstSharedMem[indexIndexDstShared] = minimum;
		}

		valToNormalize += dstSharedMem[indexIndexDstShared];
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}

	valToNormalize /= ((float) NUM_POSSIBLE_DISPARITY_VALUES);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

	indexIndexDstShared = startIndexDstShared;
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		dstSharedMem[indexIndexDstShared] -= valToNormalize;
		dstMessageArray[destMessageArrayIndex] = dstSharedMem[indexIndexDstShared];
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
		currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
#else
	float dst[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dstSharedMem[indexIndexDstShared] =
				messageValsNeighbor1[currentDisparity]
						+ messageValsNeighbor2[currentDisparity]
						+ messageValsNeighbor3[currentDisparity]
						+ dataCosts[currentDisparity];
		if (dstSharedMem[indexIndexDstShared] < minimum) {
			minimum = dstSharedMem[indexIndexDstShared];
		}
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif

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
	dtStereoSharedAndRegLocalMemory<float>(dstSharedMem, dst);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	float valToNormalize = 0.0f;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	indexIndexDstShared = startIndexDstShared;
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		if (minimum < dstSharedMem[indexIndexDstShared]) {
			dstSharedMem[indexIndexDstShared] = minimum;
		}

		valToNormalize += dstSharedMem[indexIndexDstShared];
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
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
	indexIndexDstShared = startIndexDstShared;
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dstSharedMem[indexIndexDstShared] -= valToNormalize;
		dstMessageArray[destMessageArrayIndex] =
				dstSharedMem[indexIndexDstShared];
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

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

#endif
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
	int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
	int indexIntervalNextHalfIndexSharedVals = 0;
	//printf("USED SHARED MEMORY\n");
	// aggregate and find min
	half minimum = INF_BP;

#if DISP_INDEX_START_REG_LOCAL_MEM == 0
	half* dstSharedMem = nullptr;
#else
	__shared__ half dstSharedMem[BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
			* (NUM_POSSIBLE_DISPARITY_VALUES - (NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM) + ((NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM) % 2))];
#endif

	int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
	int indexIndexDstShared = startIndexDstShared;

#if DISP_INDEX_START_REG_LOCAL_MEM >= NUM_POSSIBLE_DISPARITY_VALUES
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		dstSharedMem[indexIndexDstShared] =
		messageValsNeighbor1[currentDisparity]
		+ messageValsNeighbor2[currentDisparity]
		+ messageValsNeighbor3[currentDisparity]
		+ dataCosts[currentDisparity];
		if (dstSharedMem[indexIndexDstShared] < minimum)
		{
			minimum = dstSharedMem[indexIndexDstShared];
		}
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = (indexIntervalNextHalfIndexSharedVals + 1) % 2;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}

	indexIndexDstShared = startIndexDstShared;
	indexIntervalNextHalfIndexSharedVals = 0;

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		dstSharedMem[indexIndexDstShared] =
		messageValsNeighbor1[currentDisparity]
		+ messageValsNeighbor2[currentDisparity]
		+ messageValsNeighbor3[currentDisparity]
		+ dataCosts[currentDisparity];
		if (dstSharedMem[indexIndexDstShared] < minimum)
		{
			minimum = dstSharedMem[indexIndexDstShared];
		}
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//	dtStereo<float>(dst);
//#else
	dtStereoSharedMemory<half>(dstSharedMem);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	half valToNormalize = (half)0.0f;
	indexIndexDstShared = startIndexDstShared;
	indexIntervalNextHalfIndexSharedVals = 0;

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++)
	{
		if (minimum < dstSharedMem[indexIndexDstShared])
		{
			dstSharedMem[indexIndexDstShared] = minimum;
		}

		valToNormalize += dstSharedMem[indexIndexDstShared];
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
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

		indexIndexDstShared = startIndexDstShared;
//#pragma unroll 64
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++)
		{
			dstSharedMem[indexIndexDstShared] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] = dstSharedMem[indexIndexDstShared];
			indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
			indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
			//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

#if OPTIMIZED_INDEXING_SETTING == 1
			destMessageArrayIndex +=
			currentLevelProperties.paddedWidthCheckerboardLevel;
#else
			destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
		}
	}
#else
	half dst[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dstSharedMem[indexIndexDstShared] =
				messageValsNeighbor1[currentDisparity]
						+ messageValsNeighbor2[currentDisparity]
						+ messageValsNeighbor3[currentDisparity]
						+ dataCosts[currentDisparity];
		if (dstSharedMem[indexIndexDstShared] < minimum) {
			minimum = dstSharedMem[indexIndexDstShared];
		}
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		//indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}
#endif

	indexIntervalNextHalfIndexSharedVals = 0;
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
	dtStereoSharedAndRegLocalMemory<half>(dstSharedMem, dst);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	half valToNormalize = (half)0.0f;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
	indexIndexDstShared = startIndexDstShared;
	indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		if (minimum < dstSharedMem[indexIndexDstShared]) {
			dstSharedMem[indexIndexDstShared] = minimum;
		}

		valToNormalize += dstSharedMem[indexIndexDstShared];
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
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
		} else {

		valToNormalize /= ((half) NUM_POSSIBLE_DISPARITY_VALUES);

		int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel,
				currentLevelProperties.heightLevel, 0,
				NUM_POSSIBLE_DISPARITY_VALUES);

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
		indexIndexDstShared = startIndexDstShared;
		indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
		for (int currentDisparity = 0;
				currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
				currentDisparity++) {
			dstSharedMem[indexIndexDstShared] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] =
					dstSharedMem[indexIndexDstShared];
			indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
			indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
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

#endif
}



