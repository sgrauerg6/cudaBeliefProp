//code for using shared memory in the belief prop function; seems to work but is generally slower than not using shared memory,
//so currently not using except for testing

//#include "kernalBpStereoHeader.cuh"
#define PROCESSING_ON_GPU
#include "../SharedFuncts/SharedBPProcessingFuncts.h"
#undef PROCESSING_ON_GPU

#define DISPARITY_START_SHARED_MEM 100
#define MAX_DISPARITY_END_SHARED_MEM 170
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
#define NUM_DISP_VALS_IN_DEFAULT_LOCATION NUM_POSSIBLE_DISPARITY_VALUES
#define DISPARITY_START_SHARED_MEM (NUM_POSSIBLE_DISPARITY_VALUES + 9999)
#define DISPARITY_END_SHARED_MEM (NUM_POSSIBLE_DISPARITY_VALUES + 9999)

#elif (NUM_POSSIBLE_DISPARITY_VALUES - 1) > MAX_DISPARITY_END_SHARED_MEM //disparities go beyond end of shared memory
#define DISPARITY_END_SHARED_MEM MAX_DISPARITY_END_SHARED_MEM
#define NUM_DISP_VALS_IN_DEFAULT_LOCATION (DISPARITY_START_SHARED_MEM + ((NUM_POSSIBLE_DISPARITY_VALUES - 1) - DISPARITY_END_SHARED_MEM))

#else																	//disparities end within shared memory
#define NUM_DISP_VALS_IN_DEFAULT_LOCATION DISPARITY_START_SHARED_MEM
#define DISPARITY_END_SHARED_MEM (NUM_POSSIBLE_DISPARITY_VALUES - 1)

#endif //(NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM

template<typename T>
__device__ inline T getDstVal(int currentDisparity, T* dstDefault, T* dstShared)
{
	if (currentDisparity < DISPARITY_START_SHARED_MEM)
	{
		return dstDefault[currentDisparity];
	}
	else if (currentDisparity <= DISPARITY_END_SHARED_MEM)
	{
		return dstShared[(BLOCK_SIZE_WIDTH_BP*BLOCK_SIZE_HEIGHT_BP*(currentDisparity - DISPARITY_START_SHARED_MEM)) + (threadIdx.y*BLOCK_SIZE_WIDTH_BP + threadIdx.x)];
	}
	else
	{
		return dstDefault[currentDisparity - (DISPARITY_END_SHARED_MEM - DISPARITY_START_SHARED_MEM) - 1];
	}
}

template<typename T>
__device__ inline void setDstVal(int currentDisparity, T valToSet, T* dstDefault, T* dstShared)
{
	if (currentDisparity < DISPARITY_START_SHARED_MEM)
	{
		dstDefault[currentDisparity] = valToSet;
	}
	else if (currentDisparity <= DISPARITY_END_SHARED_MEM)
	{
		dstShared[(BLOCK_SIZE_WIDTH_BP*BLOCK_SIZE_HEIGHT_BP*(currentDisparity - DISPARITY_START_SHARED_MEM)) + (threadIdx.y*BLOCK_SIZE_WIDTH_BP + threadIdx.x)] = valToSet;
	}
	else
	{
		dstDefault[currentDisparity - (DISPARITY_END_SHARED_MEM - DISPARITY_START_SHARED_MEM) - 1] = valToSet;
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T>
__device__ inline void dtStereoSharedMemory(T* dst, T* dstShared) {
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
#else
	int startIndexDst = 0;
	int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;

	int indexIndexDst = startIndexDst;
	int indexIndexDstShared = startIndexDstShared;
	indexIndexDst++;
	T lastVal;
#endif
	T prev;
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
#else
	lastVal = dst[0];
#endif

#pragma unroll
	for (int currentDisparity = 1;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
		prev = dst[currentDisparity - 1] + (T) 1.0;
		if (prev < dst[currentDisparity]) {
			dst[currentDisparity] = prev;
		}
#else
		prev = lastVal + (T) 1.0;
		if ((currentDisparity < DISPARITY_START_SHARED_MEM)
				|| (currentDisparity > DISPARITY_END_SHARED_MEM)) {
			//prev = f[currentDisparity-1] + (T)1.0;
			//prev = getDstVal(currentDisparity - 1, dst, dstShared) + (T) 1.0;
			//if (prev < f[currentDisparity])
			if (prev < dst[indexIndexDst]) {
				//	f[currentDisparity] = prev;
				dst[indexIndexDst] = prev;
			}
			lastVal = dst[indexIndexDst];
			indexIndexDst++;
		} else {
			//prev = f[currentDisparity-1] + (T)1.0;
			//prev = getDstVal(currentDisparity - 1, dst, dstShared) + (T) 1.0;
			//if (prev < f[currentDisparity])
			if (prev < dstShared[indexIndexDstShared]) {
				//	f[currentDisparity] = prev;
				//setDstVal(currentDisparity, prev, dst, dstShared);
				dstShared[indexIndexDstShared] = prev;
			}
			lastVal = dstShared[indexIndexDstShared];
			indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		}
#endif
	}

#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
	//no need to set value; only using disparity
	//indexIndexDst = NUM_POSSIBLE_DISPARITY_VALUES - 2;

#elif (NUM_POSSIBLE_DISPARITY_VALUES - 1) > MAX_DISPARITY_END_SHARED_MEM //disparities go beyond end of shared memory
	indexIndexDst = (NUM_POSSIBLE_DISPARITY_VALUES
			- ((DISPARITY_END_SHARED_MEM - DISPARITY_START_SHARED_MEM) + 1))
			- 2;
	indexIndexDstShared = (DISPARITY_END_SHARED_MEM - DISPARITY_START_SHARED_MEM)*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP + startIndexDstShared;

#else																	//disparities end within shared memory
	indexIndexDst = DISPARITY_START_SHARED_MEM - 1;
	indexIndexDstShared = ((DISPARITY_END_SHARED_MEM - DISPARITY_START_SHARED_MEM) - 1)*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP + startIndexDstShared;
#endif

#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
#else
	lastVal = getDstVal(NUM_POSSIBLE_DISPARITY_VALUES -1, dst, dstShared);
#endif

#pragma unroll
	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES - 2;
			currentDisparity >= 0; currentDisparity--) {
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
		prev = dst[currentDisparity+1] + (T)1.0;
		if (prev < dst[currentDisparity])
		{
			dst[currentDisparity] = prev;
		}
#else
		prev = lastVal + (T) 1.0;
		if ((currentDisparity < DISPARITY_START_SHARED_MEM)
				|| (currentDisparity > DISPARITY_END_SHARED_MEM)) {
			//prev = f[currentDisparity+1] + (T)1.0;
			//if (prev < f[currentDisparity])
			if (prev < dst[indexIndexDst]) {
				//	f[currentDisparity] = prev;
				dst[indexIndexDst] = prev;
			}
			lastVal = dst[indexIndexDst];
			indexIndexDst--;
		} else {
			//prev = f[currentDisparity+1] + (T)1.0;
			//if (prev < f[currentDisparity])
			if (prev < dstShared[indexIndexDstShared]) {
				dstShared[indexIndexDstShared] = prev;
				//setDstVal(currentDisparity, prev, dst, dstShared);
			}
			lastVal = dstShared[indexIndexDstShared];
			indexIndexDstShared -= BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		}
#endif
	}
}

template<>
__device__ inline void msgStereo<float, float>(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		float messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES],
		float messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		float messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES],
		float dataCosts[NUM_POSSIBLE_DISPARITY_VALUES], float* dstMessageArray,
		float disc_k_bp, bool dataAligned) {
	// aggregate and find min
	float minimum = INF_BP;

#if NUM_POSSIBLE_DISPARITY_VALUES > DISPARITY_START_SHARED_MEM
	float dst[NUM_DISP_VALS_IN_DEFAULT_LOCATION];
	__shared__ float dstSharedMem[BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
			* (DISPARITY_END_SHARED_MEM - DISPARITY_START_SHARED_MEM + 1)];

	int startIndexDst = 0;
	int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;

	int indexIndexDst = startIndexDst;
	int indexIndexDstShared = startIndexDstShared;
#else
	float dst[NUM_POSSIBLE_DISPARITY_VALUES];
	float* dstSharedMem = nullptr;
#endif



#pragma unroll
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity]
		+ messageValsNeighbor2[currentDisparity]
		+ messageValsNeighbor3[currentDisparity]
		+ dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum) {
			minimum = dst[currentDisparity];
		}
#else
		if ((currentDisparity < DISPARITY_START_SHARED_MEM)
				|| (currentDisparity > DISPARITY_END_SHARED_MEM)) {
			dst[indexIndexDst] = messageValsNeighbor1[currentDisparity]
					+ messageValsNeighbor2[currentDisparity]
					+ messageValsNeighbor3[currentDisparity]
					+ dataCosts[currentDisparity];
			if (dst[indexIndexDst] < minimum) {
				minimum = dst[indexIndexDst];
			}
			indexIndexDst++;
		} else {
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
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
/*#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
	dtStereo<float>(dst);
#else*/
	dtStereoSharedMemory<float>(dst, dstSharedMem);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	float valToNormalize = 0.0f;

#if NUM_POSSIBLE_DISPARITY_VALUES > DISPARITY_START_SHARED_MEM

	indexIndexDst = startIndexDst;
	indexIndexDstShared = startIndexDstShared;

#endif

#pragma unroll
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
		if (minimum < dst[currentDisparity]) {
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
#else
		if ((currentDisparity < DISPARITY_START_SHARED_MEM)
				|| (currentDisparity > DISPARITY_END_SHARED_MEM)) {
			if (minimum < dst[indexIndexDst]) {
				dst[indexIndexDst] = minimum;
			}

			valToNormalize += dst[indexIndexDst];
			indexIndexDst++;
		} else {
			if (minimum < dstSharedMem[indexIndexDstShared]) {
				dstSharedMem[indexIndexDstShared] = minimum;
			}

			valToNormalize += dstSharedMem[indexIndexDstShared];
			indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		}
#endif
	}

	valToNormalize /= ((float) NUM_POSSIBLE_DISPARITY_VALUES);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

#if NUM_POSSIBLE_DISPARITY_VALUES > DISPARITY_START_SHARED_MEM
	indexIndexDst = startIndexDst;
	indexIndexDstShared = startIndexDstShared;
#endif

#pragma unroll
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
		dst[currentDisparity] -= valToNormalize;
		dstMessageArray[destMessageArrayIndex] = dst[currentDisparity];
#else
		if ((currentDisparity < DISPARITY_START_SHARED_MEM)
				|| (currentDisparity > DISPARITY_END_SHARED_MEM)) {
			dst[indexIndexDst] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] = dst[indexIndexDst];
			indexIndexDst++;
		} else {
			dstSharedMem[indexIndexDstShared] -= valToNormalize;
			dstMessageArray[destMessageArrayIndex] =
					dstSharedMem[indexIndexDstShared];
			indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
		}
#endif

#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
				currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
}
