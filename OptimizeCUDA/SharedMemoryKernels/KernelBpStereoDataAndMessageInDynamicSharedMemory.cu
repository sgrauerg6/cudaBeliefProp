#define PROCESSING_ON_GPU
#include "../../SharedFuncts/SharedBPProcessingFuncts.h"
#include "../../bpStereoCudaParameters.h"
#undef PROCESSING_ON_GPU

template<typename T, typename U>
__device__ inline void msgStereo(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		T* messageValsNeighbor1Shared, T* messageValsNeighbor2Shared,
		T* messageValsNeighbor3Shared, T* dataCostsShared,
		T* messageValsNeighbor1, T* messageValsNeighbor2,
		T* messageValsNeighbor3, T* dataCosts, T* dstMessageArray,
		U disc_k_bp, bool dataAligned)
{
	printf("Data type not supported\n");
}

template<>
__device__ inline void msgStereo<half, half>(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		half* messageValsNeighbor1Shared,
		half* messageValsNeighbor2Shared,
		half* messageValsNeighbor3Shared,
		half* dataCostsShared,
		half* messageValsNeighbor1,
		half* messageValsNeighbor2,
		half* messageValsNeighbor3,
		half* dataCosts, half* dstMessageArray,
		half disc_k_bp, bool dataAligned)
{
	//printf("USED SHARED MEMORY\n");
	// aggregate and find min
	half minimum = INF_BP;

	int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
	int indexIndexDstShared = startIndexDstShared;
	int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
	int indexIntervalNextHalfIndexSharedVals = 0;

	half dst[NUM_POSSIBLE_DISPARITY_VALUES];

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dst[currentDisparity] =
				messageValsNeighbor1Shared[indexIndexDstShared]
						+ messageValsNeighbor2Shared[indexIndexDstShared]
						+ messageValsNeighbor3Shared[indexIndexDstShared]
						+ dataCostsShared[indexIndexDstShared];
		if (dst[currentDisparity] < minimum) {
			minimum = dst[currentDisparity];
		}
		indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
		indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
	}

//#pragma unroll 64
	for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity] =
				messageValsNeighbor1[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]
						+ messageValsNeighbor2[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]
						+ messageValsNeighbor3[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]
						+ dataCosts[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
		if (dst[currentDisparity] < minimum) {
			minimum = dst[currentDisparity];
		}
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//	dtStereo<float>(dst);
//#else
	dtStereo<half>(dst);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	half valToNormalize = 0.0f;

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		if (minimum < dst[currentDisparity]) {
			dst[currentDisparity] = minimum;
		}

		valToNormalize +=
				dst[currentDisparity];
	}

	valToNormalize /= ((half) NUM_POSSIBLE_DISPARITY_VALUES);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity] -=
				valToNormalize;
		dstMessageArray[destMessageArrayIndex] = dst[currentDisparity];

#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
		currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
}

template<>
__device__ inline void msgStereo<float, float>(int xVal, int yVal,
		levelProperties& currentLevelProperties,
		float* messageValsNeighbor1Shared,
				float* messageValsNeighbor2Shared,
				float* messageValsNeighbor3Shared,
				float* dataCostsShared,
		float* messageValsNeighbor1,
		float* messageValsNeighbor2,
		float* messageValsNeighbor3,
		float* dataCosts, float* dstMessageArray,
		float disc_k_bp, bool dataAligned)
{
	//printf("USED SHARED MEMORY\n");
	// aggregate and find min
	float minimum = INF_BP;

	int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
	int indexIndexDstShared = startIndexDstShared;

	float dst[NUM_POSSIBLE_DISPARITY_VALUES];

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity++) {
		dst[currentDisparity] =
				messageValsNeighbor1Shared[indexIndexDstShared]
						+ messageValsNeighbor2Shared[indexIndexDstShared]
						+ messageValsNeighbor3Shared[indexIndexDstShared]
						+ dataCostsShared[indexIndexDstShared];
		if (dst[currentDisparity] < minimum) {
			minimum = dst[currentDisparity];
		}
		indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;
	}

//#pragma unroll 64
	for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity] =
				messageValsNeighbor1[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]
						+ messageValsNeighbor2[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]
						+ messageValsNeighbor3[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]
						+ dataCosts[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
		if (dst[currentDisparity] < minimum) {
			minimum = dst[currentDisparity];
		}
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//#if (NUM_POSSIBLE_DISPARITY_VALUES - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//	dtStereo<float>(dst);
//#else
	dtStereo<float>(dst);
//#endif

	// truncate
	minimum += disc_k_bp;

	// normalize
	float valToNormalize = 0.0f;

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		if (minimum < dst[currentDisparity]) {
			dst[currentDisparity] = minimum;
		}

		valToNormalize +=
				dst[currentDisparity];
	}

	valToNormalize /= ((float) NUM_POSSIBLE_DISPARITY_VALUES);

	int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel,
			currentLevelProperties.heightLevel, 0,
			NUM_POSSIBLE_DISPARITY_VALUES);

//#pragma unroll 64
	for (int currentDisparity = 0;
			currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
			currentDisparity++) {
		dst[currentDisparity] -=
				valToNormalize;
		dstMessageArray[destMessageArrayIndex] = dst[currentDisparity];

#if OPTIMIZED_INDEXING_SETTING == 1
		destMessageArrayIndex +=
		currentLevelProperties.paddedWidthCheckerboardLevel;
#else
		destMessageArrayIndex++;
#endif //OPTIMIZED_INDEXING_SETTING == 1
	}
}


template<typename T, typename U>
ARCHITECTURE_ADDITION inline void runBPIterationInOutDataInLocalMem(int xVal, int yVal, levelProperties& currentLevelProperties, T* prevUMessageShared, T* prevDMessageShared,
		T* prevLMessageShared, T* prevRMessageShared, T* dataMessageShared, T* prevUMessage, T* prevDMessage, T* prevLMessage, T* prevRMessage, T* dataMessage,
		T* currentUMessageArray, T* currentDMessageArray, T* currentLMessageArray, T* currentRMessageArray, U disc_k_bp, bool dataAligned)
{
	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessageShared,
			prevLMessageShared, prevRMessageShared, dataMessageShared, prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevDMessageShared,
			prevLMessageShared, prevRMessageShared, dataMessageShared, prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessageShared, prevDMessageShared,
			prevRMessageShared, dataMessageShared, prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessageArray, disc_k_bp, dataAligned);

	msgStereo<T, U>(xVal, yVal, currentLevelProperties, prevUMessageShared, prevDMessageShared,
			prevLMessageShared, dataMessageShared, prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessageArray, disc_k_bp, dataAligned);
}

#if CURRENT_DATA_TYPE_PROCESSING_FROM_PYTHON == DATA_TYPE_PROCESSING_FLOAT

//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<float, float>(
		int xVal, int yVal, int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1,
		float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1,
		float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2,
		float* messageDDeviceCurrentCheckerboard2,
		float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
		int offsetData, bool dataAligned)
{
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_1)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_2
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (currentLevelProperties.heightLevel - 1)))
	{
#if (DISP_INDEX_START_REG_LOCAL_MEM < NUM_POSSIBLE_DISPARITY_VALUES)
		float prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		float prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		float prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		float prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		float dataMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
#else
		float* prevUMessage = nullptr;
		float* prevDMessage = nullptr;
		float* prevLMessage = nullptr;
		float* prevRMessage = nullptr;
		float* dataMessage = nullptr;
#endif

#if (DISP_INDEX_START_REG_LOCAL_MEM > 0)
	int numDataSharedMemoryArray = BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
								* DISP_INDEX_START_REG_LOCAL_MEM;
		extern __shared__ float dstSharedMem[];
		float *prevUMessageShared = dstSharedMem;
		float *prevDMessageShared = &dstSharedMem[numDataSharedMemoryArray];
		float *prevLMessageShared = &dstSharedMem[2*numDataSharedMemoryArray];
		float *prevRMessageShared = &dstSharedMem[3*numDataSharedMemoryArray];
		float *dataMessageShared = &dstSharedMem[4*numDataSharedMemoryArray];
#else
		float *prevUMessageShared = nullptr;
		float *prevDMessageShared = nullptr;
		float *prevLMessageShared = nullptr;
		float *prevRMessageShared = nullptr;
		float *dataMessageShared = nullptr;
#endif

		int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
		int indexIndexDstShared = startIndexDstShared;
		for (int currentDisparity = 0; currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				dataMessageShared[indexIndexDstShared] = (dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessageShared[indexIndexDstShared] = (messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessageShared[indexIndexDstShared] = (messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessageShared[indexIndexDstShared] = (messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessageShared[indexIndexDstShared] = (messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessageShared[indexIndexDstShared] = (dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessageShared[indexIndexDstShared] = (messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessageShared[indexIndexDstShared] = (messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessageShared[indexIndexDstShared] = (messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessageShared[indexIndexDstShared] = (messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;;
		}

		for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardToUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
						prevUMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardToUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
						prevUMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

		//uses the previous message values and data cost to calculate the current message values and store the results
		if (checkerboardToUpdate == CHECKERBOARD_PART_1)
		{
			runBPIterationInOutDataInLocalMem<float, float>(xVal, yVal, currentLevelProperties, prevUMessageShared, prevDMessageShared,
					prevLMessageShared, prevRMessageShared, dataMessageShared, prevUMessage, prevDMessage,
					prevLMessage, prevRMessage, dataMessage,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1, (float) disc_k_bp,
					dataAligned);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_2
		{
			runBPIterationInOutDataInLocalMem<float, float>(xVal, yVal, currentLevelProperties, prevUMessageShared, prevDMessageShared,
					prevLMessageShared, prevRMessageShared, dataMessageShared, prevUMessage, prevDMessage,
					prevLMessage, prevRMessage, dataMessage,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2, (float) disc_k_bp,
					dataAligned);
		}
	}
}

#elif CURRENT_DATA_TYPE_PROCESSING_FROM_PYTHON == DATA_TYPE_PROCESSING_HALF

//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<>
ARCHITECTURE_ADDITION inline void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<half, half>(
		int xVal, int yVal, int checkerboardToUpdate,
		levelProperties& currentLevelProperties,
		half* dataCostStereoCheckerboard1, half* dataCostStereoCheckerboard2,
		half* messageUDeviceCurrentCheckerboard1,
		half* messageDDeviceCurrentCheckerboard1,
		half* messageLDeviceCurrentCheckerboard1,
		half* messageRDeviceCurrentCheckerboard1,
		half* messageUDeviceCurrentCheckerboard2,
		half* messageDDeviceCurrentCheckerboard2,
		half* messageLDeviceCurrentCheckerboard2,
		half* messageRDeviceCurrentCheckerboard2, float disc_k_bp,
		int offsetData, bool dataAligned)
{
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_1)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_2
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (currentLevelProperties.heightLevel - 1)))
	{
#if (DISP_INDEX_START_REG_LOCAL_MEM < NUM_POSSIBLE_DISPARITY_VALUES)
		half prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		half prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		half prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		half prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
		half dataMessage[NUM_POSSIBLE_DISPARITY_VALUES - DISP_INDEX_START_REG_LOCAL_MEM];
#else
		half* prevUMessage = nullptr;
		half* prevDMessage = nullptr;
		half* prevLMessage = nullptr;
		half* prevRMessage = nullptr;
		half* dataMessage = nullptr;
#endif

#if (DISP_INDEX_START_REG_LOCAL_MEM > 0)
		int numDataSharedMemoryArray = BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP
				* (DISP_INDEX_START_REG_LOCAL_MEM + (DISP_INDEX_START_REG_LOCAL_MEM % 2));
		extern __shared__ half dstSharedMem[];
		half *prevUMessageShared = dstSharedMem;
		half *prevDMessageShared = &dstSharedMem[numDataSharedMemoryArray];
		half *prevLMessageShared = &dstSharedMem[2*numDataSharedMemoryArray];
		half *prevRMessageShared = &dstSharedMem[3*numDataSharedMemoryArray];
		half *dataMessageShared = &dstSharedMem[4*numDataSharedMemoryArray];
#else
		half *prevUMessageShared = nullptr;
		half *prevDMessageShared = nullptr;
		half *prevLMessageShared = nullptr;
		half *prevRMessageShared = nullptr;
		half *dataMessageShared = nullptr;
#endif

		int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
		int indexIndexDstShared = startIndexDstShared;
		int halfIndexSharedVals[2] = {1, (2*BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP)-1};
		int indexIntervalNextHalfIndexSharedVals = 0;

		for (int currentDisparity = 0; currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				dataMessageShared[indexIndexDstShared] = (dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessageShared[indexIndexDstShared] = (messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessageShared[indexIndexDstShared] = (messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessageShared[indexIndexDstShared] = (messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessageShared[indexIndexDstShared] = (messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessageShared[indexIndexDstShared] = (dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
				prevUMessageShared[indexIndexDstShared] = (messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevDMessageShared[indexIndexDstShared] = (messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevLMessageShared[indexIndexDstShared] = (messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				prevRMessageShared[indexIndexDstShared] = (messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
			indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
			indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
		}

		for (int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardToUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
						prevUMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardToUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (dataCostStereoCheckerboard2[retrieveIndexInDataAndMessage(xVal, yVal, currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)]);
						prevUMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal+1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(xVal, (yVal-1), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage((xVal + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = (messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(((xVal - 1) + checkerboardAdjustment), (yVal), currentLevelProperties.paddedWidthCheckerboardLevel, currentLevelProperties.heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

		//uses the previous message values and data cost to calculate the current message values and store the results
		if (checkerboardToUpdate == CHECKERBOARD_PART_1)
		{
			runBPIterationInOutDataInLocalMem<half, half>(xVal, yVal, currentLevelProperties, prevUMessageShared, prevDMessageShared,
					prevLMessageShared, prevRMessageShared, dataMessageShared, prevUMessage, prevDMessage,
					prevLMessage, prevRMessage, dataMessage,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1, (half) disc_k_bp,
					dataAligned);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_2
		{
			runBPIterationInOutDataInLocalMem<half, half>(xVal, yVal, currentLevelProperties, prevUMessageShared, prevDMessageShared,
					prevLMessageShared, prevRMessageShared, dataMessageShared, prevUMessage, prevDMessage,
					prevLMessage, prevRMessage, dataMessage,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2, (half) disc_k_bp,
					dataAligned);
		}
	}
}

#endif

