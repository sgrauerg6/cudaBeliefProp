//code for using shared memory in the belief prop function; seems to work but is generally slower than not using shared memory,
//so currently not using except for testing

#include "../BpSharedFuncts/SharedBPProcessingFuncts.h"
#include "../BpStereoCudaParameters.h"

template<typename T>
__device__ inline void dtStereoSharedActuallyRegAndRegLocalMemory(T* dstSharedMemActuallyReg, T* dst) {
  T prev;
  T lastVal;

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
  //unsigned int startIndexDstShared = threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x;
  //unsigned int indexIndexDstShared = startIndexDstShared;
  lastVal = dstSharedMemActuallyReg[0];
#else
  lastVal= dst[0];
#endif
#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
  for (unsigned int currentDisparity = 1;
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
  for (unsigned int currentDisparity = getMax(1, DISP_INDEX_START_REG_LOCAL_MEM);
      currentDisparity < STEREO_SETS_TO_PROCESS;
      currentDisparity++) {
    prev = lastVal + (T) 1.0;
    if (prev < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
      dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = prev;
    }
    lastVal = dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
  }

//#pragma unroll 64
  for (unsigned int currentDisparity = STEREO_SETS_TO_PROCESS - 2;
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
  for (unsigned int currentDisparity = getMin(STEREO_SETS_TO_PROCESS - 2,
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
    beliefprop::levelProperties& currentLevelProperties,
    float messageValsNeighbor1[STEREO_SETS_TO_PROCESS],
    float messageValsNeighbor2[STEREO_SETS_TO_PROCESS],
    float messageValsNeighbor3[STEREO_SETS_TO_PROCESS],
    float dataCosts[STEREO_SETS_TO_PROCESS], float* dstMessageArray,
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
  //    * (STEREO_SETS_TO_PROCESS - (STEREO_SETS_TO_PROCESS - DISP_INDEX_START_REG_LOCAL_MEM) + ((STEREO_SETS_TO_PROCESS - DISP_INDEX_START_REG_LOCAL_MEM) % 2))];
  float dstSharedMemActuallyReg[DISP_INDEX_START_REG_LOCAL_MEM];
#endif

  //unsigned int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
  //unsigned int indexIndexDstShared = startIndexDstShared;

  float dst[STEREO_SETS_TO_PROCESS - DISP_INDEX_START_REG_LOCAL_MEM];

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
  for (unsigned int currentDisparity = 0;
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
  for (unsigned int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
      currentDisparity < STEREO_SETS_TO_PROCESS;
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
//#if (STEREO_SETS_TO_PROCESS - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//  dtStereo<float>(dst);
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
  for (unsigned int currentDisparity = 0;
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
  for (unsigned int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
      currentDisparity < STEREO_SETS_TO_PROCESS;
      currentDisparity++) {
    if (minimum < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
      dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = minimum;
    }

    valToNormalize +=
        dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
  }
    valToNormalize /= ((float) STEREO_SETS_TO_PROCESS);

    unsigned int destMessageArrayIndex = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_,
        currentLevelProperties.heightLevel_, 0,
        STEREO_SETS_TO_PROCESS);

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
    //indexIndexDstShared = startIndexDstShared;
    //indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
    for (unsigned int currentDisparity = 0;
        currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
        currentDisparity++) {
      dstSharedMemActuallyReg[currentDisparity] -= valToNormalize;
      dstMessageArray[destMessageArrayIndex] =
          dstSharedMemActuallyReg[currentDisparity];
      //indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
      //indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
      //indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

#if bp_params::OPTIMIZED_INDEXING_SETTING == 1
      destMessageArrayIndex +=
          currentLevelProperties.paddedWidthCheckerboardLevel_;
#else
      destMessageArrayIndex++;
#endif //bp_params::OPTIMIZED_INDEXING_SETTING == 1
    }
#endif
//#pragma unroll 64
    for (unsigned int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
        currentDisparity < STEREO_SETS_TO_PROCESS;
        currentDisparity++) {
      dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] -=
          valToNormalize;
      dstMessageArray[destMessageArrayIndex] = dst[currentDisparity
          - DISP_INDEX_START_REG_LOCAL_MEM];

#if bp_params::OPTIMIZED_INDEXING_SETTING == 1
      destMessageArrayIndex +=
          currentLevelProperties.paddedWidthCheckerboardLevel_;
#else
      destMessageArrayIndex++;
#endif //bp_params::OPTIMIZED_INDEXING_SETTING == 1
    }
}

template<>
__device__ inline void msgStereo<half, half>(int xVal, int yVal,
    beliefprop::levelProperties& currentLevelProperties,
    half messageValsNeighbor1[STEREO_SETS_TO_PROCESS],
    half messageValsNeighbor2[STEREO_SETS_TO_PROCESS],
    half messageValsNeighbor3[STEREO_SETS_TO_PROCESS],
    half dataCosts[STEREO_SETS_TO_PROCESS], half* dstMessageArray,
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
  //    * (STEREO_SETS_TO_PROCESS - (STEREO_SETS_TO_PROCESS - DISP_INDEX_START_REG_LOCAL_MEM) + ((STEREO_SETS_TO_PROCESS - DISP_INDEX_START_REG_LOCAL_MEM) % 2))];
  half dstSharedMemActuallyReg[DISP_INDEX_START_REG_LOCAL_MEM];
#endif

  //unsigned int startIndexDstShared = 2*(threadIdx.y * BLOCK_SIZE_WIDTH_BP + threadIdx.x);
  //unsigned int indexIndexDstShared = startIndexDstShared;

  half dst[STEREO_SETS_TO_PROCESS - DISP_INDEX_START_REG_LOCAL_MEM];

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
//#pragma unroll 64
  for (unsigned int currentDisparity = 0;
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
  for (unsigned int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
      currentDisparity < STEREO_SETS_TO_PROCESS;
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
//#if (STEREO_SETS_TO_PROCESS - 1) <= DISPARITY_START_SHARED_MEM //no shared memory used
//  dtStereo<float>(dst);
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
  for (unsigned int currentDisparity = 0;
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
  for (unsigned int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
      currentDisparity < STEREO_SETS_TO_PROCESS;
      currentDisparity++) {
    if (minimum < dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM]) {
      dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] = minimum;
    }

    valToNormalize +=
        dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM];
  }

  if (__hisnan(valToNormalize) || ((__hisinf(valToNormalize)) != 0))
  {
    unsigned int destMessageArrayIndex = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_,
        currentLevelProperties.heightLevel_, 0,
        STEREO_SETS_TO_PROCESS);

    for (unsigned int currentDisparity = 0;
        currentDisparity < STEREO_SETS_TO_PROCESS;
        currentDisparity++) {
      dstMessageArray[destMessageArrayIndex] = (half) 0.0;
#if bp_params::OPTIMIZED_INDEXING_SETTING == 1
      destMessageArrayIndex +=
          currentLevelProperties.paddedWidthCheckerboardLevel_;
#else
      destMessageArrayIndex++;
#endif //bp_params::OPTIMIZED_INDEXING_SETTING == 1
    }
  }
  else
  {
    valToNormalize /= ((half) STEREO_SETS_TO_PROCESS);

    unsigned int destMessageArrayIndex = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal,
        currentLevelProperties.paddedWidthCheckerboardLevel_,
        currentLevelProperties.heightLevel_, 0,
        STEREO_SETS_TO_PROCESS);

#if DISP_INDEX_START_REG_LOCAL_MEM > 0
    //indexIndexDstShared = startIndexDstShared;
    //indexIntervalNextHalfIndexSharedVals = 0;
//#pragma unroll 64
    for (unsigned int currentDisparity = 0;
        currentDisparity < DISP_INDEX_START_REG_LOCAL_MEM;
        currentDisparity++) {
      dstSharedMemActuallyReg[currentDisparity] -= valToNormalize;
      dstMessageArray[destMessageArrayIndex] =
          dstSharedMemActuallyReg[currentDisparity];
      //indexIndexDstShared += halfIndexSharedVals[indexIntervalNextHalfIndexSharedVals];
      //indexIntervalNextHalfIndexSharedVals = !indexIntervalNextHalfIndexSharedVals;
      //indexIndexDstShared += BLOCK_SIZE_WIDTH_BP * BLOCK_SIZE_HEIGHT_BP;

#if bp_params::OPTIMIZED_INDEXING_SETTING == 1
      destMessageArrayIndex +=
          currentLevelProperties.paddedWidthCheckerboardLevel_;
#else
      destMessageArrayIndex++;
#endif //bp_params::OPTIMIZED_INDEXING_SETTING == 1
    }
#endif
//#pragma unroll 64
    for (unsigned int currentDisparity = DISP_INDEX_START_REG_LOCAL_MEM;
        currentDisparity < STEREO_SETS_TO_PROCESS;
        currentDisparity++) {
      dst[currentDisparity - DISP_INDEX_START_REG_LOCAL_MEM] -=
          valToNormalize;
      dstMessageArray[destMessageArrayIndex] = dst[currentDisparity
          - DISP_INDEX_START_REG_LOCAL_MEM];

#if bp_params::OPTIMIZED_INDEXING_SETTING == 1
      destMessageArrayIndex +=
          currentLevelProperties.paddedWidthCheckerboardLevel_;
#else
      destMessageArrayIndex++;
#endif //bp_params::OPTIMIZED_INDEXING_SETTING == 1
    }
  }
}




