/*template<>
__device__ half2 getZeroVal<half2>()
{
  return __floats2half2_rn (0.0, 0.0);
}


__device__ half2 getMinBothPartsHalf2(half2 val1, half2 val2)
{
  half2 val1Less = __hlt2(val1, val2);
  half2 val2LessOrEqual = __hle2(val2, val1);
  return __hadd2(__hmul2(val1Less, val1), __hmul2(val2LessOrEqual, val2));
}

template<>
__device__ void dtStereo<half2>(half2 f[bp_params::STEREO_SETS_TO_PROCESS])
{
  half2 prev;
  for (int currentDisparity = 1; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
  {
    prev = __hadd2(f[currentDisparity-1], __float2half2_rn(1.0f));
    f[currentDisparity] = getMinBothPartsHalf2(prev, f[currentDisparity]);
  }

  for (int currentDisparity = bp_params::STEREO_SETS_TO_PROCESS-2; currentDisparity >= 0; currentDisparity--)
  {
    prev = __hadd2(f[currentDisparity+1], __float2half2_rn(1.0f));
    f[currentDisparity] = getMinBothPartsHalf2(prev, f[currentDisparity]);
  }
}*/


/*template<>
__device__ void msgStereo<half2>(half2 messageValsNeighbor1[bp_params::STEREO_SETS_TO_PROCESS], half2 messageValsNeighbor2[bp_params::STEREO_SETS_TO_PROCESS],
    half2 messageValsNeighbor3[bp_params::STEREO_SETS_TO_PROCESS], half2 dataCosts[bp_params::STEREO_SETS_TO_PROCESS],
    half2 dst[bp_params::STEREO_SETS_TO_PROCESS], half2 disc_k_bp)
{
  // aggregate and find min
  half2 minimum = __float2half2_rn(INF_BP);

  for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
  {
    dst[currentDisparity] = __hadd2(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
    dst[currentDisparity] = __hadd2(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
    dst[currentDisparity] = __hadd2(dst[currentDisparity], dataCosts[currentDisparity]);

    minimum = getMinBothPartsHalf2(dst[currentDisparity], minimum);
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
  dtStereo<half2>(dst);

  // truncate
  minimum = __hadd2(minimum, disc_k_bp);

  // normalize
  half2 valToNormalize = __float2half2_rn(0.0f);

  for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
  {
    dst[currentDisparity] = getMinBothPartsHalf2(minimum, dst[currentDisparity]);
    valToNormalize = __hadd2(valToNormalize, dst[currentDisparity]);
  }

  //if either valToNormalize in half2 is infinite or NaN, set destination vector to 0 for all disparities
  //note that may cause results to differ a little from ideal
  if (((__hisnan(__low2half(valToNormalize)))
      || ((__hisinf(__low2half(valToNormalize)) != 0)))
      || ((__hisnan(__high2half(valToNormalize)))
          || ((__hisinf(__high2half(valToNormalize)) != 0))))
  {
    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __floats2half2_rn(0.0f, 0.0f);
    }
  }
  else
  {
    valToNormalize = __h2div(valToNormalize,
        __float2half2_rn((float) bp_params::STEREO_SETS_TO_PROCESS));

    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __hsub2(dst[currentDisparity],
          valToNormalize);
    }
  }
  //check if both values in half2 are inf or nan
  /*if (((__hisnan(__low2half(valToNormalize)))
      || ((__hisinf(__low2half(valToNormalize)) != 0)))
      && ((__hisnan(__high2half(valToNormalize)))
          || ((__hisinf(__high2half(valToNormalize)) != 0))))
  {
    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __floats2half2_rn(0.0f, 0.0f);
    }
  }
  else if (((__hisnan(__low2half(valToNormalize)))
      || ((__hisinf(__low2half(valToNormalize)) != 0))))
  {
    //lower half of half2 is inf or nan
    valToNormalize = __h2div(valToNormalize,
        __float2half2_rn((float) bp_params::STEREO_SETS_TO_PROCESS));

    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __hsub2(dst[currentDisparity],
          valToNormalize);
    }

    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __halves2half2((half)0.0f,
          __high2half(dst[currentDisparity]));
    }
  }
  else if ((__hisnan(__high2half(valToNormalize)))
      || ((__hisinf(__high2half(valToNormalize)) != 0)))
  {
    //higher half of half2 is inf or nan
    valToNormalize = __h2div(valToNormalize,
        __float2half2_rn((float) bp_params::STEREO_SETS_TO_PROCESS));

    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __hsub2(dst[currentDisparity],
          valToNormalize);
    }

    for (int currentDisparity = 0;
        currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
        currentDisparity++)
    {
      dst[currentDisparity] = __halves2half2(
          __low2half(dst[currentDisparity]), (half)0.0f);
    }
  }
}*/


//device portion of the kernel function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
/*template<>
__device__ void runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMem<half2>(int xVal, int yVal,
    int checkerboardToUpdate, beliefprop::levelProperties& currentLevelProperties, half2* dataCostStereoCheckerboard1, half2* dataCostStereoCheckerboard2,
    half2* messageUDeviceCurrentCheckerboard1,
    half2* messageDDeviceCurrentCheckerboard1,
    half2* messageLDeviceCurrentCheckerboard1,
    half2* messageRDeviceCurrentCheckerboard1,
    half2* messageUDeviceCurrentCheckerboard2,
    half2* messageDDeviceCurrentCheckerboard2,
    half2* messageLDeviceCurrentCheckerboard2,
    half2* messageRDeviceCurrentCheckerboard2,
    float disc_k_bp, int offsetData)
{
}

  int indexWriteTo;
  int checkerboardAdjustment;

  //checkerboardAdjustment used for indexing into current checkerboard to update
  if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0)
  {
    checkerboardAdjustment = ((yVal)%2);
  }
  else //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
  {
    checkerboardAdjustment = ((yVal+1)%2);
  }

  //may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
  //make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
  //if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
  if ((xVal >= (1/*switch to 0 if trying to match half results exactly*//* - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
  {
    half2 prevUMessage[bp_params::STEREO_SETS_TO_PROCESS];
    half2 prevDMessage[bp_params::STEREO_SETS_TO_PROCESS];
    half2 prevLMessage[bp_params::STEREO_SETS_TO_PROCESS];
    half2 prevRMessage[bp_params::STEREO_SETS_TO_PROCESS];

    half2 dataMessage[bp_params::STEREO_SETS_TO_PROCESS];

    if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0)
    {
      half* messageLDeviceCurrentCheckerboard2Half = (half*)messageLDeviceCurrentCheckerboard2;
      half* messageRDeviceCurrentCheckerboard2Half = (half*)messageRDeviceCurrentCheckerboard2;

      for (int currentDisparity = 0;
          currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
          currentDisparity++)
      {
        dataMessage[currentDisparity] =
            dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
                xVal, yVal, widthLevelCheckerboardPart,
                heightLevel, currentDisparity,
                bp_params::STEREO_SETS_TO_PROCESS, offsetData)];
        prevUMessage[currentDisparity] =
            messageUDeviceCurrentCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(
                xVal, (yVal + 1), widthLevelCheckerboardPart,
                heightLevel, currentDisparity,
                bp_params::STEREO_SETS_TO_PROCESS)];
        prevDMessage[currentDisparity] =
            messageDDeviceCurrentCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(
                xVal, (yVal - 1), widthLevelCheckerboardPart,
                heightLevel, currentDisparity,
                bp_params::STEREO_SETS_TO_PROCESS)];
        prevLMessage[currentDisparity] =
            __halves2half2(
                messageLDeviceCurrentCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                    ((xVal * 2) + checkerboardAdjustment),
                    yVal, widthLevelCheckerboardPart * 2,
                    heightLevel, currentDisparity,
                    bp_params::STEREO_SETS_TO_PROCESS)],
                messageLDeviceCurrentCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                    ((xVal * 2 + 1) + checkerboardAdjustment),
                    yVal, widthLevelCheckerboardPart * 2,
                    heightLevel, currentDisparity,
                    bp_params::STEREO_SETS_TO_PROCESS)]);

        //if ((((xVal * 2) - 1) + checkerboardAdjustment) >= 0)
        {
          prevRMessage[currentDisparity] =
              __halves2half2(
                  messageRDeviceCurrentCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                      (((xVal * 2) - 1)
                          + checkerboardAdjustment),
                      yVal,
                      widthLevelCheckerboardPart * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)],
                  messageRDeviceCurrentCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                      (((xVal * 2 + 1) - 1)
                          + checkerboardAdjustment),
                      yVal,
                      widthLevelCheckerboardPart * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]);
        }
        /*else
        {
          prevRMessage[currentDisparity] =
              __halves2half2((half)0.0f,
                  messageRDeviceCurrentCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                      (((xVal * 2 + 1) - 1)
                          + checkerboardAdjustment),
                      yVal,
                      widthLevelCheckerboardPart * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]);
        }*//*
      }
    }
    else //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
    {
      half* messageLDeviceCurrentCheckerboard1Half = (half*)messageLDeviceCurrentCheckerboard1;
      half* messageRDeviceCurrentCheckerboard1Half = (half*)messageRDeviceCurrentCheckerboard1;

      for (int currentDisparity = 0;
          currentDisparity < bp_params::STEREO_SETS_TO_PROCESS;
          currentDisparity++)
      {
        dataMessage[currentDisparity] =
            dataCostStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(
                xVal, yVal, widthLevelCheckerboardPart,
                heightLevel, currentDisparity,
                bp_params::STEREO_SETS_TO_PROCESS, offsetData)];
        prevUMessage[currentDisparity] =
            messageUDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
                xVal, (yVal + 1), widthLevelCheckerboardPart,
                heightLevel, currentDisparity,
                bp_params::STEREO_SETS_TO_PROCESS)];
        prevDMessage[currentDisparity] =
            messageDDeviceCurrentCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
                xVal, (yVal - 1), widthLevelCheckerboardPart,
                heightLevel, currentDisparity,
                bp_params::STEREO_SETS_TO_PROCESS)];
        prevLMessage[currentDisparity] =
            __halves2half2(
                messageLDeviceCurrentCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                    ((xVal * 2)
                        + checkerboardAdjustment),
                    yVal, widthLevelCheckerboardPart * 2,
                    heightLevel, currentDisparity,
                    bp_params::STEREO_SETS_TO_PROCESS)],
                messageLDeviceCurrentCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                    ((xVal * 2 + 1)
                        + checkerboardAdjustment),
                    yVal, widthLevelCheckerboardPart * 2,
                    heightLevel, currentDisparity,
                    bp_params::STEREO_SETS_TO_PROCESS)]);

        //if ((((xVal * 2) - 1) + checkerboardAdjustment) >= 0)
        {
          prevRMessage[currentDisparity] =
              __halves2half2(
                  messageRDeviceCurrentCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                      (((xVal * 2) - 1)
                          + checkerboardAdjustment),
                      yVal,
                      widthLevelCheckerboardPart * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)],
                  messageRDeviceCurrentCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                      (((xVal * 2 + 1) - 1)
                          + checkerboardAdjustment),
                      yVal,
                      widthLevelCheckerboardPart * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]);
        }
        /*else
        {
          prevRMessage[currentDisparity] =
              __halves2half2((half) 0.0,
                  messageRDeviceCurrentCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                      (((xVal * 2 + 1) - 1)
                          + checkerboardAdjustment),
                      yVal,
                      widthLevelCheckerboardPart * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]);
        }*//*
      }
    }

    half2 currentUMessage[bp_params::STEREO_SETS_TO_PROCESS];
    half2 currentDMessage[bp_params::STEREO_SETS_TO_PROCESS];
    half2 currentLMessage[bp_params::STEREO_SETS_TO_PROCESS];
    half2 currentRMessage[bp_params::STEREO_SETS_TO_PROCESS];

    //uses the previous message values and data cost to calculate the current message values and store the results
    beliefprop::runBPIterationUpdateMsgVals<half2>(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
              currentUMessage, currentDMessage, currentLMessage, currentRMessage, __float2half2_rn(disc_k_bp));

    //write the calculated message values to global memory
    for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
    {
      indexWriteTo = beliefprop::retrieveIndexInDataAndMessage(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS);
      if (checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_0)
      {
        messageUDeviceCurrentCheckerboard1[indexWriteTo] = currentUMessage[currentDisparity];
        messageDDeviceCurrentCheckerboard1[indexWriteTo] = currentDMessage[currentDisparity];
        messageLDeviceCurrentCheckerboard1[indexWriteTo] = currentLMessage[currentDisparity];
        messageRDeviceCurrentCheckerboard1[indexWriteTo] = currentRMessage[currentDisparity];
      }
      else //checkerboardToUpdate == beliefprop::Checkerboard_Parts::CHECKERBOARD_PART_1
      {
        messageUDeviceCurrentCheckerboard2[indexWriteTo] = currentUMessage[currentDisparity];
        messageDDeviceCurrentCheckerboard2[indexWriteTo] = currentDMessage[currentDisparity];
        messageLDeviceCurrentCheckerboard2[indexWriteTo] = currentLMessage[currentDisparity];
        messageRDeviceCurrentCheckerboard2[indexWriteTo] = currentRMessage[currentDisparity];
      }
    }
  }
}
*/

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
/*template<>
__global__ void retrieveOutputDisparity<half2>(beliefprop::levelProperties currentLevelProperties, half2* dataCostStereoCheckerboard1, half2* dataCostStereoCheckerboard2, half2* messageUPrevStereoCheckerboard1, half2* messageDPrevStereoCheckerboard1, half2* messageLPrevStereoCheckerboard1, half2* messageRPrevStereoCheckerboard1, half2* messageUPrevStereoCheckerboard2, half2* messageDPrevStereoCheckerboard2, half2* messageLPrevStereoCheckerboard2, half2* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice)
{

}*/

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
/*template<typename T>
__global__ void retrieveOutputDisparityCheckerboardStereo(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
  //get the x and y indices for the current CUDA thread
  int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (withinImageBounds(xVal, yVal, widthLevel, heightLevel))
  {
    int widthCheckerboard = getCheckerboardWidth<T>(widthLevel);
    int xValInCheckerboardPart = xVal/2;

    if (((yVal+xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
    {
      int  checkerboardPartAdjustment = (yVal%2);

      if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
      {
        // keep track of "best" disparity for current pixel
        int bestDisparity = 0;
        T best_val = INF_BP;
        for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
        {
          T val = messageUPrevStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
             messageDPrevStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
             messageLPrevStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
             messageRPrevStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
             dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)];

          if (val < (best_val)) {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
        disparityBetweenImagesDevice[yVal*widthLevel + xVal] = bestDisparity;
      }
      else
      {
        disparityBetweenImagesDevice[yVal*widthLevel + xVal] = 0;
      }
    }
    else //pixel from part 2 of checkerboard
    {
      int  checkerboardPartAdjustment = ((yVal + 1) % 2);

      if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
      {


        // keep track of "best" disparity for current pixel
        int bestDisparity = 0;
        T best_val = INF_BP;
        for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
        {
          T val = messageUPrevStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
            messageDPrevStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
            messageLPrevStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage((xValInCheckerboardPart + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
            messageRPrevStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage((xValInCheckerboardPart - 1 + checkerboardPartAdjustment), yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)] +
            dataCostStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)];

          if (val < (best_val))
          {
            best_val = val;
            bestDisparity = currentDisparity;
          }
        }
        disparityBetweenImagesDevice[yVal*widthLevel + xVal] = bestDisparity;
      }
      else
      {
        disparityBetweenImagesDevice[yVal*widthLevel + xVal] = 0;
      }
    }
  }
}

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<>
__global__ void retrieveOutputDisparityCheckerboardStereo<half2>(half2* dataCostStereoCheckerboard1, half2* dataCostStereoCheckerboard2, half2* messageUPrevStereoCheckerboard1, half2* messageDPrevStereoCheckerboard1, half2* messageLPrevStereoCheckerboard1, half2* messageRPrevStereoCheckerboard1, half2* messageUPrevStereoCheckerboard2, half2* messageDPrevStereoCheckerboard2, half2* messageLPrevStereoCheckerboard2, half2* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
  //get the x and y indices for the current CUDA thread
  int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  if (withinImageBounds(xVal*2, yVal, widthLevel, heightLevel))
  {
    int widthCheckerboard = getCheckerboardWidth<half2>(widthLevel);
    int xValInCheckerboardPart = xVal/2;

    if (((yVal+xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
    {
      int  checkerboardPartAdjustment = (yVal%2);

      half* messageLPrevStereoCheckerboard2Half = (half*)messageLPrevStereoCheckerboard2;
      half* messageRPrevStereoCheckerboard2Half = (half*)messageRPrevStereoCheckerboard2;

      if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
      {
        // keep track of "best" disparity for current pixel
        int bestDisparity1 = 0;
        int bestDisparity2 = 0;
        float best_val1 = INF_BP;
        float best_val2 = INF_BP;
        for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
        {
          half2 val = __hadd2(messageUPrevStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal + 1), widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)],
                       messageDPrevStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, (yVal - 1), widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)]);
          val =
              __hadd2(val,
                  __halves2half2(
                      messageLPrevStereoCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                          ((xValInCheckerboardPart * 2)
                              + checkerboardPartAdjustment),
                          yVal, widthCheckerboard * 2,
                          heightLevel,
                          currentDisparity,
                          bp_params::STEREO_SETS_TO_PROCESS)],
                  messageLPrevStereoCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                      ((xValInCheckerboardPart * 2 + 1)
                          + checkerboardPartAdjustment),
                      yVal, widthCheckerboard * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]));
          val =
              __hadd2(val,
                  __halves2half2(
                      messageRPrevStereoCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                          ((xValInCheckerboardPart * 2)
                              - 1
                              + checkerboardPartAdjustment),
                          yVal, widthCheckerboard * 2,
                          heightLevel,
                          currentDisparity,
                          bp_params::STEREO_SETS_TO_PROCESS)],
                  messageRPrevStereoCheckerboard2Half[beliefprop::retrieveIndexInDataAndMessage(
                      ((xValInCheckerboardPart * 2 + 1)
                          - 1
                          + checkerboardPartAdjustment),
                      yVal, widthCheckerboard * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]));
          val = __hadd2(val, dataCostStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(xValInCheckerboardPart, yVal, widthCheckerboard, heightLevel, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS)]);

          float valLow = __low2float ( val);
          float valHigh = __high2float ( val);
          if (valLow < best_val1)
          {
            best_val1 = valLow;
            bestDisparity1 = currentDisparity;
          }
          if (valHigh < best_val2)
          {
            best_val2 = valHigh;
            bestDisparity2 = currentDisparity;
          }
        }
        disparityBetweenImagesDevice[yVal*widthLevel + (xVal*2 - checkerboardPartAdjustment)] = bestDisparity1;
        if (((xVal*2) + 2) < widthLevel)
        {
          disparityBetweenImagesDevice[yVal*widthLevel + (xVal*2 - checkerboardPartAdjustment) + 2] = bestDisparity2;
        }
      }
      else
      {
        disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)] =
            0;
        if (((xVal * 2) + 2) < widthLevel)
        {
          disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)
              + 2] = 0;
        }
      }
    }
    else //pixel from part 2 of checkerboard
    {
      int  checkerboardPartAdjustment = ((yVal + 1) % 2);
      half* messageLPrevStereoCheckerboard1Half = (half*)messageLPrevStereoCheckerboard1;
      half* messageRPrevStereoCheckerboard1Half = (half*)messageRPrevStereoCheckerboard1;

      if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1) && (yVal < (heightLevel - 1)))
      {
        // keep track of "best" disparity for current pixel
        int bestDisparity1 = 0;
        int bestDisparity2 = 0;
        float best_val1 = INF_BP;
        float best_val2 = INF_BP;
        for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
        {
          half2 val =
              __hadd2(
                  messageUPrevStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
                      xValInCheckerboardPart, (yVal + 1),
                      widthCheckerboard, heightLevel,
                      currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)],
                      messageDPrevStereoCheckerboard1[beliefprop::retrieveIndexInDataAndMessage(
                          xValInCheckerboardPart,
                          (yVal - 1),
                          widthCheckerboard,
                          heightLevel,
                          currentDisparity,
                          bp_params::STEREO_SETS_TO_PROCESS)]);
          val =
              __hadd2(val,
                  __halves2half2(
                      messageLPrevStereoCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                          ((xValInCheckerboardPart * 2)
                              + checkerboardPartAdjustment),
                          yVal, widthCheckerboard * 2,
                          heightLevel,
                          currentDisparity,
                          bp_params::STEREO_SETS_TO_PROCESS)],
                  messageLPrevStereoCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                      ((xValInCheckerboardPart * 2 + 1)
                          + checkerboardPartAdjustment),
                      yVal, widthCheckerboard * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]));
          val =
              __hadd2(val,
                  __halves2half2(
                      messageRPrevStereoCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                          ((xValInCheckerboardPart * 2)
                              - 1
                              + checkerboardPartAdjustment),
                          yVal, widthCheckerboard * 2,
                          heightLevel,
                          currentDisparity,
                          bp_params::STEREO_SETS_TO_PROCESS)],
                  messageRPrevStereoCheckerboard1Half[beliefprop::retrieveIndexInDataAndMessage(
                      ((xValInCheckerboardPart * 2 + 1)
                          - 1
                          + checkerboardPartAdjustment),
                      yVal, widthCheckerboard * 2,
                      heightLevel, currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]));

          val =
              __hadd2(val,
                  dataCostStereoCheckerboard2[beliefprop::retrieveIndexInDataAndMessage(
                      xValInCheckerboardPart, yVal,
                      widthCheckerboard, heightLevel,
                      currentDisparity,
                      bp_params::STEREO_SETS_TO_PROCESS)]);

          float val1 = __low2float(val);
          float val2 = __high2float(val);
          if (val1 < best_val1) {
            best_val1 = val1;
            bestDisparity1 = currentDisparity;
          }
          if (val2 < best_val2) {
            best_val2 = val2;
            bestDisparity2 = currentDisparity;
          }
        }

        disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)] =
            bestDisparity1;
        if (((xVal * 2) + 2) < widthLevel) {
          disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)
              + 2] = bestDisparity2;
        }
      }
      else
      {
        disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)] =
            0;
        if (((xVal * 2) + 2) < widthLevel) {
          disparityBetweenImagesDevice[yVal * widthLevel + (xVal * 2 - checkerboardPartAdjustment)
              + 2] = 0;
        }
      }
    }
  }
}
*/

/*template<>
__global__ void initializeBottomLevelData<half2>(beliefprop::levelProperties currentLevelProperties, float* image1PixelsDevice, float* image2PixelsDevice, half2* dataCostDeviceStereoCheckerboard1, half2* dataCostDeviceStereoCheckerboard2, float lambda_bp, float data_k_bp)
{
  //get the x and y indices for the current CUDA thread
  int xVal = blockIdx.x * blockDim.x + threadIdx.x;
  int yVal = blockIdx.y * blockDim.y + threadIdx.y;

  int indexVal;
  int imageCheckerboardWidth = getCheckerboardWidth<half2>(widthImages);
  int xInCheckerboard = xVal / 2;

  if (withinImageBounds(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages))
  {
    int imageXPixelIndexStart = 0;
    int checkerboardNum = 1;

    //check which checkerboard data values for and make necessary adjustment to start
    if (((yVal) % 2) == 0) {
      if (((xVal) % 2) == 0) {
        checkerboardNum = 1;
      } else {
        checkerboardNum = 2;
      }
    } else {
      if (((xVal) % 2) == 0) {
        checkerboardNum = 2;
      } else {
        checkerboardNum = 1;
      }
    }

    imageXPixelIndexStart = xVal*2;
    if ((((yVal) % 2) == 0) && (checkerboardNum == 2)) {
      imageXPixelIndexStart -= 1;
    }
    if ((((yVal) % 2) == 1) && (checkerboardNum == 1)) {
      imageXPixelIndexStart -= 1;
    }

    //make sure that it is possible to check every disparity value
    if ((((imageXPixelIndexStart + 2) - (bp_params::STEREO_SETS_TO_PROCESS-1)) >= 0))
    {
      for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
      {
        float currentPixelImage1_low = 0.0;
        float currentPixelImage2_low = 0.0;

        if ((((imageXPixelIndexStart) - (bp_params::STEREO_SETS_TO_PROCESS-1)) >= 0))
        {
          if (withinImageBounds(imageXPixelIndexStart, yVal, widthImages,
              heightImages)) {
            currentPixelImage1_low = image1PixelsDevice[yVal
                * widthImages + imageXPixelIndexStart];
            currentPixelImage2_low = image2PixelsDevice[yVal
                * widthImages + (imageXPixelIndexStart - currentDisparity)];
          }
        }

        float currentPixelImage1_high = 0.0;
        float currentPixelImage2_high = 0.0;

        if (withinImageBounds(imageXPixelIndexStart + 2, yVal, widthImages,
            heightImages))
        {
          currentPixelImage1_high = image1PixelsDevice[yVal * widthImages
              + (imageXPixelIndexStart + 2)];
          currentPixelImage2_high = image2PixelsDevice[yVal * widthImages
              + ((imageXPixelIndexStart + 2) - currentDisparity)];
        }

        indexVal = beliefprop::retrieveIndexInDataAndMessage(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS);

        half lowVal = (half)(lambda_bp * min(abs(currentPixelImage1_low - currentPixelImage2_low), data_k_bp));
        half highVal = (half)(lambda_bp * min(abs(currentPixelImage1_high - currentPixelImage2_high), data_k_bp));

        //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
        if (checkerboardNum == 1)
        {
          dataCostDeviceStereoCheckerboard1[indexVal] = __halves2half2(lowVal, highVal);
        }
        else
        {
          dataCostDeviceStereoCheckerboard2[indexVal] = __halves2half2(lowVal, highVal);
        }
      }
    }
    else
    {
      for (int currentDisparity = 0; currentDisparity < bp_params::STEREO_SETS_TO_PROCESS; currentDisparity++)
      {
        indexVal = beliefprop::retrieveIndexInDataAndMessage(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, bp_params::STEREO_SETS_TO_PROCESS);

        //data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
        if (((xVal + yVal) % 2) == 0)
        {
          dataCostDeviceStereoCheckerboard1[indexVal] = getZeroVal<half2>();
        }
        else
        {
          dataCostDeviceStereoCheckerboard2[indexVal] = getZeroVal<half2>();
        }
      }
    }
  }
}*/
