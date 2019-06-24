/*
 * KernelBpStereoCPU_NEON.h
 *
 *  Created on: Jun 23, 2019
 *      Author: scott
 */

#ifndef KERNELBPSTEREOCPU_NEON_H_
#define KERNELBPSTEREOCPU_NEON_H_

#include <arm_neon.h>

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<> inline
void KernelBpStereoCPU::dtStereoCPU<float32x4_t>(float32x4_t f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	float32x4_t prev;
	float32x4_t vectorAllOneVal = vdupq_n_f32(1.0f);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = vaddq_f32(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = vaddq_f32(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = vminnmq_f32(prev, f[currentDisparity]);
	}
}


// compute current message
template<> inline
void KernelBpStereoCPU::msgStereoCPU<float32x4_t>(float32x4_t messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		float32x4_t messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		float32x4_t dst[NUM_POSSIBLE_DISPARITY_VALUES], float32x4_t disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	float32x4_t minimum = vdupq_n_f32(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] =  vaddq_f32(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] =  vaddq_f32(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] =  vaddq_f32(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = vminnmq_f32(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<float32x4_t>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum =  vaddq_f32(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	float32x4_t valToNormalize = vdupq_n_f32(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = vminnmq_f32(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = vaddq_f32(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = vdivq_f32(valToNormalize, vdupq_n_f32((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] =  vsubq_f32(dst[currentDisparity], valToNormalize);
	}
}


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseNEON<float>(int checkerboardToUpdate, levelProperties& currentLevelProperties,
		float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1,
		float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1,
		float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2,
		float* messageDDeviceCurrentCheckerboard2,
		float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
{
	int numDataInNEONVector = 4;
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;
	float32x4_t disc_k_bp_vector = vdupq_n_f32(disc_k_bp);

	#pragma omp parallel for
	for (int yVal = 1; yVal < currentLevelProperties.heightLevel - 1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}

		int startX = (checkerboardAdjustment == 1) ? 0 : 1;
		int endFinal = std::min(
				currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment,
				widthCheckerboardRunProcessing);
		int endXAvxStart = (endFinal / numDataInNEONVector) * numDataInNEONVector
				- numDataInNEONVector;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInNEONVector)
		{
			int xValProcess = xVal;

			//need this check first for case where endXAvxStart is 0 and startX is 1
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xValProcess > endXAvxStart) {
				xValProcess = endFinal - numDataInNEONVector;
			}

			//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
			xValProcess = std::max(startX, xValProcess);

			//check if the memory is aligned for AVX instructions at xValProcess location
			bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(
					xValProcess, numDataInNEONVector);

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			float32x4_t dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			//load using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								vld1q_f32(
										&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								vld1q_f32(
										&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								vld1q_f32(
										&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								vld1q_f32(
										&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								vld1q_f32(
										&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								vld1q_f32(
										&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								vld1q_f32(
										&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								vld1q_f32(
										&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								vld1q_f32(
										&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								vld1q_f32(
										&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								vld1q_f32(
										&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								vld1q_f32(
										&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								vld1q_f32(
										&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								vld1q_f32(
										&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								vld1q_f32(
										&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								vld1q_f32(
										&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] =
								vld1q_f32(
										&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] =
								vld1q_f32(
										&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] =
								vld1q_f32(
										&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] =
								vld1q_f32(
										&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}
			}

			float32x4_t currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			msgStereoCPU<float32x4_t >(prevUMessage, prevLMessage, prevRMessage,
					dataMessage, currentUMessage, disc_k_bp_vector);

			msgStereoCPU<float32x4_t >(prevDMessage, prevLMessage, prevRMessage,
					dataMessage, currentDMessage, disc_k_bp_vector);

			msgStereoCPU<float32x4_t >(prevUMessage, prevDMessage, prevRMessage,
					dataMessage, currentRMessage, disc_k_bp_vector);

			msgStereoCPU<float32x4_t >(prevUMessage, prevDMessage, prevLMessage,
					dataMessage, currentLMessage, disc_k_bp_vector);

			//save using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				//write the calculated message values to global memory
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						vst1q_f32(
								&messageUDeviceCurrentCheckerboard1[indexWriteTo],
								currentUMessage[currentDisparity]);
						vst1q_f32(
								&messageDDeviceCurrentCheckerboard1[indexWriteTo],
								currentDMessage[currentDisparity]);
						vst1q_f32(
								&messageLDeviceCurrentCheckerboard1[indexWriteTo],
								currentLMessage[currentDisparity]);
						vst1q_f32(
								&messageRDeviceCurrentCheckerboard1[indexWriteTo],
								currentRMessage[currentDisparity]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						vst1q_f32(
								&messageUDeviceCurrentCheckerboard2[indexWriteTo],
								currentUMessage[currentDisparity]);
						vst1q_f32(
								&messageDDeviceCurrentCheckerboard2[indexWriteTo],
								currentDMessage[currentDisparity]);
						vst1q_f32(
								&messageLDeviceCurrentCheckerboard2[indexWriteTo],
								currentLMessage[currentDisparity]);
						vst1q_f32(
								&messageRDeviceCurrentCheckerboard2[indexWriteTo],
								currentRMessage[currentDisparity]);
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						vst1q_f32(
								&messageUDeviceCurrentCheckerboard1[indexWriteTo],
								currentUMessage[currentDisparity]);
						vst1q_f32(
								&messageDDeviceCurrentCheckerboard1[indexWriteTo],
								currentDMessage[currentDisparity]);
						vst1q_f32(
								&messageLDeviceCurrentCheckerboard1[indexWriteTo],
								currentLMessage[currentDisparity]);
						vst1q_f32(
								&messageRDeviceCurrentCheckerboard1[indexWriteTo],
								currentRMessage[currentDisparity]);
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						vst1q_f32(
								&messageUDeviceCurrentCheckerboard2[indexWriteTo],
								currentUMessage[currentDisparity]);
						vst1q_f32(
								&messageDDeviceCurrentCheckerboard2[indexWriteTo],
								currentDMessage[currentDisparity]);
						vst1q_f32(
								&messageLDeviceCurrentCheckerboard2[indexWriteTo],
								currentLMessage[currentDisparity]);
						vst1q_f32(
								&messageRDeviceCurrentCheckerboard2[indexWriteTo],
								currentRMessage[currentDisparity]);
					}
				}
			}
		}
	}
}

//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUUseNEON<float16_t>(int checkerboardToUpdate, levelProperties& currentLevelProperties,
		float16_t* dataCostStereoCheckerboard1, float16_t* dataCostStereoCheckerboard2,
		float16_t* messageUDeviceCurrentCheckerboard1,
		float16_t* messageDDeviceCurrentCheckerboard1,
		float16_t* messageLDeviceCurrentCheckerboard1,
		float16_t* messageRDeviceCurrentCheckerboard1,
		float16_t* messageUDeviceCurrentCheckerboard2,
		float16_t* messageDDeviceCurrentCheckerboard2,
		float16_t* messageLDeviceCurrentCheckerboard2,
		float16_t* messageRDeviceCurrentCheckerboard2, float disc_k_bp)
{
	int numDataInNEONVector = 4;
	int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel / 2;
	float32x4_t disc_k_bp_vector = vdupq_n_f32(disc_k_bp);

	#pragma omp parallel for
	for (int yVal = 1; yVal < currentLevelProperties.heightLevel - 1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}

		int startX = (checkerboardAdjustment == 1) ? 0 : 1;
		int endFinal = std::min(
				currentLevelProperties.widthCheckerboardLevel - checkerboardAdjustment,
				widthCheckerboardRunProcessing);
		int endXAvxStart = (endFinal / numDataInNEONVector) * numDataInNEONVector
				- numDataInNEONVector;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInNEONVector)
		{
			int xValProcess = xVal;

			//need this check first for case where endXAvxStart is 0 and startX is 1
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xValProcess > endXAvxStart) {
				xValProcess = endFinal - numDataInNEONVector;
			}

			//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
			xValProcess = std::max(startX, xValProcess);

			//check if the memory is aligned for AVX instructions at xValProcess location
			bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(
					xValProcess, numDataInNEONVector);

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			float32x4_t dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			//load using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						dataMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValProcess, yVal,
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal + 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValProcess, (yVal - 1),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												(xValProcess
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] =
								vcvt_f32_f16(vld1d_f16(
										&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
												((xValProcess - 1)
														+ checkerboardAdjustment),
												(yVal),
												currentLevelProperties.paddedWidthCheckerboardLevel,
												currentLevelProperties.heightLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
				}
			}

			float32x4_t currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			float32x4_t currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

			msgStereoCPU<float32x4_t >(prevUMessage, prevLMessage, prevRMessage,
					dataMessage, currentUMessage, disc_k_bp_vector);

			msgStereoCPU<float32x4_t >(prevDMessage, prevLMessage, prevRMessage,
					dataMessage, currentDMessage, disc_k_bp_vector);

			msgStereoCPU<float32x4_t >(prevUMessage, prevDMessage, prevRMessage,
					dataMessage, currentRMessage, disc_k_bp_vector);

			msgStereoCPU<float32x4_t >(prevUMessage, prevDMessage, prevLMessage,
					dataMessage, currentLMessage, disc_k_bp_vector);

			//save using aligned instructions when possible
			if (dataAlignedAtXValProcess) {
				//write the calculated message values to global memory
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						vst1q_f32(
								&messageUDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentUMessage[currentDisparity]));
						vst1q_f32(
								&messageDDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentDMessage[currentDisparity]));
						vst1q_f32(
								&messageLDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentLMessage[currentDisparity]));
						vst1q_f32(
								&messageRDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentRMessage[currentDisparity]));
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						vst1q_f32(
								&messageUDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentUMessage[currentDisparity]));
						vst1q_f32(
								&messageDDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentDMessage[currentDisparity]));
						vst1q_f32(
								&messageLDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentLMessage[currentDisparity]));
						vst1q_f32(
								&messageRDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentRMessage[currentDisparity]));
					}
				}
			} else {
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(
							xValProcess, yVal,
							currentLevelProperties.paddedWidthCheckerboardLevel,
							currentLevelProperties.heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardToUpdate == CHECKERBOARD_PART_1) {
						 vst1_f16(
								&messageUDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentUMessage[currentDisparity]));
						 vst1_f16(
								&messageDDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentDMessage[currentDisparity]));
						 vst1_f16(
								&messageLDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentLMessage[currentDisparity]));
						 vst1_f16(
								&messageRDeviceCurrentCheckerboard1[indexWriteTo],
								vcvt_f16_f32(currentRMessage[currentDisparity]));
					} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						 vst1_f16(
								&messageUDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentUMessage[currentDisparity]));
						 vst1_f16(
								&messageDDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentDMessage[currentDisparity]));
						 vst1_f16(
								&messageLDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentLMessage[currentDisparity]));
						 vst1_f16(
								&messageRDeviceCurrentCheckerboard2[indexWriteTo],
								vcvt_f16_f32(currentRMessage[currentDisparity]));
					}
				}
			}
		}
	}
}


#endif /* KERNELBPSTEREOCPU_NEON_H_ */
