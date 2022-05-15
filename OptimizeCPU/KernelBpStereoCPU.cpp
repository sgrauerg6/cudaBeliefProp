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

//This file defines the methods to perform belief propagation for disparity map estimation from stereo images on CUDA


#include "KernelBpStereoCPU.h"
#include "../SharedFuncts/SharedBPProcessingFuncts.h"

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU(
		const levelProperties& currentLevelProperties,
		float* image1PixelsDevice, float* image2PixelsDevice,
		T* dataCostDeviceStereoCheckerboard0, T* dataCostDeviceStereoCheckerboard1,
		const float lambda_bp, const float data_k_bp, const unsigned int bpSettingsDispVals)
{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, currentLevelProperties.widthLevel_*currentLevelProperties.heightLevel_,
	  [&currentLevelProperties, &image1PixelsDevice, &image2PixelsDevice, &dataCostDeviceStereoCheckerboard0,
	  &dataCostDeviceStereoCheckerboard1, &lambda_bp, &data_k_bp, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / currentLevelProperties.widthLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthLevel_;

		      initializeBottomLevelDataStereoPixel<T, DISP_VALS>(xVal, yVal, currentLevelProperties,
		  	      image1PixelsDevice, image2PixelsDevice,
			      dataCostDeviceStereoCheckerboard0, dataCostDeviceStereoCheckerboard1,
		          lambda_bp, data_k_bp, bpSettingsDispVals);
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, currentLevelProperties.widthLevel_*currentLevelProperties.heightLevel_,
	  [&currentLevelProperties, &image1PixelsDevice, &image2PixelsDevice, &dataCostDeviceStereoCheckerboard0,
	  &dataCostDeviceStereoCheckerboard1, &lambda_bp, &data_k_bp, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / currentLevelProperties.widthLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthLevel_;

		      initializeBottomLevelDataStereoPixel<T, DISP_VALS>(xVal, yVal, currentLevelProperties,
		  	      image1PixelsDevice, image2PixelsDevice,
			      dataCostDeviceStereoCheckerboard0, dataCostDeviceStereoCheckerboard1,
		          lambda_bp, data_k_bp, bpSettingsDispVals);
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	#pragma omp parallel for
#ifdef _WIN32
	for (int val = 0; val < (currentLevelProperties.widthLevel_*currentLevelProperties.heightLevel_); val++)
#else
	for (unsigned int val = 0; val < (currentLevelProperties.widthLevel_*currentLevelProperties.heightLevel_); val++)
#endif //_WIN32
	{
		const unsigned int yVal = val / currentLevelProperties.widthLevel_;
		const unsigned int xVal = val % currentLevelProperties.widthLevel_;

		initializeBottomLevelDataStereoPixel<T, DISP_VALS>(xVal, yVal, currentLevelProperties,
				image1PixelsDevice, image2PixelsDevice,
				dataCostDeviceStereoCheckerboard0, dataCostDeviceStereoCheckerboard1,
				lambda_bp, data_k_bp, bpSettingsDispVals);
	}
	#endif //CPU_PARALLELIZATION_METHOD
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::initializeCurrentLevelDataStereoCPU(
		const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties,
		const levelProperties& prevLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* dataCostDeviceToWriteTo, const unsigned int offsetNum, const unsigned int bpSettingsDispVals)
{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&checkerboardPart, &currentLevelProperties, &prevLevelProperties, &dataCostStereoCheckerboard0,
	  &dataCostStereoCheckerboard1, &dataCostDeviceToWriteTo, &offsetNum, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		      initializeCurrentLevelDataStereoPixel<T, T, DISP_VALS>(
				xVal, yVal, checkerboardPart,
				currentLevelProperties, prevLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&checkerboardPart, &currentLevelProperties, &prevLevelProperties, &dataCostStereoCheckerboard0,
	  &dataCostStereoCheckerboard1, &dataCostDeviceToWriteTo, &offsetNum, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		      initializeCurrentLevelDataStereoPixel<T, T, DISP_VALS>(
				xVal, yVal, checkerboardPart,
				currentLevelProperties, prevLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	#pragma omp parallel for
#ifdef _WIN32
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#else
	for (unsigned int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#endif //_WIN32
	{
		const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		initializeCurrentLevelDataStereoPixel<T, T, DISP_VALS>(
				xVal, yVal, checkerboardPart,
				currentLevelProperties, prevLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				dataCostDeviceToWriteTo, offsetNum, bpSettingsDispVals);
	}
	#endif //CPU_PARALLELIZATION_METHOD
}


//initialize the message values at each pixel of the current level to the default value
template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU(const levelProperties& currentLevelProperties,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const unsigned int bpSettingsDispVals)
{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&currentLevelProperties, &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
	  &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	  &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xValInCheckerboard = val % currentLevelProperties.widthCheckerboardLevel_;

		      initializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
				xValInCheckerboard, yVal, currentLevelProperties,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				bpSettingsDispVals);
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&currentLevelProperties, &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
	  &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	  &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xValInCheckerboard = val % currentLevelProperties.widthCheckerboardLevel_;

		      initializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
				xValInCheckerboard, yVal, currentLevelProperties,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				bpSettingsDispVals);
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	#pragma omp parallel for
#ifdef _WIN32
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#else
	for (unsigned int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#endif //_WIN32
	{
		const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		const unsigned int xValInCheckerboard = val % currentLevelProperties.widthCheckerboardLevel_;

		initializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
				xValInCheckerboard, yVal, currentLevelProperties,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				bpSettingsDispVals);
	}
	#endif //CPU_PARALLELIZATION_METHOD
}


template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions(
		const Checkerboard_Parts checkerboardPartUpdate, const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		float disc_k_bp, const unsigned int bpSettingsDispVals)
{
	const unsigned int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel_ / 2;

	//in cuda kernel storing data one at a time (though it is coalesced), so numDataInSIMDVector not relevant here and set to 1
	//still is a check if start of row is aligned
	const bool dataAligned = MemoryAlignedAtDataStart(0, 1);

	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, widthCheckerboardRunProcessing * currentLevelProperties.heightLevel_,
	  [&checkerboardPartUpdate, &currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1,
	  &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
	  &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	  &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &disc_k_bp, &bpSettingsDispVals,
	  &widthCheckerboardRunProcessing, &dataAligned](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / widthCheckerboardRunProcessing;
		      const unsigned int xVal = val % widthCheckerboardRunProcessing;

		      runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<T, T, DISP_VALS>(
				xVal, yVal, checkerboardPartUpdate, currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				disc_k_bp, 0, dataAligned, bpSettingsDispVals);
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, widthCheckerboardRunProcessing * currentLevelProperties.heightLevel_,
	  [&checkerboardPartUpdate, &currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1,
	  &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
	  &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	  &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &disc_k_bp, &bpSettingsDispVals,
	  &widthCheckerboardRunProcessing, &dataAligned](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / widthCheckerboardRunProcessing;
		      const unsigned int xVal = val % widthCheckerboardRunProcessing;

		      runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<T, T, DISP_VALS>(
				xVal, yVal, checkerboardPartUpdate, currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				disc_k_bp, 0, dataAligned, bpSettingsDispVals);
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	#pragma omp parallel for
#ifdef _WIN32
	for (int val = 0; val < (widthCheckerboardRunProcessing * currentLevelProperties.heightLevel_); val++)
#else
	for (unsigned int val = 0; val < (widthCheckerboardRunProcessing * currentLevelProperties.heightLevel_); val++)
#endif //_WIN32
	{
		const unsigned int yVal = val / widthCheckerboardRunProcessing;
		const unsigned int xVal = val % widthCheckerboardRunProcessing;

		runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemPixel<T, T, DISP_VALS>(
				xVal, yVal, checkerboardPartUpdate, currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				disc_k_bp, 0, dataAligned, bpSettingsDispVals);
	}
	#endif //CPU_PARALLELIZATION_METHOD
}

template<typename T, typename U, unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(
		const unsigned int xValStartProcessing, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		U prevUMessage[DISP_VALS], U prevDMessage[DISP_VALS],
		U prevLMessage[DISP_VALS], U prevRMessage[DISP_VALS], U dataMessage[DISP_VALS],
		T* currentUMessageArray, T* currentDMessageArray, T* currentLMessageArray, T* currentRMessageArray,
		const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing)
{
	msgStereoSIMD<T, U, DISP_VALS>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevLMessage, prevRMessage, dataMessage, currentUMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);

	msgStereoSIMD<T, U, DISP_VALS>(xValStartProcessing, yVal, currentLevelProperties, prevDMessage,
			prevLMessage, prevRMessage, dataMessage, currentDMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);

	msgStereoSIMD<T, U, DISP_VALS>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevDMessage, prevRMessage, dataMessage, currentRMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);

	msgStereoSIMD<T, U, DISP_VALS>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevDMessage, prevLMessage, dataMessage, currentLMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing);
}

template<typename T, typename U>
void KernelBpStereoCPU::runBPIterationInOutDataInLocalMemCPUUseSIMDVectors(
		const unsigned int xValStartProcessing, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		U* prevUMessage, U* prevDMessage, U* prevLMessage, U* prevRMessage, U* dataMessage,
		T* currentUMessageArray, T* currentDMessageArray, T* currentLMessageArray, T* currentRMessageArray,
		const U disc_k_bp_vector, const bool dataAlignedAtxValStartProcessing, const unsigned int bpSettingsDispVals)
{
	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevLMessage, prevRMessage, dataMessage, currentUMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing, bpSettingsDispVals);

	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevDMessage,
			prevLMessage, prevRMessage, dataMessage, currentDMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing, bpSettingsDispVals);

	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevDMessage, prevRMessage, dataMessage, currentRMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing, bpSettingsDispVals);

	msgStereoSIMD<T, U>(xValStartProcessing, yVal, currentLevelProperties, prevUMessage,
			prevDMessage, prevLMessage, dataMessage, currentLMessageArray,
			disc_k_bp_vector, dataAlignedAtxValStartProcessing, bpSettingsDispVals);
}

template<typename T, typename U, unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectorsProcess(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int numDataInSIMDVector,
		const unsigned int bpSettingsDispVals)
{
	const unsigned int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel_ / 2;
	const U disc_k_bp_vector = createSIMDVectorSameData<U>(disc_k_bp);

	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	  if constexpr (DISP_VALS > 0) {
		KernelBpStereoCPU::tPool.parallelize_loop(1, currentLevelProperties.heightLevel_ - 1,
		[&checkerboardToUpdate, &currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1,
	     &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
    	 &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	     &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &disc_k_bp, &numDataInSIMDVector,
	     &bpSettingsDispVals, &widthCheckerboardRunProcessing, &disc_k_bp_vector](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int yVal = a; yVal < b; yVal++) {
			//checkerboardAdjustment used for indexing into current checkerboard to update
			const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//initialize arrays for data and message values
				U dataMessage[DISP_VALS], prevUMessage[DISP_VALS], prevDMessage[DISP_VALS], prevLMessage[DISP_VALS], prevRMessage[DISP_VALS];

				//load using aligned instructions when possible
				if (dataAlignedAtXValProcess) {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard1);
						}
						else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard0);
						}
					}
				} else {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard1);
						} else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard0);
						}
					}
				}

				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U, DISP_VALS>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
							messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
							disc_k_bp_vector, dataAlignedAtXValProcess);
				}
				else {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U, DISP_VALS>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							disc_k_bp_vector, dataAlignedAtXValProcess);
				}
			}
		}
	  });
	}
	else {
        KernelBpStereoCPU::tPool.parallelize_loop(1, currentLevelProperties.heightLevel_ - 1,
		[&checkerboardToUpdate, &currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1,
	     &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
    	 &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	     &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &disc_k_bp, &numDataInSIMDVector,
	     &bpSettingsDispVals, &widthCheckerboardRunProcessing, &disc_k_bp_vector](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int yVal = a; yVal < b; yVal++) {
			//checkerboardAdjustment used for indexing into current checkerboard to update
			const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//initialize arrays for data and message values
				U* dataMessage = new U[bpSettingsDispVals];
				U* prevUMessage = new U[bpSettingsDispVals];
				U* prevDMessage = new U[bpSettingsDispVals];
				U* prevLMessage = new U[bpSettingsDispVals];
				U* prevRMessage = new U[bpSettingsDispVals];

				//load using aligned instructions when possible
				if (dataAlignedAtXValProcess) {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard1);
						}
						else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard0);
						}
					}
				} else {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard1);
						} else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard0);
						}
					}
				}

				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
							messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
							disc_k_bp_vector, dataAlignedAtXValProcess, bpSettingsDispVals);
				}
				else {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							disc_k_bp_vector, dataAlignedAtXValProcess, bpSettingsDispVals);
				}

				delete [] dataMessage;
				delete [] prevUMessage;
				delete [] prevDMessage;
				delete [] prevLMessage;
				delete [] prevRMessage;
			}
		}
	  });
	}
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	  if constexpr (DISP_VALS > 0) {
		KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(1, currentLevelProperties.heightLevel_ - 1,
		[&checkerboardToUpdate, &currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1,
	     &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
    	 &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	     &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &disc_k_bp, &numDataInSIMDVector,
	     &bpSettingsDispVals, &widthCheckerboardRunProcessing, &disc_k_bp_vector](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int yVal = a; yVal < b; yVal += numBlocks) {
			//checkerboardAdjustment used for indexing into current checkerboard to update
			const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//initialize arrays for data and message values
				U dataMessage[DISP_VALS], prevUMessage[DISP_VALS], prevDMessage[DISP_VALS], prevLMessage[DISP_VALS], prevRMessage[DISP_VALS];

				//load using aligned instructions when possible
				if (dataAlignedAtXValProcess) {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard1);
						}
						else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard0);
						}
					}
				} else {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard1);
						} else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard0);
						}
					}
				}

				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U, DISP_VALS>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
							messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
							disc_k_bp_vector, dataAlignedAtXValProcess);
				}
				else {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U, DISP_VALS>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							disc_k_bp_vector, dataAlignedAtXValProcess);
				}
			}
		}
	  });
	}
	else {
        KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(1, currentLevelProperties.heightLevel_ - 1,
		[&checkerboardToUpdate, &currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1,
	     &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0, &messageLDeviceCurrentCheckerboard0,
    	 &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	     &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &disc_k_bp, &numDataInSIMDVector,
	     &bpSettingsDispVals, &widthCheckerboardRunProcessing, &disc_k_bp_vector](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int yVal = a; yVal < b; yVal += numBlocks) {
			//checkerboardAdjustment used for indexing into current checkerboard to update
			const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//initialize arrays for data and message values
				U* dataMessage = new U[bpSettingsDispVals];
				U* prevUMessage = new U[bpSettingsDispVals];
				U* prevDMessage = new U[bpSettingsDispVals];
				U* prevLMessage = new U[bpSettingsDispVals];
				U* prevRMessage = new U[bpSettingsDispVals];

				//load using aligned instructions when possible
				if (dataAlignedAtXValProcess) {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard1);
						}
						else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard0);
						}
					}
				} else {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard1);
						} else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard0);
						}
					}
				}

				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
							messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
							disc_k_bp_vector, dataAlignedAtXValProcess, bpSettingsDispVals);
				}
				else {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							disc_k_bp_vector, dataAlignedAtXValProcess, bpSettingsDispVals);
				}

				delete [] dataMessage;
				delete [] prevUMessage;
				delete [] prevDMessage;
				delete [] prevLMessage;
				delete [] prevRMessage;
			}
		}
	  });
	}
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	if constexpr (DISP_VALS > 0) {
		#pragma omp parallel for
#ifdef _WIN32
		for (int yVal = 1; yVal < currentLevelProperties.heightLevel_ - 1; yVal++) {
#else
		for (unsigned int yVal = 1; yVal < currentLevelProperties.heightLevel_ - 1; yVal++) {
#endif //_WIN32
			//checkerboardAdjustment used for indexing into current checkerboard to update
			const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//initialize arrays for data and message values
				U dataMessage[DISP_VALS], prevUMessage[DISP_VALS], prevDMessage[DISP_VALS], prevLMessage[DISP_VALS], prevRMessage[DISP_VALS];

				//load using aligned instructions when possible
				if (dataAlignedAtXValProcess) {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard1);
						}
						else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard0);
						}
					}
				} else {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard1);
						} else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, DISP_VALS, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, messageRDeviceCurrentCheckerboard0);
						}
					}
				}

				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U, DISP_VALS>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
							messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
							disc_k_bp_vector, dataAlignedAtXValProcess);
				}
				else {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U, DISP_VALS>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							disc_k_bp_vector, dataAlignedAtXValProcess);
				}
			}
		}
	}
	else {
		#pragma omp parallel for
#ifdef _WIN32
		for (int yVal = 1; yVal < currentLevelProperties.heightLevel_ - 1; yVal++) {
#else
		for (unsigned int yVal = 1; yVal < currentLevelProperties.heightLevel_ - 1; yVal++) {
#endif //_WIN32
			//checkerboardAdjustment used for indexing into current checkerboard to update
			const unsigned int checkerboardAdjustment = (checkerboardToUpdate == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//initialize arrays for data and message values
				U* dataMessage = new U[bpSettingsDispVals];
				U* prevUMessage = new U[bpSettingsDispVals];
				U* prevDMessage = new U[bpSettingsDispVals];
				U* prevLMessage = new U[bpSettingsDispVals];
				U* prevRMessage = new U[bpSettingsDispVals];

				//load using aligned instructions when possible
				if (dataAlignedAtXValProcess) {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard1);
						}
						else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard0);
						}
					}
				} else {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard1);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard1);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard1);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard1);
						} else //checkerboardPartUpdate == CHECKERBOARD_PART_1
						{
							dataMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
							prevUMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUDeviceCurrentCheckerboard0);
							prevDMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDDeviceCurrentCheckerboard0);
							prevLMessage[currentDisparity] = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLDeviceCurrentCheckerboard0);
							prevRMessage[currentDisparity] = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
									currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRDeviceCurrentCheckerboard0);
						}
					}
				}

				if (checkerboardToUpdate == CHECKERBOARD_PART_0) {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
							messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
							disc_k_bp_vector, dataAlignedAtXValProcess, bpSettingsDispVals);
				}
				else {
					runBPIterationInOutDataInLocalMemCPUUseSIMDVectors<T, U>(xValProcess, yVal, currentLevelProperties,
							prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
							disc_k_bp_vector, dataAlignedAtXValProcess, bpSettingsDispVals);
				}

				delete [] dataMessage;
				delete [] prevUMessage;
				delete [] prevDMessage;
				delete [] prevLMessage;
				delete [] prevRMessage;
			}
		}
	}
#endif //CPU_PARALLELIZATION_METHOD
}

//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesCPU(
		const Checkerboard_Parts checkerboardToUpdate, const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const float disc_k_bp, const unsigned int bpSettingsNumDispVals)
{
	if constexpr (CPU_OPTIMIZATION_SETTING == cpu_vectorization_setting::USE_AVX_256)
	{
		//only use AVX-256 if width of processing checkerboard is over 10
		if (currentLevelProperties.widthCheckerboardLevel_ > 10)
		{
			runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<DISP_VALS>(checkerboardToUpdate,
					currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
					messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					disc_k_bp, bpSettingsNumDispVals);
		}
		else
		{
			runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T, DISP_VALS>(checkerboardToUpdate,
					currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
					messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					disc_k_bp, bpSettingsNumDispVals);
		}
	}
	else if constexpr (CPU_OPTIMIZATION_SETTING == cpu_vectorization_setting::USE_AVX_512)
	{
		//only use AVX-512 if width of processing checkerboard is over 20
		if (currentLevelProperties.widthCheckerboardLevel_ > 20)
		{
			runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<DISP_VALS>(checkerboardToUpdate,
					currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
					messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					disc_k_bp, bpSettingsNumDispVals);
		}
		else
		{
			runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T, DISP_VALS>(checkerboardToUpdate,
					currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
					messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					disc_k_bp, bpSettingsNumDispVals);
		}
	}
	else if constexpr (CPU_OPTIMIZATION_SETTING == cpu_vectorization_setting::USE_NEON)
	{
		if (currentLevelProperties.widthCheckerboardLevel_ > 5)
		{
			runBPIterationUsingCheckerboardUpdatesCPUUseSIMDVectors<DISP_VALS>(checkerboardToUpdate,
					currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
					messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					disc_k_bp, bpSettingsNumDispVals);
		}
		else
		{
			runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T, DISP_VALS>(checkerboardToUpdate,
					currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
					messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
					messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
					messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
					disc_k_bp, bpSettingsNumDispVals);
		}
	}
	else
	{
		runBPIterationUsingCheckerboardUpdatesCPUNoPackedInstructions<T, DISP_VALS>(checkerboardToUpdate,
				currentLevelProperties, dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				disc_k_bp, bpSettingsNumDispVals);
	}
}


//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoCPU(const Checkerboard_Parts checkerboardPart,
		const levelProperties& currentLevelProperties, const levelProperties& nextLevelProperties,
		T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
		T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
		const unsigned int bpSettingsDispVals)
{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&checkerboardPart, &currentLevelProperties, &nextLevelProperties, &messageUPrevStereoCheckerboard0, &messageDPrevStereoCheckerboard0,
	  &messageLPrevStereoCheckerboard0, &messageRPrevStereoCheckerboard0, &messageUPrevStereoCheckerboard1, &messageDPrevStereoCheckerboard1,
	  &messageLPrevStereoCheckerboard1, &messageRPrevStereoCheckerboard1, &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0,
	  &messageLDeviceCurrentCheckerboard0, &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	  &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

     		  copyPrevLevelToNextLevelBPCheckerboardStereoPixel<T, DISP_VALS>(xVal, yVal,
				checkerboardPart, currentLevelProperties, nextLevelProperties,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				bpSettingsDispVals);
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&checkerboardPart, &currentLevelProperties, &nextLevelProperties, &messageUPrevStereoCheckerboard0, &messageDPrevStereoCheckerboard0,
	  &messageLPrevStereoCheckerboard0, &messageRPrevStereoCheckerboard0, &messageUPrevStereoCheckerboard1, &messageDPrevStereoCheckerboard1,
	  &messageLPrevStereoCheckerboard1, &messageRPrevStereoCheckerboard1, &messageUDeviceCurrentCheckerboard0, &messageDDeviceCurrentCheckerboard0,
	  &messageLDeviceCurrentCheckerboard0, &messageRDeviceCurrentCheckerboard0, &messageUDeviceCurrentCheckerboard1, &messageDDeviceCurrentCheckerboard1,
	  &messageLDeviceCurrentCheckerboard1, &messageRDeviceCurrentCheckerboard1, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

     		  copyPrevLevelToNextLevelBPCheckerboardStereoPixel<T, DISP_VALS>(xVal, yVal,
				checkerboardPart, currentLevelProperties, nextLevelProperties,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				bpSettingsDispVals);
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	#pragma omp parallel for
#ifdef _WIN32
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#else
	for (unsigned int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#endif //_WIN32
	{
		const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		copyPrevLevelToNextLevelBPCheckerboardStereoPixel<T, DISP_VALS>(xVal, yVal,
				checkerboardPart, currentLevelProperties, nextLevelProperties,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				messageUDeviceCurrentCheckerboard0, messageDDeviceCurrentCheckerboard0,
				messageLDeviceCurrentCheckerboard0, messageRDeviceCurrentCheckerboard0,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1,
				messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				bpSettingsDispVals);
	}
	#endif //CPU_PARALLELIZATION_METHOD
}

template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoOptimizedCPU(
		const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
		T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals) {
	retrieveOutputDisparityCheckerboardStereoOptimizedCPUUseSIMDVectors<DISP_VALS>(currentLevelProperties,
	    dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
		messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
		messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
		messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
		messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
		disparityBetweenImagesDevice, bpSettingsDispVals);
    /*#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1, &messageUPrevStereoCheckerboard0, &messageDPrevStereoCheckerboard0,
	  &messageLPrevStereoCheckerboard0, &messageRPrevStereoCheckerboard0, &messageUPrevStereoCheckerboard1, &messageDPrevStereoCheckerboard1,
	  &messageLPrevStereoCheckerboard1, &messageRPrevStereoCheckerboard1, &disparityBetweenImagesDevice, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		      retrieveOutputDisparityCheckerboardStereoOptimizedPixel<T, T, DISP_VALS>(
				xVal, yVal, currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				disparityBetweenImagesDevice, bpSettingsDispVals);
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_,
	  [&currentLevelProperties, &dataCostStereoCheckerboard0, &dataCostStereoCheckerboard1, &messageUPrevStereoCheckerboard0, &messageDPrevStereoCheckerboard0,
	  &messageLPrevStereoCheckerboard0, &messageRPrevStereoCheckerboard0, &messageUPrevStereoCheckerboard1, &messageDPrevStereoCheckerboard1,
	  &messageLPrevStereoCheckerboard1, &messageRPrevStereoCheckerboard1, &disparityBetweenImagesDevice, &bpSettingsDispVals](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		      const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		      retrieveOutputDisparityCheckerboardStereoOptimizedPixel<T, T, DISP_VALS>(
				xVal, yVal, currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				disparityBetweenImagesDevice, bpSettingsDispVals);
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
	#pragma omp parallel for
#ifdef _WIN32
	for (int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#else
	for (unsigned int val = 0; val < (currentLevelProperties.widthCheckerboardLevel_*currentLevelProperties.heightLevel_); val++)
#endif //_WIN32
	{
		const unsigned int yVal = val / currentLevelProperties.widthCheckerboardLevel_;
		const unsigned int xVal = val % currentLevelProperties.widthCheckerboardLevel_;

		retrieveOutputDisparityCheckerboardStereoOptimizedPixel<T, T, DISP_VALS>(
				xVal, yVal, currentLevelProperties,
				dataCostStereoCheckerboard0, dataCostStereoCheckerboard1,
				messageUPrevStereoCheckerboard0, messageDPrevStereoCheckerboard0,
				messageLPrevStereoCheckerboard0, messageRPrevStereoCheckerboard0,
				messageUPrevStereoCheckerboard1, messageDPrevStereoCheckerboard1,
				messageLPrevStereoCheckerboard1, messageRPrevStereoCheckerboard1,
				disparityBetweenImagesDevice, bpSettingsDispVals);
	}
	#endif //CPU_PARALLELIZATION_METHOD*/
}

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel using SIMD vectors
template<typename T, typename U, typename V, typename W, unsigned int DISP_VALS>
void KernelBpStereoCPU::retrieveOutDispOptimizedCPUUseSIMDVectorsProcess(const levelProperties& currentLevelProperties,
	T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
	T* messageUPrevStereoCheckerboard0, T* messageDPrevStereoCheckerboard0,
	T* messageLPrevStereoCheckerboard0, T* messageRPrevStereoCheckerboard0,
	T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
	T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
	float* disparityBetweenImagesDevice, const unsigned int bpSettingsDispVals,
	const unsigned int numDataInSIMDVector)
{
	const unsigned int widthCheckerboardRunProcessing = currentLevelProperties.widthLevel_ / 2;

	//initially get output for each checkerboard
	//set width of disparity checkerboard to be a multiple of numDataInSIMDVector so that SIMD vectors can be aligned
	unsigned int widthDispCheckerboard = ((currentLevelProperties.paddedWidthCheckerboardLevel_ % bp_params::NUM_DATA_ALIGN_WIDTH) == 0) ?
	   currentLevelProperties.paddedWidthCheckerboardLevel_  :
	   currentLevelProperties.paddedWidthCheckerboardLevel_ + 
	   (bp_params::NUM_DATA_ALIGN_WIDTH - (currentLevelProperties.paddedWidthCheckerboardLevel_ % bp_params::NUM_DATA_ALIGN_WIDTH));
	const unsigned int numDataDispChBoard = widthDispCheckerboard * currentLevelProperties.heightLevel_;
#ifdef _WIN32
		V* disparityCheckboard0 = static_cast<V*>(_aligned_malloc(2 * numDataDispChBoard * sizeof(V), bp_params::NUM_DATA_ALIGN_WIDTH * sizeof(V)));
#else
		V* disparityCheckboard0 = static_cast<V*>(std::aligned_alloc(bp_params::NUM_DATA_ALIGN_WIDTH * sizeof(V), 2 * numDataDispChBoard * sizeof(V)));
#endif

    for (auto checkerboardGetDispMap : {Checkerboard_Parts::CHECKERBOARD_PART_0, Checkerboard_Parts::CHECKERBOARD_PART_1})
	{
	#pragma omp parallel for
#ifdef _WIN32
		for (int yVal = 1; yVal < currentLevelProperties.heightLevel_ - 1; yVal++) {
#else
		for (unsigned int yVal = 1; yVal < currentLevelProperties.heightLevel_ - 1; yVal++) {
#endif //_WIN32
			//checkerboardAdjustment used for indexing into current checkerboard to retrieve best disparities
			const unsigned int checkerboardAdjustment = (checkerboardGetDispMap == CHECKERBOARD_PART_0) ? ((yVal) % 2) : ((yVal + 1) % 2);
			const unsigned int startX = (checkerboardAdjustment == 1) ? 0 : 1;
			const unsigned int endFinal = std::min(currentLevelProperties.widthCheckerboardLevel_ - checkerboardAdjustment,
					widthCheckerboardRunProcessing);
			const unsigned int endXSIMDVectorStart = (endFinal / numDataInSIMDVector) * numDataInSIMDVector - numDataInSIMDVector;

			for (unsigned int xVal = 0; xVal < endFinal; xVal += numDataInSIMDVector) {
				unsigned int xValProcess = xVal;

				//need this check first for case where endXAvxStart is 0 and startX is 1
				//if past the last AVX start (since the next one would go beyond the row),
				//set to numDataInSIMDVector from the final pixel so processing the last numDataInAvxVector in avx
				//may be a few pixels that are computed twice but that's OK
				if (xValProcess > endXSIMDVectorStart) {
					xValProcess = endFinal - numDataInSIMDVector;
				}

				//not processing at x=0 if startX is 1 (this will cause this processing to be less aligned than ideal for this iteration)
				xValProcess = std::max(startX, xValProcess);

                //get index for output into disparity map corresponding to checkerboard
				const unsigned int indexOutput = (yVal * widthDispCheckerboard) + xValProcess;

				//check if the memory is aligned for AVX instructions at xValProcess location
				const bool dataAlignedAtXValProcess = MemoryAlignedAtDataStart(xValProcess, numDataInSIMDVector);

				//declare SIMD vectors for data and message values at each disparity
				//U dataMessage, prevUMessage, prevDMessage, prevLMessage, prevRMessage;

				//declare SIMD vectors for current best values and best disparities
				W bestVals, bestDisparities, valAtDisp;

				//load using aligned instructions when possible
				if constexpr (DISP_VALS > 0) {
					for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
						if (checkerboardGetDispMap == CHECKERBOARD_PART_0) {
							if (dataAlignedAtXValProcess) {
								valAtDisp = addVals<U, U, W>(
									loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard1),
									loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0));
								/*dataMessage = loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
								prevUMessage = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard1);
								prevDMessage = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard1);*/
							}
							else {
								valAtDisp = addVals<U, U, W>(
									loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard1),
									loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0));
								/*dataMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
									currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard0);
								prevUMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard1);
								prevDMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard1);*/
							}
							/*valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard1));
							valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard1));*/
							/*prevLMessage = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard1);
							prevRMessage = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard1);*/
						}
						else //checkerboardGetDispMap == CHECKERBOARD_PART_1
						{
							if (dataAlignedAtXValProcess) {
								valAtDisp = addVals<U, U, W>(
									loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard0),
									loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1));
								/*dataMessage = loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
								prevUMessage = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard0);
								prevDMessage = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard0);*/
							}
							else {
								valAtDisp = addVals<U, U, W>(
									loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard0),
								    loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
								    loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1));
								/*dataMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, dataCostStereoCheckerboard1);
								prevUMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageUPrevStereoCheckerboard0);
								prevDMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, DISP_VALS, messageDPrevStereoCheckerboard0);*/
							}
							/*valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard0));
							valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard0));*/
							/*prevLMessage = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageLPrevStereoCheckerboard0);
							prevRMessage = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, DISP_VALS, messageRPrevStereoCheckerboard0);*/
						}
						//add message and data values together
						/*W valAtDisp = addVals<U, U, W>(prevUMessage, prevDMessage);
						valAtDisp = addVals<W, U, W>(valAtDisp, prevLMessage);
						valAtDisp = addVals<W, U, W>(valAtDisp, prevRMessage);
						valAtDisp = addVals<W, U, W>(valAtDisp, dataMessage);*/
						if (currentDisparity == 0) {
							bestVals = valAtDisp;
							//set disp at min vals to all 0
							bestDisparities = createSIMDVectorSameData<W>(0.0f);
						}
						else {
							//update best disparity and best values
							//if value at current disparity is lower than current best value, need
							//to update best value to current value and set best disparity to current disparity
							updateBestDispBestVals(bestDisparities, bestVals, createSIMDVectorSameData<W>((float)currentDisparity), valAtDisp);
						}
					}
					if (dataAlignedAtXValProcess) {
						if (checkerboardGetDispMap == CHECKERBOARD_PART_0) {
							storePackedDataAligned<V, W>(indexOutput, disparityCheckboard0, bestDisparities);
						}
						else //checkerboardGetDispMap == CHECKERBOARD_PART_1
						{
							storePackedDataAligned<V, W>(numDataDispChBoard + indexOutput, disparityCheckboard0, bestDisparities);
						}
					}
					else {
						if (checkerboardGetDispMap == CHECKERBOARD_PART_0) {
							storePackedDataUnaligned<V, W>(indexOutput, disparityCheckboard0, bestDisparities);
						}
						else //checkerboardGetDispMap == CHECKERBOARD_PART_1
						{
							storePackedDataUnaligned<V, W>(numDataDispChBoard + indexOutput, disparityCheckboard0, bestDisparities);
						}
					}
				}
				else {
					for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++) {
						if (checkerboardGetDispMap == CHECKERBOARD_PART_0) {
							if (dataAlignedAtXValProcess) {
								valAtDisp = addVals<U, U, W>(
									loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard1),
									loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp,
								    loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0));
								/*dataMessage = loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
								prevUMessage = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard1);
								prevDMessage = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard1);*/	
							}
							else {
								valAtDisp = addVals<U, U, W>(
									loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard1),
								    loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard1));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0));
								/*dataMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard0);
								prevUMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard1);
								prevDMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard1);*/
							}
							/*valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard1));
							valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard1));*/
							/*prevLMessage = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard1);
							prevRMessage = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard1);*/
						}
						else //checkerboardGetDispMap == CHECKERBOARD_PART_1
						{
							if (dataAlignedAtXValProcess) {
								valAtDisp = addVals<U, U, W>( 
									loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard0),
									loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
								    loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1));
								/*dataMessage = loadPackedDataAligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
								prevUMessage = loadPackedDataAligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard0);
								prevDMessage = loadPackedDataAligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard0);*/
							}
							else {
								valAtDisp = addVals<U, U, W>( 
									loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard0),
									loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard0));
								valAtDisp = addVals<W, U, W>(valAtDisp,
								    loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1));
								/*dataMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, dataCostStereoCheckerboard1);
								prevUMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal + 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageUPrevStereoCheckerboard0);
								prevDMessage = loadPackedDataUnaligned<T, U>(xValProcess, yVal - 1,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageDPrevStereoCheckerboard0);*/
							}
							/*valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard0));
							valAtDisp = addVals<W, U, W>(valAtDisp, 
									loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard0));*/
							/*prevLMessage = loadPackedDataUnaligned<T, U>(xValProcess + checkerboardAdjustment, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageLPrevStereoCheckerboard0);
							prevRMessage = loadPackedDataUnaligned<T, U>((xValProcess + checkerboardAdjustment) - 1, yVal,
										currentDisparity, currentLevelProperties, bpSettingsDispVals, messageRPrevStereoCheckerboard0);*/
						}
						//add message and data values together
						/*W valAtDisp = addVals<U, U, W>(prevUMessage, prevDMessage);
						valAtDisp = addVals<W, U, W>(valAtDisp, prevLMessage);
						valAtDisp = addVals<W, U, W>(valAtDisp, prevRMessage);
						valAtDisp = addVals<W, U, W>(valAtDisp, dataMessage);*/
						if (currentDisparity == 0) {
							bestVals = valAtDisp;
							//set disp at min vals to all 0
							bestDisparities = createSIMDVectorSameData<W>(0.0f);
						}
						else {
							//update best disparity and best values
							//if value at current disparity is lower than current best value, need
							//to update best value to current value and set best disparity to current disparity
							updateBestDispBestVals(bestDisparities, bestVals, createSIMDVectorSameData<W>((float)currentDisparity), valAtDisp);
						}
					}
					//store best disparities in checkerboard being updated
					if (dataAlignedAtXValProcess) {
						if (checkerboardGetDispMap == CHECKERBOARD_PART_0) {
							storePackedDataAligned<V, W>(indexOutput, disparityCheckboard0, bestDisparities);
						}
						else //checkerboardGetDispMap == CHECKERBOARD_PART_1
						{
							storePackedDataAligned<V, W>(numDataDispChBoard + indexOutput, disparityCheckboard0, bestDisparities);
						}
					}
					else {
						if (checkerboardGetDispMap == CHECKERBOARD_PART_0) {
							storePackedDataUnaligned<V, W>(indexOutput, disparityCheckboard0, bestDisparities);
						}
						else //checkerboardGetDispMap == CHECKERBOARD_PART_1
						{
							storePackedDataUnaligned<V, W>(numDataDispChBoard + indexOutput, disparityCheckboard0, bestDisparities);
						}
					}
				}
			}
		}
	}

	//combine output disparity maps from each checkerboard
	//start with checkerboard 0 in first row since (0, 0) corresponds to (0, 0)
	//in checkerboard 0 and (1, 0) corresponds to (0, 0) in checkerboard 1
	#pragma omp parallel for
#ifdef _WIN32
	for (int y=0; y < currentLevelProperties.heightLevel_; y++)
#else
	for (unsigned int y=0; y < currentLevelProperties.heightLevel_; y++)
#endif //_WIN32
	{
		const bool startChBoard0 = ((y%2) == 0);
		unsigned int chBoardIdx = y * widthDispCheckerboard;
		for (unsigned int x=0; x < (currentLevelProperties.widthLevel_); x += 2) {
			if ((y == 0) || (y == (currentLevelProperties.heightLevel_ - 1))) {
				disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 0)] = 0;
				disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 1)] = 0;
			}
			else {
				if (startChBoard0) {
					if ((x == 0) || (x == (currentLevelProperties.widthLevel_ - 1))) {
						disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 0)] = 0;
					}
					else {
					    disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 0)] =
						  (float)disparityCheckboard0[chBoardIdx];
					}
					if ((x + 1) == (currentLevelProperties.widthLevel_ - 1)) {
                        disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 1)] = 0;
					}
					else if ((x + 1) < currentLevelProperties.widthLevel_) {
					  disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 1)] =
					  	  (float)disparityCheckboard0[numDataDispChBoard + chBoardIdx];
					}
				}
				else {
					if ((x == 0) || (x == (currentLevelProperties.widthLevel_ - 1))) {
						disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 0)] = 0;
					}
					else {
					    disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 0)] =
							(float)disparityCheckboard0[numDataDispChBoard + chBoardIdx];
					}
					if ((x + 1) == (currentLevelProperties.widthLevel_ - 1)) {
                        disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 1)] = 0;
					}
					else if ((x + 1) < currentLevelProperties.widthLevel_) {
					    disparityBetweenImagesDevice[y * currentLevelProperties.widthLevel_ + (x + 1)] =
						    (float)disparityCheckboard0[chBoardIdx];
					}
				}
				//increment checkerboard index for next x-value
			    chBoardIdx++;
			}
		}
	}
		
	//delete [] disparityCheckboard0;
	free(disparityCheckboard0);
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T, typename U, unsigned int DISP_VALS>
void KernelBpStereoCPU::dtStereoSIMD(U f[DISP_VALS])
{
	U prev;
	const U vectorAllOneVal = convertValToDatatype<U, T>(1.0f);
	for (unsigned int currentDisparity = 1; currentDisparity < DISP_VALS; currentDisparity++)
	{
		//prev = f[currentDisparity-1] + (T)1.0;
		prev = addVals<U, U, U>(f[currentDisparity - 1], vectorAllOneVal);

		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = getMinByElement<U>(prev, f[currentDisparity]);
	}

	for (int currentDisparity = (int)DISP_VALS-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = addVals<U, U, U>(f[currentDisparity + 1], vectorAllOneVal);

		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = getMinByElement<U>(prev, f[currentDisparity]);
	}
}

// compute current message
template<typename T, typename U, typename V, typename W, unsigned int DISP_VALS>
void KernelBpStereoCPU::msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		U messageValsNeighbor1[DISP_VALS],
		U messageValsNeighbor2[DISP_VALS],
		U messageValsNeighbor3[DISP_VALS],
		U dataCosts[DISP_VALS],
		T* dstMessageArray, const U& disc_k_bp, const bool dataAligned)
{
	// aggregate and find min
	//T minimum = bp_consts::INF_BP;
	W minimum = convertValToDatatype<W, V>(bp_consts::INF_BP);
	W dst[DISP_VALS];

	for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = addVals<U, U, W>(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = addVals<W, U, W>(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = addVals<W, U, W>(dst[currentDisparity], dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = getMinByElement<W>(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<V, W, DISP_VALS>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = addVals<W, U, W>(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	W valToNormalize = convertValToDatatype<W, V>(0.0);

	for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity]) {
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = getMinByElement<W>(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = addVals<W, W, W>(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= DISP_VALS;
	valToNormalize = divideVals<W, W, W>(valToNormalize, convertValToDatatype<W, V>((double)DISP_VALS));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
				currentLevelProperties.paddedWidthCheckerboardLevel_,
				currentLevelProperties.heightLevel_, 0,
				DISP_VALS);

	for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = subtractVals<W, W, W>(dst[currentDisparity], valToNormalize);

		if (dataAligned) {
			storePackedDataAligned<T, W>(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}
		else {
			storePackedDataUnaligned<T, W>(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}

		if constexpr (OPTIMIZED_INDEXING_SETTING) {
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
		}
		else {
			destMessageArrayIndex++;
		}
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
//TODO: look into defining function in .cpp file so don't need to declare inline
template<typename T, typename U>
void KernelBpStereoCPU::dtStereoSIMD(U* f, const unsigned int bpSettingsDispVals)
{
	U prev;
	const U vectorAllOneVal = convertValToDatatype<U, T>(1.0f);
	for (unsigned int currentDisparity = 1; currentDisparity < bpSettingsDispVals; currentDisparity++)
	{
		//prev = f[currentDisparity-1] + (T)1.0;
		prev = addVals<U, U, U>(f[currentDisparity - 1], vectorAllOneVal);

		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = getMinByElement<U>(prev, f[currentDisparity]);
	}

	for (int currentDisparity = (int)bpSettingsDispVals-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = addVals<U, U, U>(f[currentDisparity + 1], vectorAllOneVal);

		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = getMinByElement<U>(prev, f[currentDisparity]);
	}
}

// compute current message
template<typename T, typename U, typename V, typename W>
void KernelBpStereoCPU::msgStereoSIMDProcessing(const unsigned int xVal, const unsigned int yVal,
	const levelProperties& currentLevelProperties,
	U* messageValsNeighbor1, U* messageValsNeighbor2,
	U* messageValsNeighbor3, U* dataCosts,
	T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
	const unsigned int bpSettingsDispVals)
{
	// aggregate and find min
	//T minimum = bp_consts::INF_BP;
	W minimum = convertValToDatatype<W, V>(bp_consts::INF_BP);
	W* dst = new W[bpSettingsDispVals];

	for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
	{
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		dst[currentDisparity] = addVals<U, U, W>(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = addVals<W, U, W>(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = addVals<W, U, W>(dst[currentDisparity], dataCosts[currentDisparity]);

		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = getMinByElement<W>(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoSIMD<V, W>(dst, bpSettingsDispVals);

	// truncate
	//minimum += disc_k_bp;
	minimum = addVals<W, U, W>(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	W valToNormalize = convertValToDatatype<W, V>(0.0f);

	for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
	{
		//if (minimum < dst[currentDisparity]) {
		//	dst[currentDisparity] = minimum;
		//}
		dst[currentDisparity] = getMinByElement<W>(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = addVals<W, W, W>(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= DISP_VALS;
	valToNormalize = divideVals<W, W, W>(valToNormalize, convertValToDatatype<W, V>((float)bpSettingsDispVals));

	unsigned int destMessageArrayIndex = retrieveIndexInDataAndMessage(xVal, yVal,
			currentLevelProperties.paddedWidthCheckerboardLevel_,
			currentLevelProperties.heightLevel_, 0,
			bpSettingsDispVals);

	for (unsigned int currentDisparity = 0; currentDisparity < bpSettingsDispVals; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = subtractVals<W, W, W>(dst[currentDisparity], valToNormalize);

		if (dataAligned) {
			storePackedDataAligned<T, W>(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}
		else {
			storePackedDataUnaligned<T, W>(destMessageArrayIndex,
					dstMessageArray, dst[currentDisparity]);
		}

		if constexpr (OPTIMIZED_INDEXING_SETTING) {
			destMessageArrayIndex += currentLevelProperties.paddedWidthCheckerboardLevel_;
		}
		else {
			destMessageArrayIndex++;
		}
	}

	delete [] dst;
}

// compute current message
template<typename T, typename U, unsigned int DISP_VALS>
void KernelBpStereoCPU::msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		U messageValsNeighbor1[DISP_VALS],
		U messageValsNeighbor2[DISP_VALS],
		U messageValsNeighbor3[DISP_VALS],
		U dataCosts[DISP_VALS],
		T* dstMessageArray, const U& disc_k_bp, const bool dataAligned)
{
	msgStereoSIMDProcessing<T, U, T, U, DISP_VALS>(xVal, yVal,
			currentLevelProperties, messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts, dstMessageArray, disc_k_bp, dataAligned);
}

// compute current message
template<typename T, typename U>
void KernelBpStereoCPU::msgStereoSIMD(const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		U* messageValsNeighbor1, U* messageValsNeighbor2,
		U* messageValsNeighbor3, U* dataCosts,
		T* dstMessageArray, const U& disc_k_bp, const bool dataAligned,
		const unsigned int bpSettingsDispVals)
{
	msgStereoSIMDProcessing<T, U, T, U>(
			xVal, yVal, currentLevelProperties,
			messageValsNeighbor1, messageValsNeighbor2,
			messageValsNeighbor3, dataCosts,
			dstMessageArray, disc_k_bp, dataAligned,
			bpSettingsDispVals);
}

template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %u\n", xVal);
		printf("yVal: %u\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %u\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	} else {
		printf("xVal: %u\n", xVal);
		printf("yVal: %u\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %u\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	}
}

template<typename T, unsigned int DISP_VALS>
void KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU(
		const unsigned int xVal, const unsigned int yVal,
		const levelProperties& currentLevelProperties,
		T* dataCostStereoCheckerboard0, T* dataCostStereoCheckerboard1,
		T* messageUDeviceCurrentCheckerboard0, T* messageDDeviceCurrentCheckerboard0,
		T* messageLDeviceCurrentCheckerboard0, T* messageRDeviceCurrentCheckerboard0,
		T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1)
{
	const unsigned int checkerboardAdjustment = (((xVal + yVal) % 2) == 0) ? ((yVal)%2) : ((yVal+1)%2);
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %u\n", xVal);
		printf("yVal: %u\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %u\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessage(
							(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	}
	else {
		printf("xVal: %u\n", xVal);
		printf("yVal: %u\n", yVal);
		for (unsigned int currentDisparity = 0; currentDisparity < DISP_VALS; currentDisparity++) {
			printf("DISP: %u\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal + 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2, yVal - 1, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							xVal / 2 + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard0[retrieveIndexInDataAndMessage(
							(xVal / 2 - 1) + checkerboardAdjustment, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessage(
							xVal / 2, yVal, currentLevelProperties.paddedWidthCheckerboardLevel_, currentLevelProperties.heightLevel_,
							currentDisparity, DISP_VALS)]);
		}
	}
}
