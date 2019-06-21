#include "KernelBpStereoCPU.h"

#ifndef KERNELBPSTEREOCPU_TEMPLATESPFUNCTS
#define KERNELBPSTEREOCPU_TEMPLATESPFUNCTS

template<> inline
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU(float* image1PixelsDevice, float* image2PixelsDevice, short* dataCostDeviceStereoCheckerboard1, short* dataCostDeviceStereoCheckerboard2, int widthImages, int heightImages, float lambda_bp, float data_k_bp)
{
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	int imageCheckerboardWidth = getCheckerboardWidthCPU<short>(widthImages);
	float* dataCostDeviceStereoCheckerboard1Float = new float[imageCheckerboardWidth*heightImages*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostDeviceStereoCheckerboard2Float = new float[imageCheckerboardWidth*heightImages*NUM_POSSIBLE_DISPARITY_VALUES];

	initializeBottomLevelDataStereoCPU<float>(image1PixelsDevice, image2PixelsDevice, dataCostDeviceStereoCheckerboard1Float, dataCostDeviceStereoCheckerboard2Float, widthImages, heightImages, lambda_bp, data_k_bp);

	convertFloatToShortAVX256(dataCostDeviceStereoCheckerboard1, dataCostDeviceStereoCheckerboard1Float, imageCheckerboardWidth, heightImages);
	convertFloatToShortAVX256(dataCostDeviceStereoCheckerboard2, dataCostDeviceStereoCheckerboard2Float, imageCheckerboardWidth, heightImages);

	delete [] dataCostDeviceStereoCheckerboard1Float;
	delete [] dataCostDeviceStereoCheckerboard2Float;

#else

	printf("ERROR, SHORT DATA TYPE NOT SUPPORTED IF NOT USING AVX-256\n");

#endif
}


template<> inline
void KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU(short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* dataCostDeviceToWriteTo, int widthCurrentLevel, int heightCurrentLevel, int widthPrevLevel, int heightPrevLevel, int checkerboardPart, int offsetNum)
{
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<short>(widthCurrentLevel);
	int widthCheckerboardPrevLevel = getCheckerboardWidthCPU<short>(widthPrevLevel);

	float* dataCostDeviceStereoCheckerboard1Float = new float[widthCheckerboardPrevLevel*heightPrevLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostDeviceStereoCheckerboard2Float = new float[widthCheckerboardPrevLevel*heightPrevLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostDeviceToWriteToFloat = new float[widthCheckerboardCurrentLevel*heightCurrentLevel*NUM_POSSIBLE_DISPARITY_VALUES];

	convertShortToFloatAVX256(dataCostDeviceStereoCheckerboard1Float, dataCostStereoCheckerboard1, widthCheckerboardPrevLevel, heightPrevLevel);
	convertShortToFloatAVX256(dataCostDeviceStereoCheckerboard2Float, dataCostStereoCheckerboard2, widthCheckerboardPrevLevel, heightPrevLevel);

	initializeCurrentLevelDataStereoNoTexturesCPU<float>(
			dataCostDeviceStereoCheckerboard1Float, dataCostDeviceStereoCheckerboard2Float,
			dataCostDeviceToWriteToFloat, widthCurrentLevel, heightCurrentLevel,
			widthPrevLevel, heightPrevLevel, checkerboardPart, offsetNum);

	convertFloatToShortAVX256(dataCostDeviceToWriteTo, dataCostDeviceToWriteToFloat, widthCheckerboardCurrentLevel, heightCurrentLevel);

	delete [] dataCostDeviceStereoCheckerboard1Float;
	delete [] dataCostDeviceStereoCheckerboard2Float;
	delete [] dataCostDeviceToWriteToFloat;

#else

	printf("ERROR, SHORT DATA TYPE NOT SUPPORTED IF NOT USING AVX-256\n");

#endif
}


template<> inline
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU(short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1, short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1, short* messageUDeviceCurrentCheckerboard2, short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2, short* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel)
{
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	float* messageUDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageUDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];

	initializeMessageValsToDefaultKernelCPU<float>(messageUDeviceCurrentCheckerboard1Float, messageDDeviceCurrentCheckerboard1Float, messageLDeviceCurrentCheckerboard1Float,
			messageRDeviceCurrentCheckerboard1Float, messageUDeviceCurrentCheckerboard2Float, messageDDeviceCurrentCheckerboard2Float,
			messageLDeviceCurrentCheckerboard2Float, messageRDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);

	convertFloatToShortAVX256(messageUDeviceCurrentCheckerboard1, messageUDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageDDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageLDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageRDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageUDeviceCurrentCheckerboard2, messageUDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageDDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageLDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShortAVX256(messageRDeviceCurrentCheckerboard2, messageRDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);

	delete [] messageUDeviceCurrentCheckerboard1Float;
	delete [] messageDDeviceCurrentCheckerboard1Float;
	delete [] messageLDeviceCurrentCheckerboard1Float;
	delete [] messageRDeviceCurrentCheckerboard1Float;
	delete [] messageUDeviceCurrentCheckerboard2Float;
	delete [] messageDDeviceCurrentCheckerboard2Float;
	delete [] messageLDeviceCurrentCheckerboard2Float;
	delete [] messageRDeviceCurrentCheckerboard2Float;

#else

	printf("ERROR, SHORT DATA TYPE NOT SUPPORTED IF NOT USING AVX-256\n");

#endif
}


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(float* dataCostStereoCheckerboard1,
		float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1,
		float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1,
		float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2,
		float* messageDDeviceCurrentCheckerboard2,
		float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX256(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#elif CPU_OPTIMIZATION_SETTING == USE_AVX_512

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX512(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
				messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);
#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<float>(dataCostStereoCheckerboard1,
							dataCostStereoCheckerboard2,
							messageUDeviceCurrentCheckerboard1,
							messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1,
							messageRDeviceCurrentCheckerboard1,
							messageUDeviceCurrentCheckerboard2,
							messageDDeviceCurrentCheckerboard2,
							messageLDeviceCurrentCheckerboard2,
							messageRDeviceCurrentCheckerboard2,
							widthCheckerboardCurrentLevel, heightLevel,
							checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
				//}
			//}
		}
#endif
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(short* dataCostStereoCheckerboard1,
				short* dataCostStereoCheckerboard2,
				short* messageUDeviceCurrentCheckerboard1,
				short* messageDDeviceCurrentCheckerboard1,
				short* messageLDeviceCurrentCheckerboard1,
				short* messageRDeviceCurrentCheckerboard1,
				short* messageUDeviceCurrentCheckerboard2,
				short* messageDDeviceCurrentCheckerboard2,
				short* messageLDeviceCurrentCheckerboard2,
				short* messageRDeviceCurrentCheckerboard2, int widthLevel,
				int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUShortUseAVX256(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#else

	printf("ERROR, SHORT DATA TYPE NOT SUPPORTED IF NOT USING AVX-256\n");

#endif
}

template<> inline
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(double* dataCostStereoCheckerboard1,
		double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1,
		double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1,
		double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2,
		double* messageDDeviceCurrentCheckerboard2,
		double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUDoubleUseAVX256(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<double>(dataCostStereoCheckerboard1,
							dataCostStereoCheckerboard2,
							messageUDeviceCurrentCheckerboard1,
							messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1,
							messageRDeviceCurrentCheckerboard1,
							messageUDeviceCurrentCheckerboard2,
							messageDDeviceCurrentCheckerboard2,
							messageLDeviceCurrentCheckerboard2,
							messageRDeviceCurrentCheckerboard2,
							widthCheckerboardCurrentLevel, heightLevel,
							checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
				//}
			//}
		}
#endif
}


template<> inline
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoNoTexturesCPU(short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1, short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, short* messageUPrevStereoCheckerboard2, short* messageDPrevStereoCheckerboard2, short* messageLPrevStereoCheckerboard2, short* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	int widthCheckerboard = getCheckerboardWidthCPU<short>(widthLevel);
	float* dataCostStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageUPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageUPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];

	convertShortToFloatAVX256(dataCostStereoCheckerboard1Float, dataCostStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(dataCostStereoCheckerboard2Float, dataCostStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageUPrevStereoCheckerboard1Float, messageUPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageDPrevStereoCheckerboard1Float, messageDPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageLPrevStereoCheckerboard1Float, messageLPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageRPrevStereoCheckerboard1Float, messageRPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageUPrevStereoCheckerboard2Float, messageUPrevStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageDPrevStereoCheckerboard2Float, messageDPrevStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageLPrevStereoCheckerboard2Float, messageLPrevStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloatAVX256(messageRPrevStereoCheckerboard2Float, messageRPrevStereoCheckerboard2, widthCheckerboard, heightLevel);

	retrieveOutputDisparityCheckerboardStereoNoTexturesCPU<float>(
			dataCostStereoCheckerboard1Float, dataCostStereoCheckerboard2Float,
			messageUPrevStereoCheckerboard1Float, messageDPrevStereoCheckerboard1Float,
			messageLPrevStereoCheckerboard1Float, messageRPrevStereoCheckerboard1Float,
			messageUPrevStereoCheckerboard2Float, messageDPrevStereoCheckerboard2Float,
			messageLPrevStereoCheckerboard2Float, messageRPrevStereoCheckerboard2Float,
			disparityBetweenImagesDevice, widthLevel, heightLevel);

	delete [] dataCostStereoCheckerboard1Float;
	delete [] dataCostStereoCheckerboard2Float;
	delete [] messageUPrevStereoCheckerboard1Float;
	delete [] messageDPrevStereoCheckerboard1Float;
	delete [] messageLPrevStereoCheckerboard1Float;
	delete [] messageRPrevStereoCheckerboard1Float;
	delete [] messageUPrevStereoCheckerboard2Float;
	delete [] messageDPrevStereoCheckerboard2Float;
	delete [] messageLPrevStereoCheckerboard2Float;
	delete [] messageRPrevStereoCheckerboard2Float;

#else

	printf("ERROR, SHORT DATA TYPE NOT SUPPORTED IF NOT USING AVX-256\n");

#endif
}

#endif //KERNELBPSTEREOCPU_TEMPLATESPFUNCTS

