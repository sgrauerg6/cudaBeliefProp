/*
 * RunBpStereoOptimizedCPU.cpp
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#include "RunBpStereoOptimizedCPU.h"
#include "BpStereoProcessingOptimizedCPU.cpp"

RunBpStereoOptimizedCPU::RunBpStereoOptimizedCPU() {
	// TODO Auto-generated constructor stub

}

RunBpStereoOptimizedCPU::~RunBpStereoOptimizedCPU() {
	// TODO Auto-generated destructor stub
}

float RunBpStereoOptimizedCPU::operator()(const char* refImagePath, const char* testImagePath,
				BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTarget<beliefPropProcessingDataTypeCPU>* runBpStereo)
	{
		SmoothImageCPU smoothImageCPU;
		BpStereoProcessingOptimizedCPU<beliefPropProcessingDataTypeCPU> processImageCPU;
		return RunBpStereoSet::operator ()(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsFile, &smoothImageCPU, &processImageCPU);
	}

/*float RunBpStereoOptimizedCPU::operator()(const char* refImagePath, const char* testImagePath, BPsettings algSettings, const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTarget<beliefPropProcessingDataTypeCPU>* runBpStereo)
{
	double timeStart = 0.0;

	unsigned int heightImages = 0;
	unsigned int widthImages = 0;

	std::vector<double> runTimings;
	for (int numRun = 0; numRun < NUM_BP_STEREO_RUNS; numRun++)
	{
		//first run Stereo estimation on the first two images
		unsigned int* image1AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
				refImagePath, widthImages, heightImages);
		unsigned int* image2AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
				testImagePath, widthImages, heightImages);

		float* smoothedImage1 = new float[widthImages * heightImages];;
		float* smoothedImage2 = new float[widthImages * heightImages];;

		//start timer to retrieve the time of implementation including transfer time
		auto timeStart = std::chrono::system_clock::now();

		//first smooth the images using the CUDA Gaussian filter with the given SIGMA_BP value
		//smoothed images are stored global memory on the device at locations image1SmoothedDevice and image2SmoothedDevice
		SmoothImageCPU smooth_image;
		smooth_image(image1AsUnsignedIntArrayHost,
				widthImages, heightImages, algSettings.smoothingSigma, smoothedImage1);
		smooth_image(image2AsUnsignedIntArrayHost,
				widthImages, heightImages, algSettings.smoothingSigma, smoothedImage2);

		printf("Smooth images done\n");

		//free the host memory allocatted to original image 1 and image 2 on the host
		delete[] image1AsUnsignedIntArrayHost;
		delete[] image2AsUnsignedIntArrayHost;

		//set the width and height parameters of the imag
		algSettings.widthImages = widthImages;
		algSettings.heightImages = heightImages;

		//allocate the space for the disparity map estimation
		float* disparityMapFromImage1To2 = new float[widthImages * heightImages];

		printf("Start processing\n");
		(*runBpStereo)(smoothedImage1, smoothedImage2,
				disparityMapFromImage1To2, algSettings);
		/*BpStereoProcessingOptimizedCPU<beliefPropProcessingDataTypeCPU> processBPOnCPUOptimized;
		processBPOnCPUOptimized(smoothedImage1, smoothedImage2,
				disparityMapFromImage1To2, algSettings);*/

		//retrieve the runtime for implementation
	/*	auto timeEnd = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = timeEnd
				- timeStart;
		double runTime = diff.count();
		runTimings.push_back(runTime);

		//save the resulting disparity map images to a file
		ImageHelperFunctions::saveDisparityImageToPGM(saveDisparityMapImagePath,
				SCALE_BP, disparityMapFromImage1To2,
				widthImages, heightImages);

		//free the space allocated to the resulting disparity map
		delete [] disparityMapFromImage1To2;

		//free the space allocated to the smoothed images
		delete [] smoothedImage1;
		delete [] smoothedImage2;
	}

	fprintf(resultsFile, "Image Width: %d\n", widthImages);
	fprintf(resultsFile, "Image Height: %d\n", heightImages);
	fprintf(resultsFile, "Total Image Pixels: %d\n", widthImages * heightImages);

	std::sort(runTimings.begin(), runTimings.end());
	int nthreads = 0;
	#pragma omp parallel
	{
		nthreads = omp_get_num_threads();
	}
	printf("Number of OMP threads: %d\n", nthreads);
	fprintf(resultsFile, "Number of OMP threads: %d\n", nthreads);
	printf("Median optimized CPU runtime: %f\n", runTimings.at(NUM_BP_STEREO_RUNS/2));
	//printf("MEDIAN GPU RUN TIME (INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2));
	fprintf(resultsFile, "Median optimized CPU runtime: %f\n", runTimings.at(NUM_BP_STEREO_RUNS/2));

	return runTimings.at(NUM_BP_STEREO_RUNS/2);
}*/
