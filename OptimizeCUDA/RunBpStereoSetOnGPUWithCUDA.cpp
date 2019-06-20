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

//Defines the methods to run BP Stereo implementation on a series of images using various options

#include "RunBpStereoSetOnGPUWithCUDA.h"
#include "SmoothImageCUDA.h"
#include <chrono>
#include <vector>
#include <algorithm>
#include "imageHelpers.h"
#include "DetailedTimings.h"
#include "ProcessCUDABP.cpp"

float RunBpStereoSetOnGPUWithCUDA::operator()(const char* refImagePath, const char* testImagePath,
				BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTarget<beliefPropProcessingDataTypeCPU>* runBpStereo)
	{
		SmoothImageCUDA smoothImageCUDA;
		ProcessCUDABP<beliefPropProcessingDataTypeCPU> processImageCUDA;
		return RunBpStereoSet::operator ()(refImagePath, testImagePath, algSettings, saveDisparityMapImagePath, resultsFile, &smoothImageCUDA, &processImageCUDA);
	}

//run the disparity map estimation BP on a stereo image set and save the results between each set of images if desired
//returns the runtime (including transfer time)
/*float RunBpStereoSetOnGPUWithCUDA::operator()(const char* refImagePath, const char* testImagePath,
		BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage, ProcessBPOnTarget<beliefPropProcessingDataTypeCUDA>* runBpStereo)
{
	double timeNoTransfer = 0.0;
	double timeIncludeTransfer = 0.0;

	unsigned int heightImages = 0;
	unsigned int widthImages = 0;

	std::vector<double> timingsNoTransferVector;
	std::vector<double> timingsIncludeTransferVector;
	DetailedTimings timings;
	for (int numRun = 0; numRun < NUM_BP_STEREO_RUNS; numRun++)
	{
		//first run Stereo estimation on the first two images
		unsigned int* image1AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
				refImagePath, widthImages, heightImages);
		unsigned int* image2AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
				testImagePath, widthImages, heightImages);

		float* smoothedImage1Device;
		float* smoothedImage2Device;

		//allocate the device memory to store and x and y smoothed images
		allocateDataOnCompDevice((void**)&smoothedImage1Device, widthImages * heightImages * sizeof(float));
		allocateDataOnCompDevice((void**)&smoothedImage2Device, widthImages * heightImages * sizeof(float));
//		(cudaMalloc((void**) &smoothedImage1Device,
//				widthImages * heightImages * sizeof(float)));
//		(cudaMalloc((void**) &smoothedImage2Device,
//				widthImages * heightImages * sizeof(float)));

		//start timer to retrieve the time of implementation including transfer time
		auto timeWithTransferStart = std::chrono::system_clock::now();

		//start timer to retrieve the time of implementation not including transfer time
		auto timeNoTransferStart = std::chrono::system_clock::now();

		if (smoothImage != nullptr)
		{
			//first smooth the images using the CUDA Gaussian filter with the given SIGMA_BP value
			//smoothed images are stored global memory on the device at locations image1SmoothedDevice and image2SmoothedDevice
			//SmoothImageCUDA smooth_image;
			(*smoothImage)(image1AsUnsignedIntArrayHost,
					widthImages, heightImages, algSettings.smoothingSigma, smoothedImage1Device);
			(*smoothImage)(image2AsUnsignedIntArrayHost,
					widthImages, heightImages, algSettings.smoothingSigma, smoothedImage2Device);
		}

		//free the host memory allocatted to original image 1 and image 2 on the host
		delete[] image1AsUnsignedIntArrayHost;
		delete[] image2AsUnsignedIntArrayHost;

		//set the width and height parameters of the imag
		algSettings.widthImages = widthImages;
		algSettings.heightImages = heightImages;

		float* disparityMapFromImage1To2CompDevice;

		//allocate the space for the disparity map estimation
		allocateDataOnCompDevice((void**)&disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		if (runBpStereo != nullptr)
		{
			//ProcessCUDABP<beliefPropProcessingDataTypeCUDA> processBPOnGPUUsingCUDA;
			(*runBpStereo)(smoothedImage1Device, smoothedImage2Device,
					disparityMapFromImage1To2CompDevice, algSettings);
		}

		//retrieve the running time of the implementation not including the host/device transfer time
		//printf("Running time not including transfer time: %f (ms) \n", cutGetTimerValue(timerTransferTimeNotIncluded));
		auto timeNoTransferEnd = std::chrono::system_clock::now();
		std::chrono::duration<double> diff = timeNoTransferEnd-timeNoTransferStart;
		//printf("Running time not including transfer time: %.10lf seconds\n",
		//		timeEnd - timeStart);
		timeNoTransfer += diff.count();
		timingsNoTransferVector.push_back(diff.count());

		//allocate the space on the host for and x and y movement between images
		float* disparityMapFromImage1To2Host = new float[widthImages
				* heightImages];

		//transfer the disparity map estimation on the device to the host for output
//		(cudaMemcpy(disparityMapFromImage1To2Host,
//				disparityMapFromImage1To2Device,
//				widthImages * heightImages * sizeof(float),
//				cudaMemcpyDeviceToHost));
		transferDataFromCompDeviceToHost(disparityMapFromImage1To2Host, disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

		auto timeWithTransferEnd = std::chrono::system_clock::now();

		//printf("Running time including transfer time: %.10lf seconds\n", timeEnd-timeStart);
		std::chrono::duration<double> diffWithTransferTime = timeWithTransferEnd
				- timeWithTransferStart;
		timeIncludeTransfer = diffWithTransferTime.count();
		//stop the timer and print the total time of the BP implementation including the device-host transfer time
		//printf("Time to retrieve movement on host (including transfer): %f (ms) \n", totalTimeIncludeTransfer);

		//save the resulting disparity map images to a file
		ImageHelperFunctions::saveDisparityImageToPGM(saveDisparityMapImagePath,
				SCALE_BP, disparityMapFromImage1To2Host,
				widthImages, heightImages);

		delete[] disparityMapFromImage1To2Host;

		timingsIncludeTransferVector.push_back(timeIncludeTransfer);

		//free the space allocated to the resulting disparity map
		//cudaFree(disparityMapFromImage1To2Device);

		//free the space allocated to the smoothed images on the device
		//cudaFree(smoothedImage1Device);
		//cudaFree(smoothedImage2Device);
		freeDataOnCompDevice((void**)&disparityMapFromImage1To2CompDevice);
		freeDataOnCompDevice((void**)&smoothedImage1Device);
		freeDataOnCompDevice((void**)&smoothedImage2Device);
	}

	fprintf(resultsFile, "Image Width: %d\n", widthImages);
	fprintf(resultsFile, "Image Height: %d\n", heightImages);
	fprintf(resultsFile, "Total Image Pixels: %d\n", widthImages * heightImages);

	//uncomment to print timings for each part of implementation
	//timings.PrintMedianTimings();
	timings.PrintMedianTimingsToFile(resultsFile);

	std::sort(timingsNoTransferVector.begin(), timingsNoTransferVector.end());
	std::sort(timingsIncludeTransferVector.begin(), timingsIncludeTransferVector.end());

	printf("Median CUDA runtime (not including transfer of data to/from GPU memory): %f\n", timingsNoTransferVector.at(NUM_BP_STEREO_RUNS/2));
	//printf("MEDIAN GPU RUN TIME (INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2));
	fprintf(resultsFile, "MEDIAN GPU RUN RUNTIME (NOT INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", timingsNoTransferVector.at(NUM_BP_STEREO_RUNS/2));
	fprintf(resultsFile, "MEDIAN GPU RUN RUNTIME (INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2));

	return timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2);
}*/
