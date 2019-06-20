/*
 * RunBpStereoSet.h
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#ifndef RUNBPSTEREOSET_H_
#define RUNBPSTEREOSET_H_

#include "bpStereoParameters.h"
#include "SmoothImage.h"
#include "ProcessBPOnTargetDevice.h"
#include <cstring>
#include "imageHelpers.h"
#include "DetailedTimings.h"

class RunBpStereoSetMemoryManagement
{
public:
	RunBpStereoSetMemoryManagement() {
			// TODO Auto-generated constructor stub

		}

		virtual ~RunBpStereoSetMemoryManagement() {
			// TODO Auto-generated destructor stub
		}

	virtual void allocateDataOnCompDevice(void** arrayToAllocate, int numBytes)
	{
		//allocate the space for the disparity map estimation
		*arrayToAllocate = malloc(numBytes);
	}

	virtual void freeDataOnCompDevice(void** arrayToFree)
	{
		free(*arrayToFree);
	}

	virtual void transferDataFromCompDeviceToHost(void* destArray, void* inArray, int numBytesTransfer)
	{
		memcpy(destArray, inArray, numBytesTransfer);
	}

	void transferDataFromCompHostToDevice(void* destArray, void* inArray, int numBytesTransfer)
	{
		memcpy(destArray, inArray, numBytesTransfer);
	}
};

template <typename T>
class RunBpStereoSet {
public:
	RunBpStereoSet() {
		// TODO Auto-generated constructor stub

	}

	virtual ~RunBpStereoSet() {
		// TODO Auto-generated destructor stub
	}

	virtual float operator()(const char* refImagePath, const char* testImagePath,
			BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<T>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr)
	{
		if (runBPMemoryMangement == nullptr)
		{
			runBPMemoryMangement = new RunBpStereoSetMemoryManagement();
		}
		double timeNoTransfer = 0.0;
		double timeIncludeTransfer = 0.0;

		unsigned int heightImages = 0;
		unsigned int widthImages = 0;

		std::vector<double> timingsNoTransferVector;
		std::vector<double> timingsIncludeTransferVector;
		DetailedTimings* timingsOverall = nullptr;
		for (int numRun = 0; numRun < NUM_BP_STEREO_RUNS; numRun++)
		{
			//first run Stereo estimation on the first two images
			unsigned int* image1AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
					refImagePath, widthImages, heightImages);
			unsigned int* image2AsUnsignedIntArrayHost = ImageHelperFunctions::loadImageAsGrayScale(
					testImagePath, widthImages, heightImages);

			float* smoothedImage1;
			float* smoothedImage2;

			//allocate the device memory to store and x and y smoothed images
			runBPMemoryMangement->allocateDataOnCompDevice((void**)&smoothedImage1, widthImages * heightImages * sizeof(float));
			runBPMemoryMangement->allocateDataOnCompDevice((void**)&smoothedImage2, widthImages * heightImages * sizeof(float));

			//start timer to retrieve the time of implementation including transfer time
			auto timeWithTransferStart = std::chrono::system_clock::now();

			//start timer to retrieve the time of implementation not including transfer time
			auto timeNoTransferStart = std::chrono::system_clock::now();

			//first smooth the images using the Gaussian filter with the given SIGMA_BP value
			//smoothed images are stored on the target device at locations smoothedImage1 and smoothedImage2
			(*smoothImage)(image1AsUnsignedIntArrayHost,
						widthImages, heightImages, algSettings.smoothingSigma, smoothedImage1);
			(*smoothImage)(image2AsUnsignedIntArrayHost,
						widthImages, heightImages, algSettings.smoothingSigma, smoothedImage2);

			//free the host memory allocatted to original image 1 and image 2 on the host
			delete[] image1AsUnsignedIntArrayHost;
			delete[] image2AsUnsignedIntArrayHost;

			//set the width and height parameters of the imag
			algSettings.widthImages = widthImages;
			algSettings.heightImages = heightImages;

			float* disparityMapFromImage1To2CompDevice;

			//allocate the space for the disparity map estimation
			runBPMemoryMangement->allocateDataOnCompDevice((void**)&disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

			//ProcessCUDABP<beliefPropProcessingDataType> processBPOnGPUUsingCUDA;
			DetailedTimings* timings = (*runBpStereo)(smoothedImage1, smoothedImage2,
					disparityMapFromImage1To2CompDevice, algSettings);

			if (timings != nullptr)
			{
				if (timingsOverall == nullptr)
				{
					timingsOverall = timings;
				}
				else
				{
					timingsOverall->addTimings(timings);
					//TODO: add returned timings to overall set
				}
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
			runBPMemoryMangement->transferDataFromCompDeviceToHost(disparityMapFromImage1To2Host, disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

			auto timeWithTransferEnd = std::chrono::system_clock::now();

			std::chrono::duration<double> diffWithTransferTime = timeWithTransferEnd
					- timeWithTransferStart;
			timeIncludeTransfer = diffWithTransferTime.count();

			//save the resulting disparity map images to a file
			ImageHelperFunctions::saveDisparityImageToPGM(saveDisparityMapImagePath,
						SCALE_BP, disparityMapFromImage1To2Host,
						widthImages, heightImages);

			delete[] disparityMapFromImage1To2Host;
			timingsIncludeTransferVector.push_back(timeIncludeTransfer);

			//free the space allocated to the resulting disparity map and smoothed images on the computation device
			runBPMemoryMangement->freeDataOnCompDevice((void**)&disparityMapFromImage1To2CompDevice);
			runBPMemoryMangement->freeDataOnCompDevice((void**)&smoothedImage1);
			runBPMemoryMangement->freeDataOnCompDevice((void**)&smoothedImage2);

			//printf("RUN: %d\n", numRun);
		}

		//printf("DONE\n");

		fprintf(resultsFile, "Image Width: %d\n", widthImages);
		fprintf(resultsFile, "Image Height: %d\n", heightImages);
		fprintf(resultsFile, "Total Image Pixels: %d\n", widthImages * heightImages);

		if (timingsOverall != nullptr)
		{
			//uncomment to print timings for each part of implementation
			//timings.PrintMedianTimings();
			timingsOverall->PrintMedianTimingsToFile(resultsFile);
		}

		std::sort(timingsNoTransferVector.begin(), timingsNoTransferVector.end());
		std::sort(timingsIncludeTransferVector.begin(), timingsIncludeTransferVector.end());

		printf("Median runtime (not including transfer of data to/from comp device memory): %f\n", timingsNoTransferVector.at(NUM_BP_STEREO_RUNS/2));
		//printf("MEDIAN GPU RUN TIME (INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2));
		fprintf(resultsFile, "MEDIAN GPU RUN RUNTIME (NOT INCLUDING TRANSFER TIME OF DATA TO/FROM comp device MEMORY): %f\n", timingsNoTransferVector.at(NUM_BP_STEREO_RUNS/2));
		fprintf(resultsFile, "MEDIAN GPU RUN RUNTIME (INCLUDING TRANSFER TIME OF DATA TO/FROM comp device MEMORY): %f\n", timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2));

		return timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2);
	}
};

#endif /* RUNBPSTEREOSET_H_ */
