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
	}

	virtual ~RunBpStereoSet() {
	}

	virtual float operator()(const char* refImagePath, const char* testImagePath,
			BPsettings algSettings,	const char* saveDisparityMapImagePath, FILE* resultsFile, SmoothImage* smoothImage = nullptr, ProcessBPOnTargetDevice<T>* runBpStereo = nullptr, RunBpStereoSetMemoryManagement* runBPMemoryMangement = nullptr)
	{
		if (runBPMemoryMangement == nullptr)
		{
			runBPMemoryMangement = new RunBpStereoSetMemoryManagement();
		}

		unsigned int heightImages = 0;
		unsigned int widthImages = 0;

		std::vector<double> timingsNoTransferVector;
		std::vector<double> timingsIncludeTransferVector;
		std::vector<double> timingsSmoothing;
		std::vector<double> timingsTotalBp;
		DetailedTimings* detailedTimingsOverall = nullptr;

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

			//start timer for time to smooth image
			auto timeSmoothStart = std::chrono::system_clock::now();

			//first smooth the images using the Gaussian filter with the given SIGMA_BP value
			//smoothed images are stored on the target device at locations smoothedImage1 and smoothedImage2
			(*smoothImage)(image1AsUnsignedIntArrayHost,
						widthImages, heightImages, algSettings.smoothingSigma, smoothedImage1);
			(*smoothImage)(image2AsUnsignedIntArrayHost,
						widthImages, heightImages, algSettings.smoothingSigma, smoothedImage2);

			//end timer for image smoothing and add to image smoothing timings
			auto timeSmoothEnd = std::chrono::system_clock::now();
			std::chrono::duration<double> diffTimeSmoothing = timeSmoothEnd-timeSmoothStart;
			timingsSmoothing.push_back(diffTimeSmoothing.count());

			//free the host memory allocated to original image 1 and image 2 on the host
			delete[] image1AsUnsignedIntArrayHost;
			delete[] image2AsUnsignedIntArrayHost;

			//set the width and height parameters of the imag
			algSettings.widthImages = widthImages;
			algSettings.heightImages = heightImages;

			float* disparityMapFromImage1To2CompDevice;

			//allocate the space for the disparity map estimation
			runBPMemoryMangement->allocateDataOnCompDevice((void**)&disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

			//start bp processing time timer
			auto bpTotalTimeStart = std::chrono::system_clock::now();

			//run belief propagation on device as specified by input pointer to ProcessBPOnTargetDevice object runBpStereo
			DetailedTimings* currentDetailedTimings = (*runBpStereo)(smoothedImage1, smoothedImage2,
					disparityMapFromImage1To2CompDevice, algSettings);

			//stop the bp processing time and add vector of bp processing times
			auto bpTotalTimeEnd = std::chrono::system_clock::now();
			std::chrono::duration<double> diffBpTotalTime = bpTotalTimeEnd-bpTotalTimeStart;
			timingsTotalBp.push_back(diffBpTotalTime.count());

			//retrieve the running time of the implementation not including the host/device transfer time
			auto timeNoTransferEnd = std::chrono::system_clock::now();
			std::chrono::duration<double> diff = timeNoTransferEnd-timeNoTransferStart;
			timingsNoTransferVector.push_back(diff.count());

			//allocate the space on the host for and x and y movement between images
			float* disparityMapFromImage1To2Host = new float[widthImages
						* heightImages];

			//transfer the disparity map estimation on the device to the host for output
			runBPMemoryMangement->transferDataFromCompDeviceToHost(disparityMapFromImage1To2Host, disparityMapFromImage1To2CompDevice, widthImages * heightImages * sizeof(float));

			auto timeWithTransferEnd = std::chrono::system_clock::now();

			std::chrono::duration<double> diffWithTransferTime = timeWithTransferEnd
					- timeWithTransferStart;
			timingsIncludeTransferVector.push_back(diffWithTransferTime.count());

			//save the resulting disparity map images to a file
			ImageHelperFunctions::saveDisparityImageToPGM(saveDisparityMapImagePath,
						SCALE_BP, disparityMapFromImage1To2Host,
						widthImages, heightImages);

			delete[] disparityMapFromImage1To2Host;

			//free the space allocated to the resulting disparity map and smoothed images on the computation device
			runBPMemoryMangement->freeDataOnCompDevice((void**)&disparityMapFromImage1To2CompDevice);
			runBPMemoryMangement->freeDataOnCompDevice((void**)&smoothedImage1);
			runBPMemoryMangement->freeDataOnCompDevice((void**)&smoothedImage2);

			//check if detailed timings are returned; if so, add to overall detailed timings
			if (currentDetailedTimings != nullptr)
			{
				if (detailedTimingsOverall == nullptr)
				{
					detailedTimingsOverall = currentDetailedTimings;
				}
				else
				{
					//add timings for run to overall set
					detailedTimingsOverall->addTimings(currentDetailedTimings);
				}
			}
		}

		fprintf(resultsFile, "Image Width: %d\n", widthImages);
		fprintf(resultsFile, "Image Height: %d\n", heightImages);

		std::sort(timingsNoTransferVector.begin(), timingsNoTransferVector.end());
		std::sort(timingsIncludeTransferVector.begin(), timingsIncludeTransferVector.end());
		std::sort(timingsSmoothing.begin(), timingsSmoothing.end());
		std::sort(timingsTotalBp.begin(), timingsTotalBp.end());

		fprintf(resultsFile, "Median Smoothing Runtime: %f\n", timingsSmoothing.at(NUM_BP_STEREO_RUNS/2));
		fprintf(resultsFile, "Median Total BP Runtime: %f\n", timingsTotalBp.at(NUM_BP_STEREO_RUNS/2));
		fprintf(resultsFile, "MEDIAN RUNTIME (NOT INCLUDING TRANSFER TIME OF DATA TO/FROM comp device MEMORY): %f\n", timingsNoTransferVector.at(NUM_BP_STEREO_RUNS/2));
		fprintf(resultsFile, "MEDIAN RUNTIME (INCLUDING TRANSFER TIME OF DATA TO/FROM comp device MEMORY): %f\n", timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2));

		if (detailedTimingsOverall != nullptr)
		{
			//uncomment to print timings for each part of implementation
			//timings.PrintMedianTimings();
			detailedTimingsOverall->PrintMedianTimingsToFile(resultsFile);
		}

		return timingsIncludeTransferVector.at(NUM_BP_STEREO_RUNS/2);
	}
};

#endif /* RUNBPSTEREOSET_H_ */
