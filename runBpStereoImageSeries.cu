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

#include "runBpStereoImageSeriesHeader.cuh"

#include "smoothImageHostHeader.cuh"

#include "saveResultingDisparityMapHeader.cuh"
#include <sys/time.h>

//run the disparity map estimation BP on a series of stereo images and save the results between each set of images if desired
void runStereoEstOnImageSeries(const char* imageFiles[], int numImages,
		unsigned int& widthImages, unsigned int& heightImages,
		BPsettings algSettings, bool saveResults,
		const char* saveDisparityMapImagePaths[]) {

	double totalTimeNoTransfer = 0.0;
	double totalTimeIncludeTransfer = 0.0;
	for (int numRun = 0; numRun < NUM_BP_STEREO_RUNS; numRun++) {
		//printf("RUN %d\n", numRun);
		//first run Stereo estimation on the first two images
		unsigned int* image1AsUnsignedIntArrayHost = loadImageFromPGM(
				imageFiles[0], widthImages, heightImages);
		unsigned int* image2AsUnsignedIntArrayHost = loadImageFromPGM(
				imageFiles[1], widthImages, heightImages);

		float* smoothedImage1Device;
		float* smoothedImage2Device;

		unsigned int timerTransferTimeNotIncluded;
		unsigned int timerIncludeTransferTime;

		//create timers to time the implementation with and without the transfer time between the host and device
		struct timeval timeWithTransferStart;
		struct timeval timeNoTransferStart;
		struct timeval timeNoTransferEnd;

		//allocate the device memory to store and x and y smoothed images
		(cudaMalloc((void**) &smoothedImage1Device,
				widthImages * heightImages * sizeof(float)));
		(cudaMalloc((void**) &smoothedImage2Device,
				widthImages * heightImages * sizeof(float)));

		//declare the allocate the space for image 1 and image 2 in the device
		unsigned int* image1Device;
		unsigned int* image2Device;

		cudaMalloc((void **) &image1Device,
				widthImages * heightImages * sizeof(unsigned int));
		cudaMalloc((void **) &image2Device,
				widthImages * heightImages * sizeof(unsigned int));

		//start timer to retrieve the time of implementation including transfer time
		gettimeofday(&timeWithTransferStart, NULL);

		//transfer the image 1 and image 2 data from the host to the device
		cudaMemcpy(image1Device, image1AsUnsignedIntArrayHost,
				widthImages * heightImages * sizeof(unsigned int),
				cudaMemcpyHostToDevice);
		cudaMemcpy(image2Device, image2AsUnsignedIntArrayHost,
				widthImages * heightImages * sizeof(unsigned int),
				cudaMemcpyHostToDevice);

		//free the host memory allocatted to original image 1 and image 2 on the host
		delete[] image1AsUnsignedIntArrayHost;
		delete[] image2AsUnsignedIntArrayHost;

		//start timer to retrieve the time of implementation not including transfer time
		gettimeofday(&timeNoTransferStart, NULL);

		//first smooth the images using the CUDA Gaussian filter with the given SIGMA_BP value
		//smoothed images are stored global memory on the device at locations image1SmoothedDevice and image2SmoothedDevice
		smoothImagesAllDataInDevice(image1Device, image2Device, widthImages,
				heightImages, SIGMA_BP, smoothedImage1Device,
				smoothedImage2Device);

		//free the space used to store the original image 1 in the device (space for image 2 used for future images in the sequence)
		cudaFree(image1Device);

		//set the width and height parameters of the imag
		algSettings.widthImages = widthImages;
		algSettings.heightImages = heightImages;

		float* disparityMapFromImage1To2Device;

		//allocate the space for the disparity map estimation
		cudaMalloc((void **) &disparityMapFromImage1To2Device,
				widthImages * heightImages * sizeof(float));

		//run BP on the image if image is smaller than "chunk size"
		if ((widthImages <= WIDTH_IMAGE_CHUNK_RUN_STEREO_EST_BP)
				&& (heightImages <= HEIGHT_IMAGE_CHUNK_RUN_STEREO_EST_BP)) {
			runBeliefPropStereoCUDA(smoothedImage1Device, smoothedImage2Device,
					disparityMapFromImage1To2Device, algSettings);
		}
		//otherwise run the BP Stereo on the image set "in chunks"
		else {
			printf("RUN IMAGE IN CHUNKS\n");
			runBPStereoEstOnImageSetInChunks(smoothedImage1Device,
					smoothedImage2Device, widthImages, heightImages,
					disparityMapFromImage1To2Device, algSettings);
		}

		//retrieve the running time of the implementation not including the host/device transfer time
		//cutStopTimer(timerTransferTimeNotIncluded);
		//printf("Running time not including transfer time: %f (ms) \n", cutGetTimerValue(timerTransferTimeNotIncluded));
		//cutResetTimer(timerTransferTimeNotIncluded);
		gettimeofday(&timeNoTransferEnd, NULL);
		double timeStart = timeNoTransferStart.tv_sec
				+ (timeNoTransferStart.tv_usec / 1000000.0);
		double timeEnd = timeNoTransferEnd.tv_sec
				+ (timeNoTransferEnd.tv_usec / 1000000.0);
		//printf("Running time not including transfer time: %.10lf seconds\n",
		//		timeEnd - timeStart);
		totalTimeNoTransfer += (timeEnd - timeStart);

		if (saveResults) {
			saveResultingDisparityMap(saveDisparityMapImagePaths[0],
					disparityMapFromImage1To2Device, SCALE_BP, widthImages,
					heightImages, timeWithTransferStart, totalTimeIncludeTransfer);
		}

		//now go through the rest of the images, the "second" image of one set as the "first one" of the next and also using
		//the previous Stereo values
		for (int numImage = 2; numImage < numImages; numImage++) {
			//use the previous "second image" as the next "first image"
			cudaMemcpy(smoothedImage1Device, smoothedImage2Device,
					widthImages * heightImages * sizeof(float),
					cudaMemcpyDeviceToDevice);

			//load the next image from memory...this is will be image 2
			image2AsUnsignedIntArrayHost = loadImageFromPGM(
					imageFiles[numImage], widthImages, heightImages);

			//transfer the image from the host to the device
			cudaMemcpy(image2Device, image2AsUnsignedIntArrayHost,
					widthImages * heightImages * sizeof(float),
					cudaMemcpyHostToDevice);

			//smooth the image and smoothed image now in device
			smoothSingleImageAllDataInDevice(image2Device, widthImages,
					heightImages, SIGMA_BP, smoothedImage2Device);

			//now free the host memory allocatted to original image
			delete[] image2AsUnsignedIntArrayHost;

			//run BP on the image if image is smaller than "chunk size"
			if ((widthImages <= WIDTH_IMAGE_CHUNK_RUN_STEREO_EST_BP)
					&& (heightImages <= HEIGHT_IMAGE_CHUNK_RUN_STEREO_EST_BP)) {
				runBeliefPropStereoCUDA(smoothedImage1Device,
						smoothedImage2Device, disparityMapFromImage1To2Device,
						algSettings);
			}
			//otherwise run the BP Stereo on the image set "in chunks" (although for a small image, there may only be one chunk)
			else {
				runBPStereoEstOnImageSetInChunks(smoothedImage1Device,
						smoothedImage2Device, widthImages, heightImages,
						disparityMapFromImage1To2Device, algSettings);
			}

			//save results if desired
			if (saveResults) {
				//TODO: fix measuring runtime in cases w/ more than two images
				saveResultingDisparityMap(
						saveDisparityMapImagePaths[numImage - 1],
						disparityMapFromImage1To2Device, SCALE_BP, widthImages,
						heightImages, timeWithTransferStart, totalTimeIncludeTransfer);
			}
		}

		//free the space allocated to the resulting disparity map
		cudaFree(disparityMapFromImage1To2Device);
		//free the space allocated to the original image 2 in device
		cudaFree(image2Device);
		//free the space allocated to the smoothed images on the device
		cudaFree(smoothedImage1Device);
		cudaFree(smoothedImage2Device);
		//printf("\n");
	}

	//printf("Total time: %f\n", totalTime);
	float averageGpuRunTimeNoTransfer = totalTimeNoTransfer / NUM_BP_STEREO_RUNS;
	float averageGpuRunTimeIncludeTransfer = totalTimeIncludeTransfer / NUM_BP_STEREO_RUNS;
	printf("AVERAGE GPU RUN TIME (NOT INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", averageGpuRunTimeNoTransfer);
	printf("AVERAGE GPU RUN TIME (INCLUDING TRANSFER TIME OF DATA TO/FROM GPU MEMORY): %f\n", averageGpuRunTimeIncludeTransfer);
}

