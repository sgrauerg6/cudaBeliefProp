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

//Defines the functions to store the resulting disparity map

#include "saveResultingDisparityMapHeader.cuh"

//save the output disparity map using the scale defined in scaleDisparityInOutput at each pixel to the file at disparityMapSaveImagePath
//also takes in the timer to time the implementation including the transfer time from the device to the host
void saveResultingDisparityMap(const char* disparityMapSaveImagePath,
		float*& disparityMapFromImage1To2Device, float scaleDisparityInOutput,
		unsigned int widthImages, unsigned int heightImages,
		std::chrono::time_point<std::chrono::system_clock>& timeWithTransferStart,
		double& totalTimeIncludeTransfer) {
	//allocate the space on the host for and x and y movement between images
	float* disparityMapFromImage1To2Host = new float[widthImages * heightImages];

	//transfer the disparity map estimation on the device to the host for output
	(cudaMemcpy(disparityMapFromImage1To2Host, disparityMapFromImage1To2Device, widthImages*heightImages*sizeof(float),
						  cudaMemcpyDeviceToHost) );

	auto timeWithTransferEnd = std::chrono::system_clock::now();

	//printf("Running time including transfer time: %.10lf seconds\n", timeEnd-timeStart);
	std::chrono::duration<double> diff = timeWithTransferEnd-timeWithTransferStart;
	totalTimeIncludeTransfer += diff.count();
	//stop the timer and print the total time of the BP implementation including the device-host transfer time
	//printf("Time to retrieve movement on host (including transfer): %f (ms) \n", totalTimeIncludeTransfer);

	//save the resulting disparity map images to a file
	saveDisparityImageToPGM(disparityMapSaveImagePath, scaleDisparityInOutput, disparityMapFromImage1To2Host, widthImages, heightImages);

	delete [] disparityMapFromImage1To2Host;
}

