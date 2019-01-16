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

//Defines the functions used to run CUDA BP Stereo estimation on an image set where the BP is run on the set in "chunks" rather than all at once (likely
//because there is limited storage on the device)

#include "runBpStereoDivideImageHeader.cuh"

//run the BP Stereo estimation algorithm in "chunks" on a set of images of given width and height stored on the device (already been loaded and "smoothed" if desired)
//assume that the device memory is already allocated for output disparity map in disparityMapImageSetDevice
void runBPStereoEstOnImageSetInChunks(float* image1Device, float* image2Device, unsigned int widthImages, unsigned int heightImages, float*& disparityMapImageSetDevice, BPsettings& runBPAlgSettings, DetailedTimings& timings, unsigned int widthImageChunk, unsigned int heightImageChunk, unsigned int imageChunkPaddingX, unsigned int imageChunkPaddingY)
{
	float* currentPortionImage1Device;
	float* currentPortionImage2Device;

	float* disparityMapPortionDevice;


	for (int yStartPortionInFullImage = 0; yStartPortionInFullImage < (int)heightImages; yStartPortionInFullImage += (int)heightImageChunk)
	{
		for (int xStartPortionInFullImage = 0; xStartPortionInFullImage < (int)widthImages; xStartPortionInFullImage += (int)widthImageChunk)
		{
			int currentLeftSidePadding = min((int)imageChunkPaddingX, (int)xStartPortionInFullImage);
			int currentTopPadding = min((int)imageChunkPaddingY, (int)yStartPortionInFullImage);

			//these values can be negative when at the "end" of the image...so can't use unsigned int
			int currentRightSidePadding = min((int)imageChunkPaddingX, (int)widthImages - ((int)xStartPortionInFullImage + (int)widthImageChunk));
			int currentBottomPadding = min((int)imageChunkPaddingY, (int)heightImages - ((int)yStartPortionInFullImage + (int)heightImageChunk));

			int startXPortionWithPadding = xStartPortionInFullImage - currentLeftSidePadding;
			int startYPortionWithPadding = yStartPortionInFullImage - currentTopPadding;

			int endXPortionWithPadding = xStartPortionInFullImage + widthImageChunk + currentRightSidePadding;
			int endYPortionWithPadding = yStartPortionInFullImage + heightImageChunk + currentBottomPadding;

			int widthCurrentPortion = endXPortionWithPadding - startXPortionWithPadding;
			int heightCurrentPortion = endYPortionWithPadding - startYPortionWithPadding;

			//allocate the space for the current portion of the image set and the retrieved disparity map...need to do this separately each time since portions could be different sizes
			cudaMalloc((void**) &disparityMapPortionDevice, widthCurrentPortion*heightCurrentPortion*sizeof(float));

			cudaMalloc((void**) &currentPortionImage1Device, widthCurrentPortion*heightCurrentPortion*sizeof(float));
			cudaMalloc((void**) &currentPortionImage2Device, widthCurrentPortion*heightCurrentPortion*sizeof(float));


			extractSubMatrixInDeviceFrom2dArrayInDevice(currentPortionImage1Device, image1Device, startXPortionWithPadding, startYPortionWithPadding, widthCurrentPortion, heightCurrentPortion, widthImages, heightImages);
			extractSubMatrixInDeviceFrom2dArrayInDevice(currentPortionImage2Device, image2Device, startXPortionWithPadding, startYPortionWithPadding, widthCurrentPortion, heightCurrentPortion, widthImages, heightImages);


			runBPAlgSettings.widthImages = widthCurrentPortion; 
			runBPAlgSettings.heightImages = heightCurrentPortion;


			//run BP on the image portion
			runBeliefPropStereoCUDA(currentPortionImage1Device, currentPortionImage2Device, disparityMapPortionDevice, runBPAlgSettings, timings);


			//retrieve the size of the chunk to write to the device (may be smaller than chunk size if at "end")...write the portion of the Stereo without the padding
			//padding is not "negative" here like it may be when taking from the full image
			int imageChunkWidthToWrite = widthCurrentPortion - (max(0, currentLeftSidePadding) + max(0, currentRightSidePadding)); 
			int imageChunkHeightToWrite = heightCurrentPortion - (max(0, currentTopPadding) + max(0, currentBottomPadding)); 


			//now write the calculated disparity map to the output disparity map estimation
			writeSubMatrixInDeviceTo2dArrayInDevice(disparityMapPortionDevice, disparityMapImageSetDevice, currentLeftSidePadding, currentTopPadding, xStartPortionInFullImage, yStartPortionInFullImage, imageChunkWidthToWrite, imageChunkHeightToWrite, widthCurrentPortion, heightCurrentPortion, widthImages, heightImages);

			//now free the device memory related the the current portion...
			cudaFree(currentPortionImage1Device);
			cudaFree(currentPortionImage2Device);

			cudaFree(disparityMapPortionDevice);

		}
	}
}





//extract a submatrix widthMatrixRetrieve X heightMatrixRetrieve of size in device memory from a 2d array in the device given the starting point (startXIn2DArray, startYIn2DArray)
//the output outputSubMatrixDevice is a pointer to the desired submatrix in the device 
//the indices and width/height are defined in terms of number of floats "into" the array, not the number of bytes
//it's assumed that the needed device memory for outputSubMatrixDevice has already been allocated using cudaMalloc (rather than cudaMalloc2D)
__host__ void extractSubMatrixInDeviceFrom2dArrayInDevice(float*& outputFloatSubMatrixDevice, float*& twoDFloatArrayInDevice, unsigned int startXIn2DArray, unsigned int startYIn2DArray, unsigned int widthOutputMatrix, unsigned int heightOutputMatrix, unsigned int width2DArrayDataFrom, unsigned int height2DArrayDataFrom)
{
	//for now, go through each "row" in the outputSubMatrix in the device memory and copy the desired data in 2dArrayInDevice to it

	unsigned int startingIndexCopyFrom2DFloatArray = width2DArrayDataFrom*startYIn2DArray + startXIn2DArray;

	for (unsigned int numRowToCopy = 0; numRowToCopy < heightOutputMatrix; numRowToCopy++)
	{
		//copy the current row from the 2D float array in the device to the sub-matrix also in the device
		//total number of bytes copied for each row is widthOutputMatrix*sizeof(float)
		cudaMemcpy((void *)&outputFloatSubMatrixDevice[numRowToCopy*widthOutputMatrix], (void *)&twoDFloatArrayInDevice[startingIndexCopyFrom2DFloatArray + numRowToCopy*width2DArrayDataFrom], widthOutputMatrix*sizeof(float), cudaMemcpyDeviceToDevice);
	}
}


//write a submatrix widthInputMatrixWrite X heightInputMatrixWrite of size in device memory to a 2d array in the device given the starting point (startXIn2DArrayWrite, startYIn2DArrayWrite)
//of the array to write to and the starting point (startXInSubMatrixRead, startYInSubMatrixRead) of the array to read from
//the input inputSubMatrixDevice is a pointer to the submatrix in the device to write to the device memory 2dArrayInDevice at the desired location
//the indices and width/height are defined in terms of number of floats "into" the array, not the number of bytes
__host__ void writeSubMatrixInDeviceTo2dArrayInDevice(float*& inputFloatSubMatrixDevice, float*& twoDFloatArrayInDevice, unsigned int startXInSubMatrixRead, unsigned int startYInSubMatrixRead, unsigned int startXIn2DArrayWrite, unsigned int startYIn2DArrayWrite, unsigned int widthInputMatrixWrite, unsigned int heightInputMatrixWrite, unsigned int totalWidthInputMatrix, unsigned int totalHeightInputMatrix, unsigned int width2DArrayDataTo, unsigned int height2DArrayDataTo)
{
	unsigned int startingIndexCopyTo2DFloatArray = width2DArrayDataTo*startYIn2DArrayWrite + startXIn2DArrayWrite;

	for (unsigned int numRowToCopy = 0; numRowToCopy < heightInputMatrixWrite; numRowToCopy++)
	{
		//copy the current row from the input float sub-matrix on the device to the larger 2D-float array
		//total number of bytes copied for each row is widthInputMatrix*sizeof(float)
		cudaMemcpy((void *)&twoDFloatArrayInDevice[numRowToCopy*width2DArrayDataTo + startingIndexCopyTo2DFloatArray], (void *)&inputFloatSubMatrixDevice[(numRowToCopy + startYInSubMatrixRead)*totalWidthInputMatrix + startXInSubMatrixRead], widthInputMatrixWrite*sizeof(float), cudaMemcpyDeviceToDevice);
	}
}
