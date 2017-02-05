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

//Declares the functions used to load the input images and store the disparity/movement image for use in the CUDA BP implementation

#ifndef IMAGE_HELPERS_HOST_HEADER_CUH
#define IMAGE_HELPERS_HOST_HEADER_CUH

#define MAXROWS 2000
#define MAXCOLS 2000
#define MAXLENGTH 256
#define MAXVALUE 255

#include "bpStereoCudaParameters.cuh"

//functions used to load input images/save resulting movment images

//function to retrieve the disparity values from a disparity map with a known scale factor
__host__ float* retrieveDisparityValsFromStereoPGM(const char* filePathPgmImage, unsigned int widthImage, unsigned int heightImage, float scaleFactor);

//load the PGM image and return as an array of unsigned int (between values 0 and 255 assuming image is 8-bit grayscale)
__host__ unsigned int* loadImageFromPGM(const char* filePathPgmImage, unsigned int& widthImage, unsigned int& heightImage);

//save the calculated disparity map from image 1 to image 2 as a grayscale image using the SCALE_MOVEMENT factor with
//0 representing "zero" intensity and the intensity linearly increasing from there using SCALE_MOVEMENT
__host__ void saveDisparityImageToPGM(const char* filePathSaveImage, float scaleMovement, float*& calcDisparityBetweenImages, unsigned int widthImage, unsigned int heightImage);

int pgmWrite(const char* filename, unsigned int cols, unsigned int rows,
	     unsigned char* image,char* comment_string);

int pgmRead (const char *fileName, unsigned int *cols, unsigned int *rows,
	     unsigned char*& image);



#endif //IMAGE_HELPERS_HOST_HEADER_CUH
