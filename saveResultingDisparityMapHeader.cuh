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

//Declares the functions used to store the resulting disparity map from the BP implementation
#ifndef SAVE_RESULTING_DISPARITY_MAP_HEADER_CUH
#define SAVE_RESULTING_DISPARITY_MAP_HEADER_CUH

#include "bpStereoCudaParameters.cuh"

//include in order to save the resulting disparity map as a PGM
#include "imageHelpersHostHeader.cuh"
#include <chrono>

//save the output disparity map using the scale defined in scaleDisparityInOutput at each pixel to the file at disparityMapSaveImagePath
void saveResultingDisparityMap(const char* disparityMapSaveImagePath,
		float*& disparityMapFromImage1To2Device, float scaleDisparityInOutput,
		unsigned int widthImages, unsigned int heightImages,
		std::chrono::time_point<std::chrono::system_clock>& timeWithTransferStart,
		double& totalTimeIncludeTransfer);

#endif //SAVE_RESULTING_Stereo_HEADER_CUH
