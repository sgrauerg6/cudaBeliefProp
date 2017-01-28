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

//Defines the parameters to use when running stereo estimation BP on an image in "chunks" (likely because there isn't enough space on the device to run it all at once)

#ifndef RUN_BP_STEREO_DIVIDE_IMAGE_PARAMETERS_CUH
#define RUN_BP_STEREO_DIVIDE_IMAGE_PARAMETERS_CUH

#include "bpStereoCudaParameters.cuh"

//define the width and height of the image chunk to run BP on (initialized to INF_BP so images won't be divided...)
#define WIDTH_IMAGE_CHUNK_RUN_STEREO_EST_BP INF_BP
#define HEIGHT_IMAGE_CHUNK_RUN_STEREO_EST_BP INF_BP

//define the "padding" on the x and y axis (each direction) of the image chunk
#define PADDING_IMAGE_CHUNK_X_RUN_STEREO_EST_BP 0
#define PADDING_IMAGE_CHUNK_Y_RUN_STEREO_EST_BP 0

#endif //RUN_BP_STEREO_DIVIDE_IMAGE_PARAMETERS_CUH
