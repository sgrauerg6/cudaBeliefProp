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

//Declares the functions used to smooth the input images with a gaussian filter of SIGMA_BP (see bpCudaParameters.cuh) implemented in CUDA

#ifndef SMOOTH_IMAGE_HOST_HEADER_CUH
#define SMOOTH_IMAGE_HOST_HEADER_CUH

#include <cmath>
#include <memory>
#include <utility>
#include <algorithm>
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpRunImp/BpParallelParams.h"
#include "BpImage.h"

//don't smooth input images if SIGMA_BP below this
constexpr float MIN_SIGMA_VAL_SMOOTH{0.1f};

//parameter for smoothing
constexpr float WIDTH_SIGMA_1{4.0f};

//functions relating to smoothing the images before running BP
//Smoothing image always uses float data type
class SmoothImage
{
public:
  SmoothImage(const ParallelParams& parallelParams) : parallelParams_{parallelParams} {}

  //function to use the image filter to apply a guassian filter to the a single images
  //input images have each pixel stored as an unsigned int (value between 0 and 255 assuming 8-bit grayscale image used)
  //output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
  //normalize mask so it integrates to one
  virtual void operator()(const BpImage<unsigned int>& inImage, const float sigmaVal, float* smoothedImage) = 0;

protected:
  //create a Gaussian filter from a sigma value
  std::pair<std::unique_ptr<float[]>, unsigned int> makeFilter(const float sigma);

  //parallel parameters to use parallel operations (number of threads on CPU / thread block config in CUDA)
  const ParallelParams& parallelParams_;

private:
  //normalize filter mask so it integrates to one
  void normalizeFilter(const std::unique_ptr<float[]>& filter, const unsigned int sizeFilter);
};

#endif //SMOOTH_IMAGE_HOST_HEADER_CUH
