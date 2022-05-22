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
#include "../ImageDataAndProcessing/BpImage.h"
#include <algorithm>

constexpr float MIN_SIGMA_VAL_SMOOTH{0.1f}; //don't smooth input images if SIGMA_BP below this

//more parameters for smoothing
constexpr float WIDTH_SIGMA_1{4.0f};
//constexpr unsigned int MAX_SIZE_FILTER{25u};

//functions relating to smoothing the images before running BP
template <typename T=float*>
class SmoothImage
{
public:
	//normalize filter mask so it integrates to one
	void normalizeFilter(const std::unique_ptr<float[]>& filter, const unsigned int sizeFilter)
	{
		float sum{0.0f};
		for (unsigned int i = 1; i < sizeFilter; i++) {
			sum += std::abs(filter[i]);
		}
		sum = 2*sum + std::abs(filter[0]);
		for (unsigned int i = 0; i < sizeFilter; i++) {
			filter[i] /= sum;
		}
	}

	//this function creates a Gaussian filter given a sigma value
	std::pair<std::unique_ptr<float[]>, unsigned int> makeFilter(const float sigma)
	{
		const float sigmaUse{std::max(sigma, 0.01f)};
		const unsigned int sizeFilter{(unsigned int)std::ceil(sigmaUse * WIDTH_SIGMA_1) + 1u};
		std::unique_ptr<float[]> mask = std::make_unique<float[]>(sizeFilter);
		for (unsigned int i = 0; i < sizeFilter; i++) {
			mask[i] = std::exp(-0.5*((i/sigmaUse) * (i/sigmaUse)));
		}
		normalizeFilter(mask, sizeFilter);

		return {std::move(mask), sizeFilter};
	}

	//function to use the image filter to apply a guassian filter to the a single images
	//input images have each pixel stored as an unsigned int (value between 0 and 255 assuming 8-bit grayscale image used)
	//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
	//normalize mask so it integrates to one
	virtual void operator()(const BpImage<unsigned int>& inImage, const float sigmaVal, T smoothedImage) = 0;
};

#endif //SMOOTH_IMAGE_HOST_HEADER_CUH
