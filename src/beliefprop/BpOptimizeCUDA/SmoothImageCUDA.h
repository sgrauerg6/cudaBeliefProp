/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file SmoothImageCUDA.h
 * @author Scott Grauer-Gray
 * @brief Declares child class of SmoothImage for smoothing images in the
 * CUDA implementation
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include <cuda_runtime.h>
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpImageProcessing/BpImageConstraints.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunProcessing/ParallelParamsBp.h"

//include for the kernel functions to be run on the GPU
#include "KernelFilterCUDA.cuh"

/**
 * @brief Child class of SmoothImage for smoothing images in the CUDA implementation
 * 
 */
class SmoothImageCUDA final : public SmoothImage {
public:
  explicit SmoothImageCUDA(const ParallelParams& cuda_params) : SmoothImage(cuda_params) {}

  /**
   * @brief For the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
   * 
   * @param in_image 
   * @param sigma 
   * @param smoothed_image 
   */
  void operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image) const override;
};

#endif /* SMOOTHIMAGECUDA_H_ */
