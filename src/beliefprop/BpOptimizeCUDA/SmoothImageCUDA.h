/*
 * SmoothImageCUDA.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include <cuda_runtime.h>
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpImageProcessing/BpImageConstraints.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunProcessing/BpParallelParams.h"

//include for the kernel functions to be run on the GPU
#include "kernelFilterHeader.cuh"

/**
 * @brief Class for smoothing images in the CUDA implementation
 * 
 */
class SmoothImageCUDA final : public SmoothImage {
public:
  SmoothImageCUDA(const ParallelParams& cuda_params) : SmoothImage(cuda_params) {}

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
