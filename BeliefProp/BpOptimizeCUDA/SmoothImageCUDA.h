/*
 * SmoothImageCUDA.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include <cuda_runtime.h>
#include "BpConstsAndParams/BpStructsAndEnums.h"
#include "BpImageProcessing/BpSmoothTypeConstraints.h"
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunProcessing/BpParallelParams.h"

//include for the kernel functions to be run on the GPU
#include "kernelFilterHeader.cuh"

class SmoothImageCUDA final : public SmoothImage {
public:
  SmoothImageCUDA(const ParallelParams& cuda_params) : SmoothImage(cuda_params) {}

  //for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
  void operator()(const BpImage<unsigned int>& in_image, float sigma, float* smoothed_image) override;
};

#endif /* SMOOTHIMAGECUDA_H_ */
