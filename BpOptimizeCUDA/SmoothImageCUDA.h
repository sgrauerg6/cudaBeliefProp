/*
 * SmoothImageCUDA.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include <cuda_runtime.h>

#include "../BpConstsAndParams/bpStereoCudaParameters.h"
#include "../BpConstsAndParams/bpStructsAndEnums.h"
#include "../BpImageProcessing/SmoothImage.h"

//include for the kernel functions to be run on the GPU
#include "kernelFilterHeader.cuh"

class SmoothImageCUDA : public SmoothImage {
public:
  SmoothImageCUDA(const beliefprop::ParallelParameters& cudaParams) : cudaParams_(cudaParams) {}

  //for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
  void operator()(const BpImage<unsigned int>& inImage, const float sigmaVal, float* smoothedImage) override;

private:
  const beliefprop::ParallelParameters& cudaParams_;
};

#endif /* SMOOTHIMAGECUDA_H_ */
