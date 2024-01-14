/*
 * SmoothImageCUDA.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include "../ParameterFiles/bpStereoCudaParameters.h"
#include "../ParameterFiles/bpStructsAndEnums.h"

//include for the kernel functions to be run on the GPU
#include "kernelFilterHeader.cuh"

#include <cuda_runtime.h>
#include "../BpAndSmoothProcessing/SmoothImage.h"

class SmoothImageCUDA : public SmoothImage {
public:
  SmoothImageCUDA(const beliefprop::ParallelParameters& cudaParams) : cudaParams_(cudaParams) {}

  //for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
  void operator()(const BpImage<unsigned int>& inImage, const float sigmaVal, float* smoothedImage) override;

private:
  const beliefprop::ParallelParameters& cudaParams_;
};

#endif /* SMOOTHIMAGECUDA_H_ */
