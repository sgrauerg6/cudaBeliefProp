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
#include "BpImageProcessing/SmoothImage.h"
#include "BpRunImp/BpParallelParams.h"

//include for the kernel functions to be run on the GPU
#include "kernelFilterHeader.cuh"

class SmoothImageCUDA final : public SmoothImage {
public:
  SmoothImageCUDA(const ParallelParams& cudaParams) : SmoothImage(cudaParams) {}

  //for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
  void operator()(const BpImage<unsigned int>& inImage, float sigmaVal, float* smoothedImage) override;
};

#endif /* SMOOTHIMAGECUDA_H_ */
