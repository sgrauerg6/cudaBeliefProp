/*
 * SmoothImageCUDA.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include "../ParameterFiles/bpStereoCudaParameters.h"

//include for the kernal functions to be run on the GPU
#include "kernalFilterHeader.cuh"

#include <cuda_runtime.h>
#include "../BpAndSmoothProcessing/SmoothImage.h"

template <typename T=float*>
class SmoothImageCUDA : public SmoothImage<T> {
public:
	SmoothImageCUDA() {}
	virtual ~SmoothImageCUDA() {}

	//for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
	void operator()(const BpImage<unsigned int>& inImage, float sigmaVal, T smoothedImage) override;
};

#endif /* SMOOTHIMAGECUDA_H_ */
