/*
 * SmoothImageCUDA.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECUDA_H_
#define SMOOTHIMAGECUDA_H_

#include "ParameterFiles/bpStereoCudaParameters.h"

//include for the kernal functions to be run on the GPU
#include "kernalFilterHeader.cuh"

#include <cuda_runtime.h>
#include "SmoothImage.h"

class SmoothImageCUDA : public SmoothImage {
public:
	SmoothImageCUDA();
	virtual ~SmoothImageCUDA();

	//for the CUDA smoothing, the input image is on the host and the output image is on the device (GPU)
	void operator()(unsigned int* inImage, unsigned int widthImages, unsigned int heightImages, float sigmaVal, float* smoothedDevice);
};

#endif /* SMOOTHIMAGECUDA_H_ */
