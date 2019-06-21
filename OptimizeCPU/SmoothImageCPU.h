/*
 * SmoothImageCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECPU_H_
#define SMOOTHIMAGECPU_H_

#include "SmoothImage.h"
#include <algorithm>

class SmoothImageCPU : public SmoothImage {
public:
	SmoothImageCPU();
	virtual ~SmoothImageCPU();

	void operator()(unsigned int* inImage, unsigned int widthImages, unsigned int heightImages, float sigmaVal, float* smoothedImage);
};

#endif /* SMOOTHIMAGECPU_H_ */
