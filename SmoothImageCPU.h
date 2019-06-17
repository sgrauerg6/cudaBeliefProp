/*
 * SmoothImageCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECPU_H_
#define SMOOTHIMAGECPU_H_

#include "SmoothImage.h"

class SmoothImageCPU : public SmoothImage {
public:
	SmoothImageCPU();
	virtual ~SmoothImageCPU();

	void operator()(unsigned int* inImage, int widthImages,
			int heightImages, float sigmaVal, float* imageSmoothed);
};

#endif /* SMOOTHIMAGECPU_H_ */
