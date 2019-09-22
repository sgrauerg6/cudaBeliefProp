/*
 * DisparityMap.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "DisparityMap.h"

template<class T>
void DisparityMap<T>::saveDisparityMap(
		const std::string& disparity_map_file_path,
		const unsigned int scale_factor) const {
	//declare and allocate the space for the movement image to save
	BpImage<char> movementImageToSave(this->width_, this->height_);

	//go though every value in the movementBetweenImages data and retrieve the intensity value to use in the resulting "movement image" where minMovementDirection
	//represents 0 intensity and the intensity increases linearly using scaleMovement from minMovementDirection
	std::transform(this->getPointerToPixelsStart(),
			this->getPointerToPixelsStart() + (this->width_ * this->height_),
			movementImageToSave.getPointerToPixelsStart(),
			[this, scale_factor](const T& currentPixel) -> char {
				return (char)(((float)currentPixel)*(float)scale_factor + .5f);
			});

	printf("%s\n", disparity_map_file_path.c_str());

	movementImageToSave.saveImageAsPgm(disparity_map_file_path);
}

template class DisparityMap<float>;
