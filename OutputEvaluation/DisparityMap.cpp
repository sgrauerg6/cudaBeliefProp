/*
 * DisparityMap.cpp
 *
 *  Created on: Sep 13, 2019
 *      Author: scott
 */

#include "DisparityMap.h"

template<class T>
void DisparityMap<T>::saveDisparityMap(const std::string& disparity_map_file_path, const unsigned int scale_factor) const {
  //declare and allocate the space for the movement image to save
  BpImage<char> movementImageToSave(this->widthHeight_);

  //go though every value in the movementBetweenImages data and retrieve the intensity value to use in the resulting "movement image" where minMovementDirection
  //represents 0 intensity and the intensity increases linearly using scaleMovement from minMovementDirection
  std::transform(this->getPointerToPixelsStart(), this->getPointerToPixelsStart() + this->getTotalPixels(),
      movementImageToSave.getPointerToPixelsStart(),
      [this, scale_factor](const T& currentPixel) -> char {
        return (char)(((float)currentPixel)*((float)scale_factor) + 0.5f);
      });

  movementImageToSave.saveImageAsPgm(disparity_map_file_path);
}

template class DisparityMap<float>;
