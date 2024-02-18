/*
 * BpImage.cpp
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#include "BpImage.h"

template<class T>
void BpImage<T>::loadImageAsGrayScale(const std::string& filePathImage) {
  std::string filePathImageCopy(filePathImage);

  //check if PGM or PPM image (types currently supported)
  std::istringstream iss(filePathImageCopy);
  std::string token;
  while (std::getline(iss, token, '.')) {
    //continue to get last token with the file extension
  }

  //last token after "." is file extension
  //use extension to check if image is pgm or ppm
  BpImage<unsigned char> initImage;
  if (token == PGM_EXTENSION) {
    initImage = imageRead(filePathImage, image_type::PGM_IMAGE);
  } else if (token == PPM_EXTENSION) {
    initImage = imageRead(filePathImage, image_type::PPM_IMAGE, USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION);
  } else {
    //end function if not of type pgm or ppm with result
    //being BpImage with default constructor
    return;
  }

  //set width and height using data from file image and initialize
  //pixels array of size width_ * height_ with unique pointer
  widthHeight_ = {initImage.getWidth(), initImage.getHeight()};
  pixels_ = std::make_unique<T[]>(getTotalPixels());

  //convert each pixel in dataRead to data type T and place in imageData array in same location
  std::ranges::transform(initImage.getPointerToPixelsStart(), initImage.getPointerToPixelsStart() + getTotalPixels(),
    &(pixels_[0]), [] (const unsigned char i) -> T {return (T)i;});
}

template<class T>
void BpImage<T>::pnm_read(std::ifstream &file, std::string& buf) const {
  std::string doc;
  char c;

  file >> c;
  while (c == '#') {
    std::getline(file, doc);
    file >> c;
  }
  file.putback(c);

  file >> buf;
  file.ignore();
}

template<class T>
BpImage<unsigned char> BpImage<T>::imageRead(const std::string& fileName,
  const image_type imageType, const bool weightedRGBConversion) const
{
  std::string buf;

  /* read header */
  std::ifstream file(fileName, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (buf != "P5")
    std::cout << "ERROR READING FILE\n";

  unsigned int cols, rows;
  pnm_read(file, buf);
  cols = std::stoul(buf);
  pnm_read(file, buf);
  rows = std::stoul(buf);

  pnm_read(file, buf);
  if (std::stoul(buf) > UCHAR_MAX)
    std::cout << "ERROR READING FILE\n";

  BpImage<unsigned char> outImage(std::array<unsigned int, 2>{cols, rows});

  if (imageType == image_type::PGM_IMAGE) {
    /* read data */
    file.read((char*) (outImage.getPointerToPixelsStart()),
      (cols * rows * sizeof(char)));
  }
  else if (imageType == image_type::PPM_IMAGE) {
    std::unique_ptr<char[]> rgbImagePtr = std::make_unique<char[]>(3 * cols * rows);

    /* read data */
    file.read(&(rgbImagePtr[0]), 3 * cols * rows * sizeof(char));
    file.close();

    //convert the RGB image to grayscale either
    //with each channel weighted the same or weighted
    //if weightedRGBConversion set to true
    for (unsigned int i = 0; i < (rows * cols); i++) {
      float rChannelWeight = 1.0f / 3.0f;
      float bChannelWeight = 1.0f / 3.0f;
      float gChannelWeight = 1.0f / 3.0f;
      if (weightedRGBConversion) {
        rChannelWeight = 0.299f;
        bChannelWeight = 0.587f;
        gChannelWeight = 0.114f;
      }
      outImage.getPointerToPixelsStart()[i] = (unsigned char)std::floor(
        rChannelWeight * ((float) rgbImagePtr[i * 3]) +
        gChannelWeight * ((float) rgbImagePtr[i * 3 + 1]) +
        bChannelWeight * ((float) rgbImagePtr[i * 3 + 2]) +
        0.5f);
    }
  }

  file.close();
  return outImage;
}

//each type that may be used instantiated here
template class BpImage<unsigned int>;
template class BpImage<float>;
template class BpImage<unsigned char>;
template class BpImage<char>;

