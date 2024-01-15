/*
 * BpImage.h
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#ifndef BPIMAGE_H_
#define BPIMAGE_H_

#include <memory>
#include <algorithm>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include <array>
#include <string_view>
#include <type_traits>

enum class image_type { PGM_IMAGE, PPM_IMAGE };
constexpr bool USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION = true;
constexpr std::string_view PGM_EXTENSION = "pgm";
constexpr std::string_view PPM_EXTENSION = "ppm";

template <class T>
requires std::is_arithmetic_v<T>
class BpImage {
public:
  BpImage() : widthHeight_{0, 0} {}

  BpImage(const std::array<unsigned int, 2>& widthHeight) : widthHeight_{widthHeight}, pixels_(std::make_unique<T[]>(widthHeight_[0]*widthHeight_[1])) {}

  BpImage(const std::array<unsigned int, 2>& widthHeight, const T* input_pixel_vals) :
    widthHeight_{widthHeight}, pixels_(std::make_unique<T[]>(widthHeight_[0] * widthHeight_[1]))
  {
    std::copy(input_pixel_vals, input_pixel_vals + (getTotalPixels()), pixels_.get());
  }

  BpImage(const std::string& fileName) {
    loadImageAsGrayScale(fileName);
  }

  const std::unique_ptr<T[]>& getUniquePtrToPixelData() const {
    return pixels_;
  }

  T* getPointerToPixelsStart() const {
    return &(pixels_[0]);
  }

  T getPixelAtPoint(const std::array<unsigned int, 2>& xyPoint) const {
    return getPixelAtPoint(xyPoint[1]*widthHeight_[0] + xyPoint[0]);
  }

  T getPixelAtPoint(const unsigned int i) const {
    return (pixels_.get())[i];
  }

  void setPixelAtPoint(const std::array<unsigned int, 2>& xyPoint, const T val) {
    setPixelAtPoint((xyPoint[1]*widthHeight_[0] + xyPoint[0]), val);
  }

  void setPixelAtPoint(const unsigned int i, const T val) {
    (pixels_.get())[i] = val;
  }

  unsigned int getWidth() const { return widthHeight_[0]; }
  unsigned int getHeight() const { return widthHeight_[1]; }

  void saveImageAsPgm(const std::string& filename) const {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file << "P5\n" << widthHeight_[0] << " " << widthHeight_[1] << "\n" << UCHAR_MAX << "\n";
    file.write((char*)(&pixels_[0]), getTotalPixels() * sizeof(char));
    file.close();
  }

protected:

  std::array<unsigned int, 2> widthHeight_;
  std::unique_ptr<T[]> pixels_;

  void loadImageAsGrayScale(const std::string& filePathImage);

  void pnm_read(std::ifstream &file, std::string& buf) const;

  BpImage<unsigned char> imageRead(const std::string& fileName,
    const image_type imageType, const bool weightedRGBConversion = true) const;

  //currently assuming single channel
  inline unsigned int getTotalPixels(/*const unsigned int numChannels = 1*/) const {
    return (widthHeight_[0] * widthHeight_[1]/* * numChannels*/);
  }
};

#endif /* BPIMAGE_H_ */
