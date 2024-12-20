/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file BpImage.h
 * @author Scott Grauer-Gray
 * @brief Declares class to define images that are used in bp processing
 * 
 * @copyright Copyright (c) 2024
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
#include <ranges>

namespace beliefprop {
  enum class ImageType { kPgmImage, kPpmImage };
  constexpr bool kUseWeightedRGBToGrayscaleConversion{true};
  constexpr std::string_view kPGMExt{"pgm"};
  constexpr std::string_view kPPMExt{"ppm"};
};

/**
 * @brief Class to define images that are used in bp processing
 * 
 * @tparam T 
 */
template <class T>
requires std::is_arithmetic_v<T>
class BpImage {
public:
  BpImage() : width_height_{0, 0} {}

  explicit BpImage(const std::array<unsigned int, 2>& width_height) : 
    width_height_{width_height}, pixels_(std::make_unique<T[]>(width_height_[0]*width_height_[1])) {}

  explicit BpImage(const std::array<unsigned int, 2>& width_height, const T* input_pixel_vals) :
    width_height_{width_height}, pixels_(std::make_unique<T[]>(width_height_[0] * width_height_[1]))
  {
    std::ranges::copy(input_pixel_vals, input_pixel_vals + (TotalPixels()), pixels_.get());
  }

  explicit BpImage(const std::string& file_name) {
    LoadImageAsGrayScale(file_name);
  }

  const std::unique_ptr<T[]>& UniquePtrToPixelData() const {
    return pixels_;
  }

  T* PointerToPixelsStart() const {
    return pixels_.get();
  }

  T PixelAtPoint(const std::array<unsigned int, 2>& point_xy) const {
    return PixelAtPoint(point_xy[1]*width_height_[0] + point_xy[0]);
  }

  T PixelAtPoint(unsigned int i) const {
    return (pixels_.get())[i];
  }

  void SetPixelAtPoint(const std::array<unsigned int, 2>& point_xy, T val) {
    SetPixelAtPoint((point_xy[1]*width_height_[0] + point_xy[0]), val);
  }

  void SetPixelAtPoint(unsigned int i, T val) {
    (pixels_.get())[i] = val;
  }

  unsigned int Width() const { return width_height_[0]; }
  unsigned int Height() const { return width_height_[1]; }

  void SaveImageAsPgm(const std::string& filename) const {
    std::ofstream file(filename, std::ios::out | std::ios::binary);
    file << "P5\n" << width_height_[0] << " " << width_height_[1] << "\n" << UCHAR_MAX << "\n";
    file.write((char*)(pixels_.get()), TotalPixels() * sizeof(char));
    file.close();
  }

protected:
  std::array<unsigned int, 2> width_height_;
  std::unique_ptr<T[]> pixels_;

  void LoadImageAsGrayScale(const std::string& file_path_image);

  void pnm_read(std::ifstream &file, std::string& buf) const;

  BpImage<unsigned char> ImageRead(const std::string& file_name,
    beliefprop::ImageType image_type, bool weighted_rgb_conversion = true) const;

  //currently assuming single channel
  inline unsigned int TotalPixels() const {
    return (width_height_[0] * width_height_[1]);
  }
};

#endif /* BPIMAGE_H_ */
