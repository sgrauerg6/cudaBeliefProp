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
 * @file BpImage.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "BpImage.h"

template<class T>
requires std::is_arithmetic_v<T>
void BpImage<T>::LoadImageAsGrayScale(const std::string& file_path_image) {
  //check if PGM or PPM image (types currently supported)
  std::istringstream iss(file_path_image);
  std::string token;
  while (std::getline(iss, token, '.')) {
    //continue to get last token with the file extension
  }

  //last token after "." is file extension
  //use extension to check if image is pgm or ppm
  BpImage<unsigned char> init_image;
  if (token == beliefprop::kPGMExt) {
    init_image = ImageRead(file_path_image, beliefprop::ImageType::kPgmImage);
  } else if (token == beliefprop::kPPMExt) {
    init_image = ImageRead(
      file_path_image, beliefprop::ImageType::kPpmImage,
      beliefprop::kUseWeightedRGBToGrayscaleConversion);
  } else {
    //end function if not of type pgm or ppm with result
    //being BpImage with default constructor
    return;
  }

  //set width and height using data from file image and initialize
  //pixels array of size width_ * height_ with unique pointer
  width_height_ = {init_image.Width(), init_image.Height()};
  pixels_ = std::make_unique<T[]>(TotalPixels());

  //convert each pixel in dataRead to data type T and place in imageData array in same location
  std::ranges::transform(
    init_image.PointerToPixelsStart(),
    init_image.PointerToPixelsStart() + TotalPixels(),
    pixels_.get(), [] (const unsigned char i) -> T {return (T)i;});
}

template<class T>
requires std::is_arithmetic_v<T>
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
requires std::is_arithmetic_v<T>
BpImage<unsigned char> BpImage<T>::ImageRead(const std::string& file_name,
  beliefprop::ImageType image_type, bool weighted_rgb_conversion) const
{
  std::string buf;

  /* read header */
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
  pnm_read(file, buf);
  if (buf != "P5")
    std::cout << "ERROR READING FILE P5: " << file_name << std::endl;

  unsigned int cols, rows;
  pnm_read(file, buf);
  cols = std::stoul(buf);
  pnm_read(file, buf);
  rows = std::stoul(buf);

  pnm_read(file, buf);
  if (std::stoul(buf) > UCHAR_MAX)
    std::cout << "ERROR READING FILE UCHAR MAX: " << file_name << std::endl;

  BpImage<unsigned char> out_image(std::array<unsigned int, 2>{cols, rows});

  if (image_type == beliefprop::ImageType::kPgmImage) {
    /* read data */
    file.read((char*) (out_image.PointerToPixelsStart()),
      (cols * rows * sizeof(char)));
  }
  else if (image_type == beliefprop::ImageType::kPpmImage) {
    std::unique_ptr<char[]> rgb_image_ptr = std::make_unique<char[]>(3 * cols * rows);

    /* read data */
    file.read(rgb_image_ptr.get(), 3 * cols * rows * sizeof(char));
    file.close();

    //convert the RGB image to grayscale either
    //with each channel weighted the same or weighted
    //if weighted_rgb_conversion set to true
    for (unsigned int i = 0; i < (rows * cols); i++) {
      float r_channel_weight = 1.0f / 3.0f;
      float b_channel_weight = 1.0f / 3.0f;
      float g_channel_weight = 1.0f / 3.0f;
      if (weighted_rgb_conversion) {
        r_channel_weight = 0.299f;
        b_channel_weight = 0.587f;
        g_channel_weight = 0.114f;
      }
      out_image.PointerToPixelsStart()[i] = (unsigned char)std::floor(
        r_channel_weight * ((float) rgb_image_ptr[i * 3]) +
        g_channel_weight * ((float) rgb_image_ptr[i * 3 + 1]) +
        b_channel_weight * ((float) rgb_image_ptr[i * 3 + 2]) +
        0.5f);
    }
  }

  file.close();
  return out_image;
}

//each type that may be used instantiated here
template class BpImage<unsigned int>;
template class BpImage<float>;
template class BpImage<unsigned char>;
template class BpImage<char>;

