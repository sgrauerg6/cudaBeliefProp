/*
 * BpImage.cpp
 *
 *  Created on: Sep 21, 2019
 *      Author: scott
 */

#include "BpImage.h"

template<class T>
void BpImage<T>::LoadImageAsGrayScale(const std::string& file_path_image) {
  std::string file_path_image_copy(file_path_image);

  //check if PGM or PPM image (types currently supported)
  std::istringstream iss(file_path_image_copy);
  std::string token;
  while (std::getline(iss, token, '.')) {
    //continue to get last token with the file extension
  }

  //last token after "." is file extension
  //use extension to check if image is pgm or ppm
  BpImage<unsigned char> init_image;
  if (token == kPGMExt) {
    init_image = ImageRead(file_path_image, ImageType::kPgmImage);
  } else if (token == kPPMExt) {
    init_image = ImageRead(file_path_image, ImageType::kPpmImage, kUseWeightedRGBToGrayscaleConversion);
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
  std::ranges::transform(init_image.PointerToPixelsStart(), init_image.PointerToPixelsStart() + TotalPixels(),
    pixels_.get(), [] (const unsigned char i) -> T {return (T)i;});
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
BpImage<unsigned char> BpImage<T>::ImageRead(const std::string& file_name,
  ImageType image_type, bool weighted_rgb_conversion) const
{
  std::string buf;

  /* read header */
  std::ifstream file(file_name, std::ios::in | std::ios::binary);
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

  BpImage<unsigned char> out_image(std::array<unsigned int, 2>{cols, rows});

  if (image_type == ImageType::kPgmImage) {
    /* read data */
    file.read((char*) (out_image.PointerToPixelsStart()),
      (cols * rows * sizeof(char)));
  }
  else if (image_type == ImageType::kPpmImage) {
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

