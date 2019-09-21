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
#include <iterator>
#include <iostream>
#include <fstream>
#include <climits>
#include <sstream>
#include <cmath>

enum image_type { PGM_IMAGE, PPM_IMAGE };
const bool USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION = true;

template <class T>
class BpImage {
public:
	BpImage() : width_(0), height_(0)
{}

	BpImage(const unsigned int width, const unsigned int height) : width_(width), height_(height), pixels_(new T[width*height])
{}

	BpImage(const unsigned int width, const unsigned int height, const T* input_pixel_vals) : width_(width), height_(height), pixels_(new T[width*height])
	{
		std::copy(input_pixel_vals, input_pixel_vals + (width*height), pixels_.get());
	}

	BpImage(const std::string& fileName)
	{
		loadImageAsGrayScale(fileName);
	}

	T* getPointerToPixelsStart() const
	{
		return &(pixels_[0]);
	}

	const T getPixelAtPoint(const int x, const int y) const
	{
		return (pixels_.get())[y*width_ + x];
	}

	const T getPixelAtPoint(const int i) const
	{
		return (pixels_.get())[i];
	}

	void setPixelAtPoint(const int x, const int y, const T val)
	{
		(pixels_.get())[y*width_ + x] = val;
	}

	void setPixelAtPoint(const int i, const T val)
	{
		(pixels_.get())[i] = val;
	}

	unsigned int getWidth() const { return width_; }
	unsigned int getHeight() const { return height_; }

	void saveImageAsPgm(const std::string& filename)
	{
		  std::ofstream file(filename, std::ios::out | std::ios::binary);

		  file << "P5\n" << width_ << " " << height_ << "\n" << UCHAR_MAX << "\n";
		  file.write((&pixels_[0]), width_ * height_ * sizeof(char));
		  file.close();
	}


protected:

	unsigned int width_;
	unsigned int height_;
	std::unique_ptr<T[]> pixels_;

	void loadImageAsGrayScale(const std::string& filePathImage) {
		std::string pgmExtension("pgm");
		std::string ppmExtension("ppm");
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
		if (token == pgmExtension) {
			initImage = imageRead(filePathImage, image_type::PGM_IMAGE);
		} else if (token == ppmExtension) {
			initImage = imageRead(filePathImage, image_type::PGM_IMAGE,
					USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION);
		}

		width_ = initImage.getWidth();
		height_ = initImage.getHeight();
		pixels_ = std::make_unique<T[]>(width_ * height_);

		//convert each pixel in dataRead to unsigned int and place in imageData array in same location
		std::transform(initImage.getPointerToPixelsStart(),
				initImage.getPointerToPixelsStart()
						+ (width_ * height_),
				&(pixels_[0]), [] (const unsigned char i) -> T {return (T)i;});
	}

	void pnm_read(std::ifstream &file, std::string& buf)
	{
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


	BpImage<unsigned char> imageRead(const std::string& fileName, image_type imageType, bool weightedRGBConversion = true)
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

			  BpImage<unsigned char> outImage(cols, rows);

			if (imageType == image_type::PGM_IMAGE) {
				/* read data */
				file.read((char*) (outImage.getPointerToPixelsStart()),
						(cols * rows * sizeof(char)));
			} else if (imageType == image_type::PPM_IMAGE) {
				std::unique_ptr<char[]> rgbImagePtr(new char[3 * cols * rows]);

				/* read data */
				file.read(&(rgbImagePtr[0]), 3 * cols * rows * sizeof(char));
				file.close();

				//convert the RGB image to grayscale
				for (unsigned int i = 0; i < (rows * cols); i++) {
					float rChannelWeight = 1.0f / 3.0f;
					float bChannelWeight = 1.0f / 3.0f;
					float gChannelWeight = 1.0f / 3.0f;
					if (weightedRGBConversion) {
						rChannelWeight = 0.299f;
						bChannelWeight = 0.587f;
						gChannelWeight = 0.114f;
					}
					outImage.getPointerToPixelsStart()[i] = (unsigned char) std::floor(
							rChannelWeight * ((float) rgbImagePtr[i * 3])
									+ gChannelWeight
											* ((float) rgbImagePtr[i * 3 + 1])
									+ bChannelWeight
											* ((float) rgbImagePtr[i * 3 + 2])
									+ 0.5f);
				}
			}

			  file.close();
			  return outImage;
		}
};

#endif /* BPIMAGE_H_ */
