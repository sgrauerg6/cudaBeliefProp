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

enum class image_type { PGM_IMAGE, PPM_IMAGE };
const bool USE_WEIGHTED_RGB_TO_GRAYSCALE_CONVERSION = true;
const std::string PGM_EXTENSION = "pgm";
const std::string PPM_EXTENSION = "ppm";

template <class T>
class BpImage {
public:
	BpImage() : width_(0), height_(0)
{}

	BpImage(const unsigned int width, const unsigned int height) : width_(width), height_(height), pixels_(std::make_unique<T[]>(width*height))
{}

	BpImage(const unsigned int width, const unsigned int height, const T* input_pixel_vals) : width_(width), height_(height), pixels_(std::make_unique<T[]>(width*height))
	{
		std::copy(input_pixel_vals, input_pixel_vals + (width*height), pixels_.get());
	}

	BpImage(const std::string& fileName)
	{
		loadImageAsGrayScale(fileName);
	}

	const std::unique_ptr<T[]>& getUniquePtrToPixelData()
	{
		return pixels_;
	}

	T* getPointerToPixelsStart() const
	{
		return &(pixels_[0]);
	}

	const T getPixelAtPoint(const int x, const int y) const
	{
		return getPixelAtPoint(y*width_ + x);
	}

	const T getPixelAtPoint(const int i) const
	{
		return (pixels_.get())[i];
	}

	void setPixelAtPoint(const int x, const int y, const T val)
	{
		setPixelAtPoint((y*width_ + x), val);
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
		  file.write((char*)(&pixels_[0]), width_ * height_ * sizeof(char));
		  file.close();
	}

protected:

	unsigned int width_;
	unsigned int height_;
	std::unique_ptr<T[]> pixels_;

	void loadImageAsGrayScale(const std::string& filePathImage);

	void pnm_read(std::ifstream &file, std::string& buf) const;

	BpImage<unsigned char> imageRead(const std::string& fileName,
			image_type imageType, bool weightedRGBConversion = true) const;
};

#endif /* BPIMAGE_H_ */
