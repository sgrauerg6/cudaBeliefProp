/*
 * SmoothImageCPU.h
 *
 *  Created on: Jun 17, 2019
 *      Author: scott
 */

#ifndef SMOOTHIMAGECPU_H_
#define SMOOTHIMAGECPU_H_

#include <algorithm>
#include <memory>
#include "../SharedFuncts/SharedSmoothImageFuncts.h"
#include "../BpAndSmoothProcessing/SmoothImage.h"
#include "KernelBpStereoCPU.h"
#include "../ParameterFiles/bpRunSettings.h"

template <typename T=float*>
class SmoothImageCPU : public SmoothImage<T> {
public:

	//function to use the CPU-image filter to apply a guassian filter to the a single images
	//input images have each pixel stored as an unsigned in (value between 0 and 255 assuming 8-bit grayscale image used)
	//output filtered images have each pixel stored as a float after the image has been smoothed with a Gaussian filter of sigmaVal
	void operator()(const BpImage<unsigned int>& inImage, const float sigmaVal, T smoothedImage) override
	{
		//if sigmaVal < MIN_SIGMA_VAL_SMOOTH, then don't smooth image...just convert the input image
		//of unsigned ints to an output image of float values
		if (sigmaVal < MIN_SIGMA_VAL_SMOOTH) {
			//call kernal to convert input unsigned int pixels to output float pixels on the device
			convertUnsignedIntImageToFloatCPU(inImage.getPointerToPixelsStart(), smoothedImage,
					inImage.getWidth(), inImage.getHeight());
		}
		//otherwise apply a Guassian filter to the images
		else
		{
			//retrieve output filter (float array in unique_ptr) and size
			auto outFilterAndSize = this->makeFilter(sigmaVal);
			auto filter = std::move(outFilterAndSize.first);
			const unsigned int sizeFilter{outFilterAndSize.second};

			//space for intermediate image (when the image has been filtered horizontally but not vertically)
			std::unique_ptr<float[]> intermediateImage = std::make_unique<float[]>(inImage.getWidth() * inImage.getHeight());

			//first filter the image horizontally, then vertically; the result is applying a 2D gaussian filter with the given sigma value to the image
			filterImageAcrossCPU<unsigned int>(inImage.getPointerToPixelsStart(), &(intermediateImage[0]), inImage.getWidth(),
					inImage.getHeight(), &(filter[0]), sizeFilter);

			//now use the vertical filter to complete the smoothing of image 1 on the device
			filterImageVerticalCPU<float>(&(intermediateImage[0]), smoothedImage,
					inImage.getWidth(), inImage.getHeight(), &(filter[0]), sizeFilter);
		}
	}

private:
	//convert the unsigned int pixels to float pixels in an image when
	//smoothing is not desired but the pixels need to be converted to floats
	//the input image is stored as unsigned ints in the texture imagePixelsUnsignedInt
	//output filtered image stored in floatImagePixels
	void convertUnsignedIntImageToFloatCPU(unsigned int* imagePixelsUnsignedInt, float* floatImagePixels,
			const unsigned int widthImages, const unsigned int heightImages)
	{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
	KernelBpStereoCPU::tPool.parallelize_loop(0, widthImages * heightImages,
	  [&imagePixelsUnsignedInt, &floatImagePixels, &widthImages, &heightImages](const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / widthImages;
			  const unsigned int xVal = val % widthImages;
			  floatImagePixels[yVal * widthImages + xVal] =
					1.0f * imagePixelsUnsignedInt[yVal * widthImages + xVal];
		  }
	  });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
	KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, widthImages * heightImages,
	  [&imagePixelsUnsignedInt, &floatImagePixels, &widthImages, &heightImages](const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / widthImages;
			  const unsigned int xVal = val % widthImages;
			  floatImagePixels[yVal * widthImages + xVal] =
					1.0f * imagePixelsUnsignedInt[yVal * widthImages + xVal];
		  }
	  });
	#else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
		#pragma omp parallel for
#ifdef _WIN32
		for (int val = 0; val < widthImages * heightImages; val++) {
#else
		for (unsigned int val = 0; val < widthImages * heightImages; val++) {
#endif //_WIN32
			const unsigned int yVal = val / widthImages;
			const unsigned int xVal = val % widthImages;
			floatImagePixels[yVal * widthImages + xVal] =
					1.0f * imagePixelsUnsignedInt[yVal * widthImages + xVal];
		}
	#endif //CPU_PARALLELIZATION_METHOD
	}

	//apply a horizontal filter on each pixel of the image in parallel
	template<typename U>
	void filterImageAcrossCPU(U* imagePixelsToFilter, float* filteredImagePixels,
			const unsigned int widthImages, const unsigned int heightImages,
			float* imageFilter, const unsigned int sizeFilter)
	{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
		KernelBpStereoCPU::tPool.parallelize_loop(0, widthImages * heightImages,
	    [&imagePixelsToFilter, &filteredImagePixels, &widthImages, &heightImages, &imageFilter, &sizeFilter]
		(const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / widthImages;
  			  const unsigned int xVal = val % widthImages;
			  filterImageAcrossProcessPixel<U>(xVal, yVal, imagePixelsToFilter, filteredImagePixels,
					widthImages, heightImages, imageFilter, sizeFilter);
		  }
	    });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
		KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, widthImages * heightImages,
	    [&imagePixelsToFilter, &filteredImagePixels, &widthImages, &heightImages, &imageFilter, &sizeFilter]
		(const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / widthImages;
  			  const unsigned int xVal = val % widthImages;
			  filterImageAcrossProcessPixel<U>(xVal, yVal, imagePixelsToFilter, filteredImagePixels,
					widthImages, heightImages, imageFilter, sizeFilter);
		  }
	    });
    #else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
		#pragma omp parallel for
#ifdef _WIN32
		for (int val = 0; val < widthImages * heightImages; val++) {
#else
		for (unsigned int val = 0; val < widthImages * heightImages; val++) {
#endif //_WIN32
			const unsigned int yVal = val / widthImages;
			const unsigned int xVal = val % widthImages;
			filterImageAcrossProcessPixel<U>(xVal, yVal, imagePixelsToFilter, filteredImagePixels,
					widthImages, heightImages, imageFilter, sizeFilter);
		}
	#endif //CPU_PARALLELIZATION_METHOD
	}

	//apply a vertical filter on each pixel of the image in parallel
	template<typename U>
	void filterImageVerticalCPU(U* imagePixelsToFilter, T filteredImagePixels,
			const unsigned int widthImages, const unsigned int heightImages,
			float* imageFilter, const unsigned int sizeFilter)
	{
	#if (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_CHUNKS)
		KernelBpStereoCPU::tPool.parallelize_loop(0, widthImages * heightImages,
	    [&imagePixelsToFilter, &filteredImagePixels, &widthImages, &heightImages, &imageFilter, &sizeFilter]
		(const unsigned int &a, const unsigned int &b) {
		  for (unsigned int val = a; val < b; val++) {
			  const unsigned int yVal = val / widthImages;
	  		  const unsigned int xVal = val % widthImages;
			  filterImageVerticalProcessPixel<U>(xVal, yVal, imagePixelsToFilter,
					filteredImagePixels, widthImages, heightImages, imageFilter, sizeFilter);
		  }
	    });
	#elif (CPU_PARALLELIZATION_METHOD == USE_THREAD_POOL_DISTRIBUTED)
		KernelBpStereoCPU::tPool.parallelize_loop_distribute_iters(0, widthImages * heightImages,
	    [&imagePixelsToFilter, &filteredImagePixels, &widthImages, &heightImages, &imageFilter, &sizeFilter]
		(const unsigned int &a, const unsigned int &b, const unsigned int &numBlocks) {
		  for (unsigned int val = a; val < b; val += numBlocks) {
			  const unsigned int yVal = val / widthImages;
	  		  const unsigned int xVal = val % widthImages;
			  filterImageVerticalProcessPixel<U>(xVal, yVal, imagePixelsToFilter,
					filteredImagePixels, widthImages, heightImages, imageFilter, sizeFilter);
		  }
	    });
    #else //(CPU_PARALLELIZATION_METHOD == USE_OPENMP)
		#pragma omp parallel for
#ifdef _WIN32
		for (int val = 0; val < widthImages * heightImages; val++) {
#else
		for (unsigned int val = 0; val < widthImages * heightImages; val++) {
#endif //_WIN32
			const unsigned int yVal = val / widthImages;
			const unsigned int xVal = val % widthImages;
			filterImageVerticalProcessPixel<U>(xVal, yVal, imagePixelsToFilter,
					filteredImagePixels, widthImages, heightImages, imageFilter, sizeFilter);
		}
	#endif //CPU_PARALLELIZATION_METHOD
	}
};

#endif /* SMOOTHIMAGECPU_H_ */
