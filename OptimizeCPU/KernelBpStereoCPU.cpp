/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//This file defines the methods to perform belief propagation for disparity map estimation from stereo images on CUDA


#include "KernelBpStereoCPU.h"

//checks if the current point is within the image bounds
bool KernelBpStereoCPU::withinImageBoundsCPU(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

//retrieve the current 1-D index value of the given point at the given disparity in the data cost and message data
int KernelBpStereoCPU::retrieveIndexInDataAndMessageCPU(int xVal, int yVal, int width, int height, int currentDisparity, int totalNumDispVals, int offsetData)
{
	return RETRIEVE_INDEX_IN_DATA_OR_MESSAGE_ARRAY_EQUATION_CPU + offsetData;
}

template<typename T>
int KernelBpStereoCPU::getCheckerboardWidthCPU(int imageWidth)
{
	return (int)ceil(((float)imageWidth) / 2.0);
}

template<typename T>
T KernelBpStereoCPU::getZeroValCPU()
{
	return (T)0.0;
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<typename T>
void KernelBpStereoCPU::dtStereoCPU(T f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	T prev;
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = f[currentDisparity-1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		prev = f[currentDisparity+1] + (T)1.0;
		if (prev < f[currentDisparity])
			f[currentDisparity] = prev;
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<>
void KernelBpStereoCPU::dtStereoCPU<__m256>(__m256 f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m256 prev;
	__m256 vectorAllOneVal = _mm256_set1_ps(1.0f);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm256_add_ps(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm256_add_ps(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm256_min_ps(prev, f[currentDisparity]);
	}
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<>
void KernelBpStereoCPU::dtStereoCPU<__m256d>(__m256d f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m256d prev;
	__m256d vectorAllOneVal = _mm256_set1_pd(1.0);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm256_add_pd(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm256_add_pd(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm256_min_pd(prev, f[currentDisparity]);
	}
}

#if CPU_OPTIMIZATION_SETTING == USE_AVX_512

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<>
void KernelBpStereoCPU::dtStereoCPU<__m512>(__m512 f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m512 prev;
	__m512 vectorAllOneVal = _mm512_set1_ps(1.0f);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm512_add_ps(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm512_add_ps(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm512_min_ps(prev, f[currentDisparity]);
	}
}


// compute current message
template<>
void KernelBpStereoCPU::msgStereoCPU<__m512>(__m512 messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m512 messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m512 dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512 dst[NUM_POSSIBLE_DISPARITY_VALUES], __m512 disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512 minimum = _mm512_set1_ps(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm512_add_ps(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm512_add_ps(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_ps(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m512>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_ps(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m512 valToNormalize = _mm512_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm512_min_ps(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_ps(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_ps(valToNormalize, _mm512_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm512_sub_ps(dst[currentDisparity], valToNormalize);
	}
}


//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
template<>
void KernelBpStereoCPU::dtStereoCPU<__m512d>(__m512d f[NUM_POSSIBLE_DISPARITY_VALUES])
{
	__m512d prev;
	__m512d vectorAllOneVal = _mm512_set1_pd(1.0);
	for (int currentDisparity = 1; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		prev = _mm512_add_pd(f[currentDisparity-1], vectorAllOneVal);
		//prev = f[currentDisparity-1] + (T)1.0;
		/*if (prev < f[currentDisparity])
					f[currentDisparity] = prev;*/
		f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
	}

	for (int currentDisparity = NUM_POSSIBLE_DISPARITY_VALUES-2; currentDisparity >= 0; currentDisparity--)
	{
		//prev = f[currentDisparity+1] + (T)1.0;
		prev = _mm512_add_pd(f[currentDisparity+1], vectorAllOneVal);
		//if (prev < f[currentDisparity])
		//	f[currentDisparity] = prev;
		f[currentDisparity] = _mm512_min_pd(prev, f[currentDisparity]);
	}
}


// compute current message
template<>
void KernelBpStereoCPU::msgStereoCPU<__m512d>(__m512d messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m512d messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m512d dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m512d dst[NUM_POSSIBLE_DISPARITY_VALUES], __m512d disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m512d minimum = _mm512_set1_pd(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm512_add_pd(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm512_add_pd(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm512_min_pd(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m512d>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm512_add_pd(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m512d valToNormalize = _mm512_set1_pd(0.0);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm512_min_pd(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm512_add_pd(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm512_div_pd(valToNormalize, _mm512_set1_pd((double)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm512_sub_pd(dst[currentDisparity], valToNormalize);
	}
}

#endif

// compute current message
template<typename T>
void KernelBpStereoCPU::msgStereoCPU(T messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], T messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
	T messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], T dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
	T dst[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp)
{
	// aggregate and find min
	T minimum = INF_BP;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		if (dst[currentDisparity] < minimum)
			minimum = dst[currentDisparity];
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<T>(dst);

	// truncate 
	minimum += disc_k_bp;

	// normalize
	T valToNormalize = 0;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}

		valToNormalize += dst[currentDisparity];
	}
	
	valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++) 
		dst[currentDisparity] -= valToNormalize;
}

// compute current message
template<>
void KernelBpStereoCPU::msgStereoCPU<__m128i>(__m128i messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m128i messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m128i dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m128i dst[NUM_POSSIBLE_DISPARITY_VALUES], __m128i disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m256 minimum = _mm256_set1_ps(INF_BP);
	__m256 dstFloat[NUM_POSSIBLE_DISPARITY_VALUES];

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dstFloat[currentDisparity] = _mm256_add_ps((_mm256_cvtph_ps(messageValsNeighbor1[currentDisparity])), (_mm256_cvtph_ps(messageValsNeighbor2[currentDisparity])));
		dstFloat[currentDisparity] = _mm256_add_ps(dstFloat[currentDisparity], _mm256_cvtph_ps(messageValsNeighbor3[currentDisparity]));
		dstFloat[currentDisparity] = _mm256_add_ps(dstFloat[currentDisparity], _mm256_cvtph_ps(dataCosts[currentDisparity]));
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_ps(minimum, dstFloat[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m256>(dstFloat);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_ps(minimum, _mm256_cvtph_ps(disc_k_bp));

	// normalize
	//T valToNormalize = 0;
	__m256 valToNormalize = _mm256_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dstFloat[currentDisparity] = _mm256_min_ps(minimum, dstFloat[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_ps(valToNormalize, dstFloat[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_cvtps_ph(_mm256_sub_ps(dstFloat[currentDisparity], valToNormalize), 0);
	}
}

// compute current message
template<>
void KernelBpStereoCPU::msgStereoCPU<__m256>(__m256 messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m256 messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m256 dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256 dst[NUM_POSSIBLE_DISPARITY_VALUES], __m256 disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m256 minimum = _mm256_set1_ps(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm256_add_ps(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm256_add_ps(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm256_add_ps(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_ps(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m256>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_ps(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m256 valToNormalize = _mm256_set1_ps(0.0f);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm256_min_ps(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_ps(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_ps(valToNormalize, _mm256_set1_ps((float)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_sub_ps(dst[currentDisparity], valToNormalize);
	}
}


// compute current message
template<>
void KernelBpStereoCPU::msgStereoCPU<__m256d>(__m256d messageValsNeighbor1[NUM_POSSIBLE_DISPARITY_VALUES], __m256d messageValsNeighbor2[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d messageValsNeighbor3[NUM_POSSIBLE_DISPARITY_VALUES], __m256d dataCosts[NUM_POSSIBLE_DISPARITY_VALUES],
		__m256d dst[NUM_POSSIBLE_DISPARITY_VALUES], __m256d disc_k_bp)
{
	// aggregate and find min
	//T minimum = INF_BP;
	__m256d minimum = _mm256_set1_pd(INF_BP);

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		dst[currentDisparity] = _mm256_add_pd(messageValsNeighbor1[currentDisparity], messageValsNeighbor2[currentDisparity]);
		dst[currentDisparity] = _mm256_add_pd(dst[currentDisparity], messageValsNeighbor3[currentDisparity]);
		dst[currentDisparity] = _mm256_add_pd(dst[currentDisparity], dataCosts[currentDisparity]);
		//dst[currentDisparity] = messageValsNeighbor1[currentDisparity] + messageValsNeighbor2[currentDisparity] + messageValsNeighbor3[currentDisparity] + dataCosts[currentDisparity];
		//if (dst[currentDisparity] < minimum)
		//	minimum = dst[currentDisparity];
		minimum = _mm256_min_pd(minimum, dst[currentDisparity]);
	}

	//retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method (see "Efficient Belief Propagation for Early Vision")
	dtStereoCPU<__m256d>(dst);

	// truncate
	//minimum += disc_k_bp;
	minimum = _mm256_add_pd(minimum, disc_k_bp);

	// normalize
	//T valToNormalize = 0;
	__m256d valToNormalize = _mm256_set1_pd(0.0);


	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		/*if (minimum < dst[currentDisparity])
		{
			dst[currentDisparity] = minimum;
		}*/
		dst[currentDisparity] = _mm256_min_pd(minimum, dst[currentDisparity]);

		//valToNormalize += dst[currentDisparity];
		valToNormalize = _mm256_add_pd(valToNormalize, dst[currentDisparity]);
	}

	//valToNormalize /= NUM_POSSIBLE_DISPARITY_VALUES;
	valToNormalize = _mm256_div_pd(valToNormalize, _mm256_set1_pd((double)NUM_POSSIBLE_DISPARITY_VALUES));

	for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
	{
		//dst[currentDisparity] -= valToNormalize;
		dst[currentDisparity] = _mm256_sub_pd(dst[currentDisparity], valToNormalize);
	}
}

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<typename T>
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU(float* image1PixelsDevice, float* image2PixelsDevice, T* dataCostDeviceStereoCheckerboard1, T* dataCostDeviceStereoCheckerboard2, int widthImages, int heightImages, float lambda_bp, float data_k_bp)
{
	int imageCheckerboardWidth = getCheckerboardWidthCPU<T>(widthImages);

	#pragma omp parallel for
	for (int val = 0; val < (widthImages*heightImages); val++)
	{
		int yVal = val / widthImages;
		int xVal = val % widthImages;
	/*for (int yVal = 0; yVal < heightImages; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthImages; xVal++)
		{*/
			int indexVal;
			int xInCheckerboard = xVal / 2;

			if (withinImageBoundsCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages))
			{
				//make sure that it is possible to check every disparity value
				if ((xVal - (NUM_POSSIBLE_DISPARITY_VALUES-1)) >= 0)
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						float currentPixelImage1 = 0.0f;
						float currentPixelImage2 = 0.0f;

						if (withinImageBoundsCPU(xVal, yVal, widthImages, heightImages))
						{
							currentPixelImage1 = image1PixelsDevice[yVal * widthImages
									+ xVal];
							currentPixelImage2 = image2PixelsDevice[yVal * widthImages
									+ (xVal - currentDisparity)];
						}

						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = (T)(lambda_bp * std::min(((T)abs(currentPixelImage1 - currentPixelImage2)), (T)data_k_bp));
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = (T)(lambda_bp * std::min(((T)abs(currentPixelImage1 - currentPixelImage2)), (T)data_k_bp));
						}
					}
				}
				else
				{
					for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
					{
						indexVal = retrieveIndexInDataAndMessageCPU(xInCheckerboard, yVal, imageCheckerboardWidth, heightImages, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

						//data cost is equal to dataWeight value for weighting times the absolute difference in corresponding pixel intensity values capped at dataCostCap
						if (((xVal + yVal) % 2) == 0)
						{
							dataCostDeviceStereoCheckerboard1[indexVal] = getZeroValCPU<T>();
						}
						else
						{
							dataCostDeviceStereoCheckerboard2[indexVal] = getZeroValCPU<T>();
						}
					}
				}
			}
		//}
	}
}

void KernelBpStereoCPU::convertShortToFloat(float* destinationFloat, short* inputShort, int widthArray, int heightArray)
{
	int numDataInAvxVector = 8;
	#pragma omp parallel for
	for (int yVal = 0; yVal < heightArray; yVal++)
	{
		int startX = 0;
		int endXAvxStart = (((widthArray - startX) / numDataInAvxVector)
				* numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthArray;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			if (xVal > endXAvxStart) {
				xVal = endFinal - numDataInAvxVector;
			}
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++)
			{
				//load __m128i vector and convert to __m256 (set of 8 32-bit floats)
				__m256 data32Bit =
						_mm256_cvtph_ps(
								_mm_loadu_si128((__m128i*)(&inputShort[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthArray, heightArray, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)])));

				//store the __m256
				_mm256_storeu_ps(
						(&destinationFloat[retrieveIndexInDataAndMessageCPU(
								xVal, yVal, widthArray,
								heightArray, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)]), data32Bit);
			}
		}
	}
}

void KernelBpStereoCPU::convertFloatToShort(short* destinationShort, float* inputFloat, int widthArray, int heightArray)
{
	int numDataInAvxVector = 8;
	#pragma omp parallel for
	for (int yVal = 0; yVal < heightArray; yVal++)
	{
		int startX = 0;
		int endXAvxStart = (((widthArray - startX) / numDataInAvxVector)
				* numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthArray;

		for (int xVal = 0; xVal < endFinal; xVal += numDataInAvxVector)
		{
			if (xVal > endXAvxStart) {
				xVal = endFinal - numDataInAvxVector;
			}
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++)
			{
				//load __m256 vector and convert to __m128i (that is storing 16-bit floats)
				__m128i data16Bit =
						_mm256_cvtps_ph(
								_mm256_loadu_ps(&inputFloat[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthArray, heightArray, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]), 0);

				//store the 16-bit floats
				_mm_storeu_si128(
						((__m128i*)&destinationShort[retrieveIndexInDataAndMessageCPU(
								xVal, yVal, widthArray,
								heightArray, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)]), data16Bit);
			}
		}
	}
}

template<>
void KernelBpStereoCPU::initializeBottomLevelDataStereoCPU<short>(float* image1PixelsDevice, float* image2PixelsDevice, short* dataCostDeviceStereoCheckerboard1, short* dataCostDeviceStereoCheckerboard2, int widthImages, int heightImages, float lambda_bp, float data_k_bp)
{
	int imageCheckerboardWidth = getCheckerboardWidthCPU<short>(widthImages);
	float* dataCostDeviceStereoCheckerboard1Float = new float[imageCheckerboardWidth*heightImages*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostDeviceStereoCheckerboard2Float = new float[imageCheckerboardWidth*heightImages*NUM_POSSIBLE_DISPARITY_VALUES];

	initializeBottomLevelDataStereoCPU<float>(image1PixelsDevice, image2PixelsDevice, dataCostDeviceStereoCheckerboard1Float, dataCostDeviceStereoCheckerboard2Float, widthImages, heightImages, lambda_bp, data_k_bp);

	convertFloatToShort(dataCostDeviceStereoCheckerboard1, dataCostDeviceStereoCheckerboard1Float, imageCheckerboardWidth, heightImages);
	convertFloatToShort(dataCostDeviceStereoCheckerboard2, dataCostDeviceStereoCheckerboard2Float, imageCheckerboardWidth, heightImages);

	delete [] dataCostDeviceStereoCheckerboard1Float;
	delete [] dataCostDeviceStereoCheckerboard2Float;
}


template<typename T>
void KernelBpStereoCPU::printDataAndMessageValsAtPointKernelCPU(int xVal, int yVal, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, int widthLevelCheckerboardPart,
		int heightLevel)
{
	if (((xVal + yVal) % 2) == 0) {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	} else {
		printf("xVal: %d\n", xVal);
		printf("yVal: %d\n", yVal);
		for (int currentDisparity = 0;
				currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
				currentDisparity++) {
			printf("DISP: %d\n", currentDisparity);
			printf("messageUPrevStereoCheckerboard: %f \n",
					(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageDPrevStereoCheckerboard: %f \n",
					(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageLPrevStereoCheckerboard: %f \n",
					(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("messageRPrevStereoCheckerboard: %f \n",
					(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			printf("dataCostStereoCheckerboard: %f \n",
					(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
		}
	}
}


template<typename T>
void KernelBpStereoCPU::printDataAndMessageValsToPointKernelCPU(int xVal, int yVal, T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, int widthLevelCheckerboardPart,
		int heightLevel)
{
	int checkerboardAdjustment;
	if (((xVal + yVal) % 2) == 0)
		{
			checkerboardAdjustment = ((yVal)%2);
		}
		else //checkerboardToUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal+1)%2);
		}
	if (((xVal + yVal) % 2) == 0) {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		} else {
			printf("xVal: %d\n", xVal);
			printf("yVal: %d\n", yVal);
			for (int currentDisparity = 0;
					currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
					currentDisparity++) {
				printf("DISP: %d\n", currentDisparity);
				printf("messageUPrevStereoCheckerboard: %f \n",
						(float) messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal + 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageDPrevStereoCheckerboard: %f \n",
						(float) messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal - 1, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageLPrevStereoCheckerboard: %f \n",
						(float) messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								xVal / 2 + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("messageRPrevStereoCheckerboard: %f \n",
						(float) messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
								(xVal / 2 - 1) + checkerboardAdjustment, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
				printf("dataCostStereoCheckerboard: %f \n",
						(float) dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
								xVal / 2, yVal, widthLevelCheckerboardPart, heightLevel,
								currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
			}
		}
}


//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<typename T>
void KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* dataCostDeviceToWriteTo, int widthCurrentLevel, int heightCurrentLevel, int widthPrevLevel, int heightPrevLevel, int checkerboardPart, int offsetNum)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<T>(widthCurrentLevel);
	int widthCheckerboardPrevLevel = getCheckerboardWidthCPU<T>(widthPrevLevel);

	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboardCurrentLevel*heightCurrentLevel); val++)
	{
		int yVal = val / widthCheckerboardCurrentLevel;
		int xVal = val % widthCheckerboardCurrentLevel;

	/*for (int yVal = 0; yVal < heightCurrentLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthCheckerboardCurrentLevel; xVal++)
		{*/
			//if (withinImageBoundsCPU(xVal, yVal, widthCheckerboardCurrentLevel,
			//		heightCurrentLevel))
			{
				//add 1 or 0 to the x-value depending on checkerboard part and row adding to; CHECKERBOARD_PART_1 with slot at (0, 0) has adjustment of 0 in row 0,
				//while CHECKERBOARD_PART_2 with slot at (0, 1) has adjustment of 1 in row 0
				int checkerboardPartAdjustment = 0;

				if (checkerboardPart == CHECKERBOARD_PART_1) {
					checkerboardPartAdjustment = (yVal % 2);
				} else if (checkerboardPart == CHECKERBOARD_PART_2) {
					checkerboardPartAdjustment = ((yVal + 1) % 2);
				}

				//the corresponding x-values at the "lower" level depends on which checkerboard the pixel is in
				int xValPrev = xVal * 2 + checkerboardPartAdjustment;

				if (withinImageBoundsCPU(xValPrev, (yVal * 2 + 1),
						widthCheckerboardPrevLevel, heightPrevLevel)) {
					for (int currentDisparity = 0;
							currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
							currentDisparity++) {
						dataCostDeviceToWriteTo[retrieveIndexInDataAndMessageCPU(xVal,
								yVal, widthCheckerboardCurrentLevel,
								heightCurrentLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES)] =
								(dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
										xValPrev, (yVal * 2),
										widthCheckerboardPrevLevel,
										heightPrevLevel, currentDisparity,
										NUM_POSSIBLE_DISPARITY_VALUES,
										offsetNum)]
										+ dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2),
												widthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)]
										+ dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												widthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)]
										+ dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
												xValPrev, (yVal * 2 + 1),
												widthCheckerboardPrevLevel,
												heightPrevLevel,
												currentDisparity,
												NUM_POSSIBLE_DISPARITY_VALUES,
												offsetNum)]);
					}
				}
			}
		//}
	}
}


template<>
void KernelBpStereoCPU::initializeCurrentLevelDataStereoNoTexturesCPU<short>(short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* dataCostDeviceToWriteTo, int widthCurrentLevel, int heightCurrentLevel, int widthPrevLevel, int heightPrevLevel, int checkerboardPart, int offsetNum)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<short>(widthCurrentLevel);
	int widthCheckerboardPrevLevel = getCheckerboardWidthCPU<short>(widthPrevLevel);

	float* dataCostDeviceStereoCheckerboard1Float = new float[widthCheckerboardPrevLevel*heightPrevLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostDeviceStereoCheckerboard2Float = new float[widthCheckerboardPrevLevel*heightPrevLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostDeviceToWriteToFloat = new float[widthCheckerboardCurrentLevel*heightCurrentLevel*NUM_POSSIBLE_DISPARITY_VALUES];

	convertShortToFloat(dataCostDeviceStereoCheckerboard1Float, dataCostStereoCheckerboard1, widthCheckerboardPrevLevel, heightPrevLevel);
	convertShortToFloat(dataCostDeviceStereoCheckerboard2Float, dataCostStereoCheckerboard2, widthCheckerboardPrevLevel, heightPrevLevel);

	initializeCurrentLevelDataStereoNoTexturesCPU<float>(
			dataCostDeviceStereoCheckerboard1Float, dataCostDeviceStereoCheckerboard2Float,
			dataCostDeviceToWriteToFloat, widthCurrentLevel, heightCurrentLevel,
			widthPrevLevel, heightPrevLevel, checkerboardPart, offsetNum);

	convertFloatToShort(dataCostDeviceToWriteTo, dataCostDeviceToWriteToFloat, widthCheckerboardCurrentLevel, heightCurrentLevel);

	delete [] dataCostDeviceStereoCheckerboard1Float;
	delete [] dataCostDeviceStereoCheckerboard2Float;
	delete [] dataCostDeviceToWriteToFloat;
}


//initialize the message values at each pixel of the current level to the default value
template<typename T>
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU(T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1,
												T* messageRDeviceCurrentCheckerboard1, T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2,
												T* messageLDeviceCurrentCheckerboard2, T* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel)
{
	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboardAtLevel*heightLevel); val++)
	{
		int yVal = val / widthCheckerboardAtLevel;
		int xValInCheckerboard = val % widthCheckerboardAtLevel;
	/*for (int yVal = 0; yVal < heightLevel; yVal++)
	{
		#pragma omp parallel for
		for (int xValInCheckerboard = 0;
				xValInCheckerboard < widthCheckerboardAtLevel;
				xValInCheckerboard++)
		{*/
			//if (withinImageBoundsCPU(xValInCheckerboard, yVal,
			//		widthCheckerboardAtLevel, heightLevel))
			{
				//initialize message values in both checkerboards

				//set the message value at each pixel for each disparity to 0
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
				}

				//retrieve the previous message value at each movement at each pixel
				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
					messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(
							xValInCheckerboard, yVal, widthCheckerboardAtLevel,
							heightLevel, currentDisparity,
							NUM_POSSIBLE_DISPARITY_VALUES)] = getZeroValCPU<T>();
				}
			}
		//}
	}
}

template<>
void KernelBpStereoCPU::initializeMessageValsToDefaultKernelCPU<short>(short* messageUDeviceCurrentCheckerboard1, short* messageDDeviceCurrentCheckerboard1, short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1, short* messageUDeviceCurrentCheckerboard2, short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2, short* messageRDeviceCurrentCheckerboard2, int widthCheckerboardAtLevel, int heightLevel)
{
	float* messageUDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRDeviceCurrentCheckerboard1Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageUDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRDeviceCurrentCheckerboard2Float = new float[widthCheckerboardAtLevel*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];

	initializeMessageValsToDefaultKernelCPU<float>(messageUDeviceCurrentCheckerboard1Float, messageDDeviceCurrentCheckerboard1Float, messageLDeviceCurrentCheckerboard1Float,
			messageRDeviceCurrentCheckerboard1Float, messageUDeviceCurrentCheckerboard2Float, messageDDeviceCurrentCheckerboard2Float,
			messageLDeviceCurrentCheckerboard2Float, messageRDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);

	convertFloatToShort(messageUDeviceCurrentCheckerboard1, messageUDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageDDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageLDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageRDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageUDeviceCurrentCheckerboard2, messageUDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageDDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageLDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);
	convertFloatToShort(messageRDeviceCurrentCheckerboard2, messageRDeviceCurrentCheckerboard2Float, widthCheckerboardAtLevel, heightLevel);

	delete [] messageUDeviceCurrentCheckerboard1Float;
	delete [] messageDDeviceCurrentCheckerboard1Float;
	delete [] messageLDeviceCurrentCheckerboard1Float;
	delete [] messageRDeviceCurrentCheckerboard1Float;
	delete [] messageUDeviceCurrentCheckerboard2Float;
	delete [] messageDDeviceCurrentCheckerboard2Float;
	delete [] messageLDeviceCurrentCheckerboard2Float;
	delete [] messageRDeviceCurrentCheckerboard2Float;
}


//device portion of the kernel function to run the current iteration of belief propagation where the input messages and data costs come in as array in local memory
//and the output message values are stored in local memory
template<typename T>
void KernelBpStereoCPU::runBPIterationInOutDataInLocalMemCPU(T prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T dataMessage[NUM_POSSIBLE_DISPARITY_VALUES],
								T currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES], T currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES], T disc_k_bp)
{
	msgStereoCPU<T>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
			currentUMessage, disc_k_bp);

	msgStereoCPU<T>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
			currentDMessage, disc_k_bp);

	msgStereoCPU<T>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
			currentRMessage, disc_k_bp);

	msgStereoCPU<T>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
			currentLMessage, disc_k_bp);
}


//device portion of the kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the
//"checkerboard" scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
//this function uses local memory to store the message and data values at each disparity in the intermediate step of current message computation
//this function uses linear memory bound to textures to access the current data and message values
template<typename T>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU(
		T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2,
		int widthLevelCheckerboardPart, int heightLevel,
		int checkerboardToUpdate, int xVal, int yVal, int offsetData, float disc_k_bp)
{
	int indexWriteTo;
	int checkerboardAdjustment;

	//checkerboardAdjustment used for indexing into current checkerboard to update
	if (checkerboardToUpdate == CHECKERBOARD_PART_1)
	{
		checkerboardAdjustment = ((yVal)%2);
	}
	else //checkerboardToUpdate == CHECKERBOARD_PART_2
	{
		checkerboardAdjustment = ((yVal+1)%2);
	}

	//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
	//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
	//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
	{
		T prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		T dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				dataMessage[currentDisparity] = dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)];
				prevUMessage[currentDisparity] = messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevDMessage[currentDisparity] = messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevLMessage[currentDisparity] = messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevRMessage[currentDisparity] = messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				dataMessage[currentDisparity] = dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES, offsetData)];
				prevUMessage[currentDisparity] = messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevDMessage[currentDisparity] = messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevLMessage[currentDisparity] = messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
				prevRMessage[currentDisparity] = messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)];
			}
		}

		T currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
		T currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

		//uses the previous message values and data cost to calculate the current message values and store the results
		runBPIterationInOutDataInLocalMemCPU<T>(prevUMessage, prevDMessage, prevLMessage, prevRMessage, dataMessage,
							currentUMessage, currentDMessage, currentLMessage, currentRMessage, (T)disc_k_bp);

		//write the calculated message values to global memory
		for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
		{
			indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthLevelCheckerboardPart, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
			if (checkerboardToUpdate == CHECKERBOARD_PART_1)
			{
				messageUDeviceCurrentCheckerboard1[indexWriteTo] = currentUMessage[currentDisparity];
				messageDDeviceCurrentCheckerboard1[indexWriteTo] = currentDMessage[currentDisparity];
				messageLDeviceCurrentCheckerboard1[indexWriteTo] = currentLMessage[currentDisparity];
				messageRDeviceCurrentCheckerboard1[indexWriteTo] = currentRMessage[currentDisparity];
			}
			else //checkerboardToUpdate == CHECKERBOARD_PART_2
			{
				messageUDeviceCurrentCheckerboard2[indexWriteTo] = currentUMessage[currentDisparity];
				messageDDeviceCurrentCheckerboard2[indexWriteTo] = currentDMessage[currentDisparity];
				messageLDeviceCurrentCheckerboard2[indexWriteTo] = currentLMessage[currentDisparity];
				messageRDeviceCurrentCheckerboard2[indexWriteTo] = currentRMessage[currentDisparity];
			}
		}
	}
}


//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<typename T>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2,
								T* messageUDeviceCurrentCheckerboard1, T* messageDDeviceCurrentCheckerboard1, T* messageLDeviceCurrentCheckerboard1, T* messageRDeviceCurrentCheckerboard1,
								T* messageUDeviceCurrentCheckerboard2, T* messageDDeviceCurrentCheckerboard2, T* messageLDeviceCurrentCheckerboard2,
								T* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<T>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
							T>(dataCostStereoCheckerboard1,
							dataCostStereoCheckerboard2,
							messageUDeviceCurrentCheckerboard1,
							messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1,
							messageRDeviceCurrentCheckerboard1,
							messageUDeviceCurrentCheckerboard2,
							messageDDeviceCurrentCheckerboard2,
							messageLDeviceCurrentCheckerboard2,
							messageRDeviceCurrentCheckerboard2,
							widthCheckerboardCurrentLevel, heightLevel,
							checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
				//}
			//}
		}
}

//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
template<>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<
		float>(float* dataCostStereoCheckerboard1,
		float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1,
		float* messageDDeviceCurrentCheckerboard1,
		float* messageLDeviceCurrentCheckerboard1,
		float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2,
		float* messageDDeviceCurrentCheckerboard2,
		float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX256(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#elif CPU_OPTIMIZATION_SETTING == USE_AVX_512

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX512(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
				messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
				messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
				messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);
#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<float>(dataCostStereoCheckerboard1,
							dataCostStereoCheckerboard2,
							messageUDeviceCurrentCheckerboard1,
							messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1,
							messageRDeviceCurrentCheckerboard1,
							messageUDeviceCurrentCheckerboard2,
							messageDDeviceCurrentCheckerboard2,
							messageLDeviceCurrentCheckerboard2,
							messageRDeviceCurrentCheckerboard2,
							widthCheckerboardCurrentLevel, heightLevel,
							checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
				//}
			//}
		}
#endif
}

template<>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<
		short>(short* dataCostStereoCheckerboard1,
				short* dataCostStereoCheckerboard2,
				short* messageUDeviceCurrentCheckerboard1,
				short* messageDDeviceCurrentCheckerboard1,
				short* messageLDeviceCurrentCheckerboard1,
				short* messageRDeviceCurrentCheckerboard1,
				short* messageUDeviceCurrentCheckerboard2,
				short* messageDDeviceCurrentCheckerboard2,
				short* messageLDeviceCurrentCheckerboard2,
				short* messageRDeviceCurrentCheckerboard2, int widthLevel,
				int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUShortUseAVX256(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#else
		int widthCheckerboard = getCheckerboardWidthCPU<short>(widthLevel);
		float* dataCostStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* dataCostStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageUDeviceCurrentCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageDDeviceCurrentCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageLDeviceCurrentCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageRDeviceCurrentCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageUDeviceCurrentCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageDDeviceCurrentCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageLDeviceCurrentCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
		float* messageRDeviceCurrentCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];

		convertShortToFloat(dataCostStereoCheckerboard1Float, dataCostStereoCheckerboard1, widthCheckerboard, heightLevel);
		convertShortToFloat(dataCostStereoCheckerboard2Float, dataCostStereoCheckerboard2, widthCheckerboard, heightLevel);
		convertShortToFloat(messageUDeviceCurrentCheckerboard1Float, messageUDeviceCurrentCheckerboard1, widthCheckerboard, heightLevel);
		convertShortToFloat(messageDDeviceCurrentCheckerboard1Float, messageDDeviceCurrentCheckerboard1, widthCheckerboard, heightLevel);
		convertShortToFloat(messageLDeviceCurrentCheckerboard1Float, messageLDeviceCurrentCheckerboard1, widthCheckerboard, heightLevel);
		convertShortToFloat(messageRDeviceCurrentCheckerboard1Float, messageRDeviceCurrentCheckerboard1, widthCheckerboard, heightLevel);
		convertShortToFloat(messageUDeviceCurrentCheckerboard2Float, messageUDeviceCurrentCheckerboard2, widthCheckerboard, heightLevel);
		convertShortToFloat(messageDDeviceCurrentCheckerboard2Float, messageDDeviceCurrentCheckerboard2, widthCheckerboard, heightLevel);
		convertShortToFloat(messageLDeviceCurrentCheckerboard2Float, messageLDeviceCurrentCheckerboard2, widthCheckerboard, heightLevel);
		convertShortToFloat(messageRDeviceCurrentCheckerboard2Float, messageRDeviceCurrentCheckerboard2, widthCheckerboard, heightLevel);

		runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<
				float>(dataCostStereoCheckerboard1Float,
						dataCostStereoCheckerboard2Float,
						messageUDeviceCurrentCheckerboard1Float,
						messageDDeviceCurrentCheckerboard1Float,
						messageLDeviceCurrentCheckerboard1Float,
						messageRDeviceCurrentCheckerboard1Float,
						messageUDeviceCurrentCheckerboard2Float,
						messageDDeviceCurrentCheckerboard2Float,
						messageLDeviceCurrentCheckerboard2Float,
						messageRDeviceCurrentCheckerboard2Float, widthLevel,
						heightLevel, checkerboardPartUpdate, disc_k_bp);

		convertFloatToShort(dataCostStereoCheckerboard1, dataCostStereoCheckerboard1Float, widthCheckerboard, heightLevel);
		convertFloatToShort(dataCostStereoCheckerboard2, dataCostStereoCheckerboard2Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageUDeviceCurrentCheckerboard1, messageUDeviceCurrentCheckerboard1Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageDDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageLDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageRDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageUDeviceCurrentCheckerboard2, messageUDeviceCurrentCheckerboard2Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageDDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageLDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2Float, widthCheckerboard, heightLevel);
		convertFloatToShort(messageRDeviceCurrentCheckerboard2, messageRDeviceCurrentCheckerboard2Float, widthCheckerboard, heightLevel);

		delete [] dataCostStereoCheckerboard1Float;
		delete [] dataCostStereoCheckerboard2Float;
		delete [] messageUDeviceCurrentCheckerboard1Float;
		delete [] messageDDeviceCurrentCheckerboard1Float;
		delete [] messageLDeviceCurrentCheckerboard1Float;
		delete [] messageRDeviceCurrentCheckerboard1Float;
		delete [] messageUDeviceCurrentCheckerboard2Float;
		delete [] messageDDeviceCurrentCheckerboard2Float;
		delete [] messageLDeviceCurrentCheckerboard2Float;
		delete [] messageRDeviceCurrentCheckerboard2Float;

#endif
}

template<>
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPU<
		double>(double* dataCostStereoCheckerboard1,
		double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1,
		double* messageDDeviceCurrentCheckerboard1,
		double* messageLDeviceCurrentCheckerboard1,
		double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2,
		double* messageDDeviceCurrentCheckerboard2,
		double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{

#if CPU_OPTIMIZATION_SETTING == USE_AVX_256

	runBPIterationUsingCheckerboardUpdatesNoTexturesCPUDoubleUseAVX256(dataCostStereoCheckerboard1, dataCostStereoCheckerboard2,
			messageUDeviceCurrentCheckerboard1, messageDDeviceCurrentCheckerboard1, messageLDeviceCurrentCheckerboard1, messageRDeviceCurrentCheckerboard1,
			messageUDeviceCurrentCheckerboard2, messageDDeviceCurrentCheckerboard2, messageLDeviceCurrentCheckerboard2,
			messageRDeviceCurrentCheckerboard2, widthLevel, heightLevel, checkerboardPartUpdate, disc_k_bp);

#else

	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

		#pragma omp parallel for
		for (int val = 0; val < (widthCheckerboardCurrentLevel*heightLevel); val++)
		{
			int yVal = val / widthCheckerboardCurrentLevel;
			int xVal = val % widthCheckerboardCurrentLevel;
		/*for (int yVal = 0; yVal < heightLevel; yVal++)
		{
			#pragma omp parallel for
			for (int xVal = 0; xVal < widthLevel / 2; xVal++)
			{*/
				//if (withinImageBoundsCPU(xVal, yVal, widthLevel / 2, heightLevel)) {
					runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<double>(dataCostStereoCheckerboard1,
							dataCostStereoCheckerboard2,
							messageUDeviceCurrentCheckerboard1,
							messageDDeviceCurrentCheckerboard1,
							messageLDeviceCurrentCheckerboard1,
							messageRDeviceCurrentCheckerboard1,
							messageUDeviceCurrentCheckerboard2,
							messageDDeviceCurrentCheckerboard2,
							messageLDeviceCurrentCheckerboard2,
							messageRDeviceCurrentCheckerboard2,
							widthCheckerboardCurrentLevel, heightLevel,
							checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
				//}
			//}
		}
#endif
}

void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX256(float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m256 disc_k_bp_vector = _mm256_set1_ps(disc_k_bp);

	int numDataInAvxVector = 8;
	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal < endFinal; xVal += numDataInAvxVector)
		{
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xVal > endXAvxStart)
			{
				xVal = endFinal - numDataInAvxVector;
			}

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
				__m256 prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m256 dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm256_loadu_ps(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_ps(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_ps(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_ps(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_ps(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm256_loadu_ps(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_ps(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_ps(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_ps(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_ps(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m256 currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256 currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m256>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m256>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m256>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m256>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm256_storeu_ps(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_ps(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_ps(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_ps(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_storeu_ps(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_ps(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_ps(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_ps(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}

		/*for (int xVal = endXAvxStart + 8; xVal < endFinal; xVal ++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					float>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}*/
	}
}

void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUDoubleUseAVX256(double* dataCostStereoCheckerboard1, double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1, double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2, double* messageDDeviceCurrentCheckerboard2, double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m256d disc_k_bp_vector = _mm256_set1_pd((double)disc_k_bp);

	int numDataInAvxVector = 4;
	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal < endFinal; xVal += numDataInAvxVector)
		{
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xVal > endXAvxStart)
			{
				xVal = endFinal - numDataInAvxVector;
			}

			int indexWriteTo;

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
				__m256d prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m256d dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm256_loadu_pd(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_pd(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_pd(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_pd(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_pd(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm256_loadu_pd(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm256_loadu_pd(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm256_loadu_pd(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm256_loadu_pd(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm256_loadu_pd(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m256d currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m256d currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m256d>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m256d>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m256d>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m256d>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm256_storeu_pd(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_pd(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_pd(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_pd(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm256_storeu_pd(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm256_storeu_pd(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm256_storeu_pd(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm256_storeu_pd(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}

		/*for (int xVal = endXAvxStart + 4; xVal < endFinal; xVal ++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					double>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}*/
	}
}


void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUShortUseAVX256(
		short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2,
		short* messageUDeviceCurrentCheckerboard1,
		short* messageDDeviceCurrentCheckerboard1,
		short* messageLDeviceCurrentCheckerboard1,
		short* messageRDeviceCurrentCheckerboard1,
		short* messageUDeviceCurrentCheckerboard2,
		short* messageDDeviceCurrentCheckerboard2,
		short* messageLDeviceCurrentCheckerboard2,
		short* messageRDeviceCurrentCheckerboard2, int widthLevel,
		int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);
	__m128i disc_k_bp_vector = _mm256_cvtps_ph(_mm256_set1_ps(disc_k_bp), 0);

	int numDataInAvxVector = 8;
	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / numDataInAvxVector) * numDataInAvxVector - numDataInAvxVector) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal < endFinal; xVal += numDataInAvxVector)
		{
			//if past the last AVX start (since the next one would go beyond the row), set to numDataInAvxVector from the final pixel so processing the last numDataInAvxVector in avx
			//may be a few pixels that are computed twice but that's OK
			if (xVal > endXAvxStart)
			{
				xVal = endFinal - numDataInAvxVector;
			}

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			__m128i dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
			__m128i prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevUMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevDMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevLMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
						prevRMessage[currentDisparity] = _mm_loadu_si128((__m128i*)(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]));
					}
				}

				__m128i currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m128i currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m128i currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m128i currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				msgStereoCPU<__m128i>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m128i>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m128i>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m128i>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					int indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm_storeu_si128((__m128i*)&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm_storeu_si128((__m128i*)&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm_storeu_si128((__m128i*)&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			//}
		}

		/*for (int xVal = endXAvxStart + 8; xVal < endFinal; xVal ++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					float>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}*/
	}
}

#if CPU_OPTIMIZATION_SETTING == USE_AVX_512

//kernal function to run the current iteration of belief propagation in parallel using the checkerboard update method where half the pixels in the "checkerboard"
//scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUFloatUseAVX512(float* dataCostStereoCheckerboard1, float* dataCostStereoCheckerboard2,
		float* messageUDeviceCurrentCheckerboard1, float* messageDDeviceCurrentCheckerboard1, float* messageLDeviceCurrentCheckerboard1, float* messageRDeviceCurrentCheckerboard1,
		float* messageUDeviceCurrentCheckerboard2, float* messageDDeviceCurrentCheckerboard2, float* messageLDeviceCurrentCheckerboard2,
		float* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

	/*#pragma omp parallel for
	for (int val = 0; val < (widthLevel / 2)*heightLevel; val++)
	{
		int yVal = val / (widthLevel / 2);
		int xVal = val % (widthLevel / 2);*/
	//checkerboardAdjustment used for indexing into current checkerboard to update
	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / 16) * 16 - 16) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal <= endXAvxStart; xVal += 16)
		{
			int indexWriteTo;

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthCheckerboardCurrentLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			{
				__m512 prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m512 dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm512_loadu_ps(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_ps(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_ps(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_ps(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_ps(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm512_loadu_ps(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_ps(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_ps(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_ps(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_ps(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m512 currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512 disc_k_bp_vector = _mm512_set1_ps(disc_k_bp);

				msgStereoCPU<__m512>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m512>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m512>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m512>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm512_storeu_ps(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_ps(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_ps(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_ps(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm512_storeu_ps(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_ps(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_ps(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_ps(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			}
			//}
		}

		for (int xVal = endXAvxStart + 16; xVal < endFinal; xVal ++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					float>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}
	}
}

void KernelBpStereoCPU::runBPIterationUsingCheckerboardUpdatesNoTexturesCPUDoubleUseAVX512(double* dataCostStereoCheckerboard1, double* dataCostStereoCheckerboard2,
		double* messageUDeviceCurrentCheckerboard1, double* messageDDeviceCurrentCheckerboard1, double* messageLDeviceCurrentCheckerboard1, double* messageRDeviceCurrentCheckerboard1,
		double* messageUDeviceCurrentCheckerboard2, double* messageDDeviceCurrentCheckerboard2, double* messageLDeviceCurrentCheckerboard2,
		double* messageRDeviceCurrentCheckerboard2, int widthLevel, int heightLevel, int checkerboardPartUpdate, float disc_k_bp)
{
	int widthCheckerboardCurrentLevel = getCheckerboardWidthCPU<float>(widthLevel);

	#pragma omp parallel for
	for (int yVal = 1; yVal < heightLevel-1; yVal++)
	{
		//checkerboardAdjustment used for indexing into current checkerboard to update
		int checkerboardAdjustment;
		if (checkerboardPartUpdate == CHECKERBOARD_PART_1) {
			checkerboardAdjustment = ((yVal) % 2);
		} else //checkerboardPartUpdate == CHECKERBOARD_PART_2
		{
			checkerboardAdjustment = ((yVal + 1) % 2);
		}
		int startX = (checkerboardAdjustment == 1 ? 0 : 1);
		int endXAvxStart = ((((widthCheckerboardCurrentLevel - startX) - checkerboardAdjustment) / 8) * 8 - 8) + startX;
		int endFinal = widthCheckerboardCurrentLevel - checkerboardAdjustment;

		for (int xVal = startX; xVal <= endXAvxStart; xVal += 8)
		{
			int indexWriteTo;

			//may want to look into (xVal < (widthLevelCheckerboardPart - 1) since it may affect the edges
			//make sure that the current point is not an edge/corner that doesn't have four neighbors that can pass values to it
			//if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthLevelCheckerboardPart - 1)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			if ((xVal >= (1 - checkerboardAdjustment)) && (xVal < (widthCheckerboardCurrentLevel - checkerboardAdjustment)) && (yVal > 0) && (yVal < (heightLevel - 1)))
			{
				__m512d prevUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d prevDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d prevLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d prevRMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				__m512d dataMessage[NUM_POSSIBLE_DISPARITY_VALUES];

				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						dataMessage[currentDisparity] = _mm512_loadu_pd(&dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_pd(&messageUDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_pd(&messageDDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_pd(&messageLDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_pd(&messageRDeviceCurrentCheckerboard2[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						dataMessage[currentDisparity] = _mm512_loadu_pd(&dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevUMessage[currentDisparity] = _mm512_loadu_pd(&messageUDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal+1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevDMessage[currentDisparity] = _mm512_loadu_pd(&messageDDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(xVal, (yVal-1), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevLMessage[currentDisparity] = _mm512_loadu_pd(&messageLDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU((xVal + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
						prevRMessage[currentDisparity] = _mm512_loadu_pd(&messageRDeviceCurrentCheckerboard1[retrieveIndexInDataAndMessageCPU(((xVal - 1) + checkerboardAdjustment), (yVal), widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES)]);
					}
				}

				__m512d currentUMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d currentDMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d currentLMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d currentRMessage[NUM_POSSIBLE_DISPARITY_VALUES];
				__m512d disc_k_bp_vector = _mm512_set1_pd((double)disc_k_bp);

				msgStereoCPU<__m512d>(prevUMessage, prevLMessage, prevRMessage, dataMessage,
						currentUMessage, disc_k_bp_vector);

				msgStereoCPU<__m512d>(prevDMessage, prevLMessage, prevRMessage, dataMessage,
						currentDMessage, disc_k_bp_vector);

				msgStereoCPU<__m512d>(prevUMessage, prevDMessage, prevRMessage, dataMessage,
						currentRMessage, disc_k_bp_vector);

				msgStereoCPU<__m512d>(prevUMessage, prevDMessage, prevLMessage, dataMessage,
						currentLMessage, disc_k_bp_vector);

				//write the calculated message values to global memory
				for (int currentDisparity = 0; currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES; currentDisparity++)
				{
					indexWriteTo = retrieveIndexInDataAndMessageCPU(xVal, yVal, widthCheckerboardCurrentLevel, heightLevel, currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);
					if (checkerboardPartUpdate == CHECKERBOARD_PART_1)
					{
						_mm512_storeu_pd(&messageUDeviceCurrentCheckerboard1[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_pd(&messageDDeviceCurrentCheckerboard1[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_pd(&messageLDeviceCurrentCheckerboard1[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_pd(&messageRDeviceCurrentCheckerboard1[indexWriteTo], currentRMessage[currentDisparity]);
					}
					else //checkerboardPartUpdate == CHECKERBOARD_PART_2
					{
						_mm512_storeu_pd(&messageUDeviceCurrentCheckerboard2[indexWriteTo], currentUMessage[currentDisparity]);
						_mm512_storeu_pd(&messageDDeviceCurrentCheckerboard2[indexWriteTo], currentDMessage[currentDisparity]);
						_mm512_storeu_pd(&messageLDeviceCurrentCheckerboard2[indexWriteTo], currentLMessage[currentDisparity]);
						_mm512_storeu_pd(&messageRDeviceCurrentCheckerboard2[indexWriteTo], currentRMessage[currentDisparity]);
					}
				}
			}
			//}
		}

		for (int xVal = endXAvxStart + 8; xVal < endFinal; xVal++)
		{
			runBPIterationUsingCheckerboardUpdatesDeviceNoTexBoundAndLocalMemCPU<
					double>(dataCostStereoCheckerboard1,
					dataCostStereoCheckerboard2,
					messageUDeviceCurrentCheckerboard1,
					messageDDeviceCurrentCheckerboard1,
					messageLDeviceCurrentCheckerboard1,
					messageRDeviceCurrentCheckerboard1,
					messageUDeviceCurrentCheckerboard2,
					messageDDeviceCurrentCheckerboard2,
					messageLDeviceCurrentCheckerboard2,
					messageRDeviceCurrentCheckerboard2,
					widthCheckerboardCurrentLevel, heightLevel,
					checkerboardPartUpdate, xVal, yVal, 0, disc_k_bp);
		}
	}
}

#endif



//kernal to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernal works from the point of view of the pixel at the prev level that is being copied to four different places
template<typename T>
void KernelBpStereoCPU::copyPrevLevelToNextLevelBPCheckerboardStereoNoTexturesCPU(
		T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1,
		T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1,
		T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2,
		T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2,
		T* messageUDeviceCurrentCheckerboard1,
		T* messageDDeviceCurrentCheckerboard1,
		T* messageLDeviceCurrentCheckerboard1,
		T* messageRDeviceCurrentCheckerboard1,
		T* messageUDeviceCurrentCheckerboard2,
		T* messageDDeviceCurrentCheckerboard2,
		T* messageLDeviceCurrentCheckerboard2,
		T* messageRDeviceCurrentCheckerboard2, int widthCheckerboardPrevLevel,
		int heightLevelPrev, int widthCheckerboardNextLevel,
		int heightLevelNext, int checkerboardPart)
{
	#pragma omp parallel for
	for (int val = 0; val < (widthCheckerboardPrevLevel*heightLevelPrev); val++)
	{
		int yVal = val / widthCheckerboardPrevLevel;
		int xVal = val % widthCheckerboardPrevLevel;
	/*for (int yVal = 0; yVal < heightLevelPrev; yVal++)
	{
		#pragma omp parallel for
		for (int xVal = 0; xVal < widthCheckerboardPrevLevel; xVal++)
		{*/
			/*if (withinImageBoundsCPU(xVal, yVal, widthCheckerboardPrevLevel,
					heightLevelPrev))*/ {
				int heightCheckerboardNextLevel = heightLevelNext;

				int indexCopyTo;
				int indexCopyFrom;

				int checkerboardPartAdjustment;

				T prevValU;
				T prevValD;
				T prevValL;
				T prevValR;

				if (checkerboardPart == CHECKERBOARD_PART_1) {
					checkerboardPartAdjustment = (yVal % 2);
				} else if (checkerboardPart == CHECKERBOARD_PART_2) {
					checkerboardPartAdjustment = ((yVal + 1) % 2);
				}

				for (int currentDisparity = 0;
						currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
						currentDisparity++) {
					indexCopyFrom = retrieveIndexInDataAndMessageCPU(xVal, yVal,
							widthCheckerboardPrevLevel, heightLevelPrev,
							currentDisparity, NUM_POSSIBLE_DISPARITY_VALUES);

					if (checkerboardPart == CHECKERBOARD_PART_1) {
						prevValU =
								messageUPrevStereoCheckerboard1[indexCopyFrom];
						prevValD =
								messageDPrevStereoCheckerboard1[indexCopyFrom];
						prevValL =
								messageLPrevStereoCheckerboard1[indexCopyFrom];
						prevValR =
								messageRPrevStereoCheckerboard1[indexCopyFrom];
					} else if (checkerboardPart == CHECKERBOARD_PART_2) {
						prevValU =
								messageUPrevStereoCheckerboard2[indexCopyFrom];
						prevValD =
								messageDPrevStereoCheckerboard2[indexCopyFrom];
						prevValL =
								messageLPrevStereoCheckerboard2[indexCopyFrom];
						prevValR =
								messageRPrevStereoCheckerboard2[indexCopyFrom];
					}

					if (withinImageBoundsCPU(xVal * 2 + checkerboardPartAdjustment,
							yVal * 2, widthCheckerboardNextLevel,
							heightCheckerboardNextLevel)) {
						indexCopyTo = retrieveIndexInDataAndMessageCPU(
								(xVal * 2 + checkerboardPartAdjustment),
								(yVal * 2), widthCheckerboardNextLevel,
								heightCheckerboardNextLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES);

						messageUDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValR;

						messageUDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValR;
					}

					if (withinImageBoundsCPU(xVal * 2 + checkerboardPartAdjustment,
							yVal * 2 + 1, widthCheckerboardNextLevel,
							heightCheckerboardNextLevel)) {
						indexCopyTo = retrieveIndexInDataAndMessageCPU(
								(xVal * 2 + checkerboardPartAdjustment),
								(yVal * 2 + 1), widthCheckerboardNextLevel,
								heightCheckerboardNextLevel, currentDisparity,
								NUM_POSSIBLE_DISPARITY_VALUES);

						messageUDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard1[indexCopyTo] =
								prevValR;

						messageUDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValU;
						messageDDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValD;
						messageLDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValL;
						messageRDeviceCurrentCheckerboard2[indexCopyTo] =
								prevValR;
					}
				}
			}
		}
	//}
}


//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
template<typename T>
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoNoTexturesCPU(T* dataCostStereoCheckerboard1, T* dataCostStereoCheckerboard2, T* messageUPrevStereoCheckerboard1, T* messageDPrevStereoCheckerboard1, T* messageLPrevStereoCheckerboard1, T* messageRPrevStereoCheckerboard1, T* messageUPrevStereoCheckerboard2, T* messageDPrevStereoCheckerboard2, T* messageLPrevStereoCheckerboard2, T* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
		#pragma omp parallel for
		for (int val = 0; val < (widthLevel*heightLevel); val++)
			{
				int yVal = val / widthLevel;
				int xVal = val % widthLevel;
				/*for (int yVal = 0; yVal < heightLevel; yVal++)
				{		#pragma omp parallel for
		for (int xVal = 0; xVal < widthLevel; xVal++)
		{*/
			//if (withinImageBoundsCPU(xVal, yVal, widthLevel, heightLevel))
			{
				int widthCheckerboard = getCheckerboardWidthCPU<T>(widthLevel);
				int xValInCheckerboardPart = xVal / 2;

				if (((yVal + xVal) % 2) == 0) //if true, then pixel is from part 1 of the checkerboard; otherwise, it's from part 2
						{
					int checkerboardPartAdjustment = (yVal % 2);

					if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1)
							&& (yVal < (heightLevel - 1))) {
						// keep track of "best" disparity for current pixel
						int bestDisparity = 0;
						T best_val = INF_BP;
						for (int currentDisparity = 0;
								currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
								currentDisparity++) {
							T val =
									messageUPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
											xValInCheckerboardPart, (yVal + 1),
											widthCheckerboard, heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageDPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													(yVal - 1),
													widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageLPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageRPrevStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart - 1
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ dataCostStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)];

							if (val < (best_val)) {
								best_val = val;
								bestDisparity = currentDisparity;
							}
						}
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								bestDisparity;
					} else {
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								0;
					}
				} else //pixel from part 2 of checkerboard
				{
					int checkerboardPartAdjustment = ((yVal + 1) % 2);

					if ((xVal >= 1) && (xVal < (widthLevel - 1)) && (yVal >= 1)
							&& (yVal < (heightLevel - 1))) {

						// keep track of "best" disparity for current pixel
						int bestDisparity = 0;
						T best_val = INF_BP;
						for (int currentDisparity = 0;
								currentDisparity < NUM_POSSIBLE_DISPARITY_VALUES;
								currentDisparity++) {
							T val =
									messageUPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
											xValInCheckerboardPart, (yVal + 1),
											widthCheckerboard, heightLevel,
											currentDisparity,
											NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageDPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													(yVal - 1),
													widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageLPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ messageRPrevStereoCheckerboard1[retrieveIndexInDataAndMessageCPU(
													(xValInCheckerboardPart - 1
															+ checkerboardPartAdjustment),
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)]
											+ dataCostStereoCheckerboard2[retrieveIndexInDataAndMessageCPU(
													xValInCheckerboardPart,
													yVal, widthCheckerboard,
													heightLevel,
													currentDisparity,
													NUM_POSSIBLE_DISPARITY_VALUES)];

							if (val < (best_val)) {
								best_val = val;
								bestDisparity = currentDisparity;
							}
						}
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								bestDisparity;
					} else {
						disparityBetweenImagesDevice[yVal * widthLevel + xVal] =
								0;
					}
				}
			}
		//}
	}
}

template<>
void KernelBpStereoCPU::retrieveOutputDisparityCheckerboardStereoNoTexturesCPU<short>(short* dataCostStereoCheckerboard1, short* dataCostStereoCheckerboard2, short* messageUPrevStereoCheckerboard1, short* messageDPrevStereoCheckerboard1, short* messageLPrevStereoCheckerboard1, short* messageRPrevStereoCheckerboard1, short* messageUPrevStereoCheckerboard2, short* messageDPrevStereoCheckerboard2, short* messageLPrevStereoCheckerboard2, short* messageRPrevStereoCheckerboard2, float* disparityBetweenImagesDevice, int widthLevel, int heightLevel)
{
	int widthCheckerboard = getCheckerboardWidthCPU<short>(widthLevel);
	float* dataCostStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* dataCostStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageUPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRPrevStereoCheckerboard1Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageUPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageDPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageLPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];
	float* messageRPrevStereoCheckerboard2Float = new float[widthCheckerboard*heightLevel*NUM_POSSIBLE_DISPARITY_VALUES];

	convertShortToFloat(dataCostStereoCheckerboard1Float, dataCostStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloat(dataCostStereoCheckerboard2Float, dataCostStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloat(messageUPrevStereoCheckerboard1Float, messageUPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloat(messageDPrevStereoCheckerboard1Float, messageDPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloat(messageLPrevStereoCheckerboard1Float, messageLPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloat(messageRPrevStereoCheckerboard1Float, messageRPrevStereoCheckerboard1, widthCheckerboard, heightLevel);
	convertShortToFloat(messageUPrevStereoCheckerboard2Float, messageUPrevStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloat(messageDPrevStereoCheckerboard2Float, messageDPrevStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloat(messageLPrevStereoCheckerboard2Float, messageLPrevStereoCheckerboard2, widthCheckerboard, heightLevel);
	convertShortToFloat(messageRPrevStereoCheckerboard2Float, messageRPrevStereoCheckerboard2, widthCheckerboard, heightLevel);

	retrieveOutputDisparityCheckerboardStereoNoTexturesCPU<float>(
			dataCostStereoCheckerboard1Float, dataCostStereoCheckerboard2Float,
			messageUPrevStereoCheckerboard1Float, messageDPrevStereoCheckerboard1Float,
			messageLPrevStereoCheckerboard1Float, messageRPrevStereoCheckerboard1Float,
			messageUPrevStereoCheckerboard2Float, messageDPrevStereoCheckerboard2Float,
			messageLPrevStereoCheckerboard2Float, messageRPrevStereoCheckerboard2Float,
			disparityBetweenImagesDevice, widthLevel, heightLevel);

	delete [] dataCostStereoCheckerboard1Float;
	delete [] dataCostStereoCheckerboard2Float;
	delete [] messageUPrevStereoCheckerboard1Float;
	delete [] messageDPrevStereoCheckerboard1Float;
	delete [] messageLPrevStereoCheckerboard1Float;
	delete [] messageRPrevStereoCheckerboard1Float;
	delete [] messageUPrevStereoCheckerboard2Float;
	delete [] messageDPrevStereoCheckerboard2Float;
	delete [] messageLPrevStereoCheckerboard2Float;
	delete [] messageRPrevStereoCheckerboard2Float;
}
