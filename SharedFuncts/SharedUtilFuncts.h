/*
 * SharedUtilFuncts.h
 *
 *  Created on: Jun 24, 2019
 *      Author: scott
 */

#ifndef SHAREDUTILFUNCTS_H_
#define SHAREDUTILFUNCTS_H_

#ifdef PROCESSING_ON_GPU
#define ARCHITECTURE_ADDITION __device__
#else
#define ARCHITECTURE_ADDITION
#endif

template<typename T>
ARCHITECTURE_ADDITION inline T getMin(T val1, T val2)
{
	return ((val1 < val2) ? val1 : val2);
}

template<typename T>
ARCHITECTURE_ADDITION inline T getMax(T val1, T val2)
{
	return ((val1 > val2) ? val1 : val2);
}

//checks if the current point is within the image bounds
ARCHITECTURE_ADDITION inline bool withinImageBounds(int xVal, int yVal, int width, int height)
{
	return ((xVal >= 0) && (xVal < width) && (yVal >= 0) && (yVal < height));
}

#endif /* SHAREDUTILFUNCTS_H_ */
