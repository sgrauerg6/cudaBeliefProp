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
ARCHITECTURE_ADDITION inline T getMin(const T val1, const T val2) {
	return ((val1 < val2) ? val1 : val2);
}

template<typename T>
ARCHITECTURE_ADDITION inline T getMax(const T val1, const T val2) {
	return ((val1 > val2) ? val1 : val2);
}

//checks if the current point is within the image bounds
//assumed that input x/y vals are above zero since their unsigned int so no need for >= 0 check
ARCHITECTURE_ADDITION inline bool withinImageBounds(const unsigned int xVal, const unsigned int yVal, const unsigned int width, const unsigned int height) {
	return ((xVal < width) && (yVal < height));
}

#endif /* SHAREDUTILFUNCTS_H_ */
