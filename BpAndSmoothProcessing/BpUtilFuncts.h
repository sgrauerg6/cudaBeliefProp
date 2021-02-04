/*
 * BpUtilFuncts.h
 * Namespace with utility functions for belief propagation processing
 *
 *  Created on: Feb 3, 2021
 *      Author: scott
 */

#ifndef BPUTILFUNCTS_H_
#define BPUTILFUNCTS_H_

#include <cmath>
#include <array>
#include "../ParameterFiles/bpRunSettings.h"

//namespace with utility function for bp processing
namespace bp_util_functs {

	inline unsigned int getCheckerboardWidthTargetDevice(const unsigned int widthLevel) {
		return (unsigned int)std::ceil(((float)widthLevel) / 2.0f);
	}

	inline unsigned int getPaddedCheckerboardWidth(const unsigned int checkerboardWidth)
	{
		//add "padding" to checkerboard width if necessary for alignment
		return ((checkerboardWidth % bp_params::NUM_DATA_ALIGN_WIDTH) == 0) ?
				checkerboardWidth :
				(checkerboardWidth + (bp_params::NUM_DATA_ALIGN_WIDTH - (checkerboardWidth % bp_params::NUM_DATA_ALIGN_WIDTH)));
	}

	template <typename T>
	inline unsigned long getNumDataForAlignedMemoryAtLevel(const std::array<unsigned int, 2>& widthHeightLevel,
			const unsigned int totalPossibleMovements)
	{
		const unsigned long numDataAtLevel = (unsigned long)getPaddedCheckerboardWidth(getCheckerboardWidthTargetDevice(widthHeightLevel[0]))
			* ((unsigned long)widthHeightLevel[1]) * (unsigned long)totalPossibleMovements;
		unsigned long numBytesAtLevel = numDataAtLevel * sizeof(T);

		if ((numBytesAtLevel % bp_params::BYTES_ALIGN_MEMORY) == 0) {
			return numDataAtLevel;
		}
		else {
			numBytesAtLevel += (bp_params::BYTES_ALIGN_MEMORY - (numBytesAtLevel % bp_params::BYTES_ALIGN_MEMORY));
			return (numBytesAtLevel / sizeof(T));
		}
	}
};

#endif /* BPUTILFUNCTS_H_ */
