/*
 * RunBpStereoSet.cpp
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#include "RunBpStereoSet.h"



#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class RunBpStereoSet<float16_t, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;
#else
template class RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES>;
#endif //COMPILING_FOR_ARM

#endif
