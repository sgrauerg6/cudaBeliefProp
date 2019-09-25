/*
 * RunBpStereoSet.cpp
 *
 *  Created on: Jun 16, 2019
 *      Author: scott
 */

#include "RunBpStereoSet.h"



#if (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_FLOAT)

template class RunBpStereoSet<float>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_DOUBLE)

template class RunBpStereoSet<double>;

#elif (CURRENT_DATA_TYPE_PROCESSING == DATA_TYPE_PROCESSING_HALF)

#ifdef COMPILING_FOR_ARM
template class RunBpStereoSet<float16_t>;
#else
template class RunBpStereoSet<short>;
#endif //COMPILING_FOR_ARM

#endif
