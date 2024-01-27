/*
 * ARMTemplateSpFuncts.h
 *
 *  Created on: Jun 26, 2024
 *      Author: scott
 */

#ifndef ARMTEMPLATESPFUNCTS_H_
#define ARMTEMPLATESPFUNCTS_H_

#include "RunImp/RunImpGenFuncts.h"

#ifdef COMPILING_FOR_ARM

#include <arm_neon.h>

template<> inline
float16_t GenProcessingFuncts::getZeroVal<float16_t>()
{
  return (float16_t)0.0f;
}

template<> inline
float GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<float16_t, float>(const float16_t valToConvert)
{
  return (float)valToConvert;
}

template<> inline
float16_t GenProcessingFuncts::convertValToDifferentDataTypeIfNeeded<float, float16_t>(const float valToConvert)
{
  //seems like simple cast function works
  return (float16_t)valToConvert;
}

#endif //COMPILING_FOR_ARM

#endif /* ARMTEMPLATESPFUNCTS_H_ */
