/*
Copyright (C) 2024 Scott Grauer-Gray

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

/**
 * @file ARMTemplateSpFuncts.h
 * @author Scott Grauer-Gray
 * @brief Contains template specializations for ARM/NEON vector processing
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef ARMTEMPLATESPFUNCTS_H_
#define ARMTEMPLATESPFUNCTS_H_

#include "RunImp/UtilityFuncts.h"

#if defined(COMPILING_FOR_ARM)

#include <arm_neon.h>

template<> inline
float16_t util_functs::ZeroVal<float16_t>()
{
  return (float16_t)0.0f;
}

template<> inline
float util_functs::ConvertValToDifferentDataTypeIfNeeded<float16_t, float>(
  float16_t valToConvert)
{
  return (float)valToConvert;
}

template<> inline
float16_t util_functs::ConvertValToDifferentDataTypeIfNeeded<float, float16_t>(
  float valToConvert)
{
  //seems like simple cast function works
  return (float16_t)valToConvert;
}

#endif //COMPILING_FOR_ARM

#endif /* ARMTEMPLATESPFUNCTS_H_ */
