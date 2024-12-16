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
 * @file AVXTemplateSpFuncts.h
 * @author Scott Grauer-Gray
 * @brief Contains template specializations for AVX vector processing
 * 
 * @copyright Copyright (c) 2024
 */

#ifndef AVXTEMPLATESPFUNCTS_H_
#define AVXTEMPLATESPFUNCTS_H_

//this is only processed when on x86
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "RunImp/UtilityFuncts.h"

//used code from https://github.com/microsoft/DirectXMath/blob/master/Extensions/DirectXMathF16C.h
//for the values conversion on Windows since _cvtsh_ss and _cvtss_sh not supported in Visual Studio
template<> inline
short util_functs::ZeroVal<short>()
{
#ifdef _WIN32
  __m128 dataInAvxReg = _mm_set_ss(0.0);
  __m128i convertedData = _mm_cvtps_ph(dataInAvxReg, 0);
  return ((short*)& convertedData)[0];
#else
  return _cvtss_sh(0.0f, 0);
#endif
}

template<> inline
float util_functs::ConvertValToDifferentDataTypeIfNeeded<short, float>(short data)
{
#ifdef _WIN32
  __m128i dataInAvxReg = _mm_cvtsi32_si128(static_cast<int>(data));
  __m128 convertedData = _mm_cvtph_ps(dataInAvxReg);
  return ((float*)& convertedData)[0];
#else
  return _cvtsh_ss(data);
#endif
}

template<> inline
short util_functs::ConvertValToDifferentDataTypeIfNeeded<float, short>(float data)
{
#ifdef _WIN32
  __m128 dataInAvxReg = _mm_set_ss(data);
  __m128i convertedData = _mm_cvtps_ph(dataInAvxReg, 0);
  return ((short*)&convertedData)[0];
#else
  return _cvtss_sh(data, 0);
#endif
}

#endif //AVXTEMPLATESPFUNCTS_H_
