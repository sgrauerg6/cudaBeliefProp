/*
 * AVXTemplateSpFuncts.h
 *
 *  Created on: Jun 26, 2024
 *      Author: scott
 */

#ifndef AVXTEMPLATESPFUNCTS_H_
#define AVXTEMPLATESPFUNCTS_H_

//this is only processed when on x86
#ifdef _WIN32
#include <intrin.h>
#else
#include <x86intrin.h>
#endif

#include "RunImp/RunImpGenFuncts.h"

//used code from https://github.com/microsoft/DirectXMath/blob/master/Extensions/DirectXMathF16C.h
//for the values conversion on Windows since _cvtsh_ss and _cvtss_sh not supported in Visual Studio
template<> inline
short run_imp_util::getZeroVal<short>()
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
float run_imp_util::convertValToDifferentDataTypeIfNeeded<short, float>(short data)
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
short run_imp_util::convertValToDifferentDataTypeIfNeeded<float, short>(float data)
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
