/*
 Copyright (C) 2006 Pedro Felzenszwalb

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

#include "stereo.h"

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp0()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp0()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp0()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[0].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp1()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp1()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp1()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[1].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp2()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp2()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp2()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[2].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp3()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp3()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp3()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[3].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp4()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp4()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp4()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[4].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp5()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp5()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp5()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[5].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp6()
{
  return new RunBpStereoCPUSingleThread<float, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp6()
{
  return new RunBpStereoCPUSingleThread<double, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp6()
{
  return new RunBpStereoCPUSingleThread<short, bp_params::STEREO_SETS_TO_PROCESS[6].numDispVals_>();
}

#endif //_WIN32
