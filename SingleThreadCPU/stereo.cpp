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

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp0()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp0()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp0()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp1()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp1()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp1()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp2()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp2()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp2()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp3()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp3()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp3()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp4()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp4()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp4()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp5()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp5()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp5()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>();
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoCPUSingleThreadFloat_KnownDisp6()
{
	return new RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>();
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoCPUSingleThreadDouble_KnownDisp6()
{
	return new RunBpStereoCPUSingleThread<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>();
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoCPUSingleThreadShort_KnownDisp6()
{
	return new RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>();
}

#endif //_WIN32
