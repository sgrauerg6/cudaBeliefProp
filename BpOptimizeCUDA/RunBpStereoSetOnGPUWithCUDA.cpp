/*
 Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//Defines the methods to run BP Stereo implementation on a series of images using various options

#include "RunBpStereoSetOnGPUWithCUDA.h"

#ifdef _WIN32

__declspec(dllexport) RunBpStereoSet<float, 0>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, 0>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, 0>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, 0>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, 0>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, 0>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp0()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp0()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp0()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp1()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp1()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp1()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp2()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp2()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp2()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp3()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp3()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp3()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[3]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp4()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp4()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp4()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[4]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp5()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp5()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp5()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[5]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoSetOnGPUWithCUDAFloat_KnownDisp6()
{
  return new RunBpStereoSetOnGPUWithCUDA<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoSetOnGPUWithCUDADouble_KnownDisp6()
{
  return new RunBpStereoSetOnGPUWithCUDA<double, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

__declspec(dllexport) RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>* __cdecl createRunBpStereoSetOnGPUWithCUDAShort_KnownDisp6()
{
  return new RunBpStereoSetOnGPUWithCUDA<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[6]>(beliefprop::ParallelParameters(bp_params::LEVELS_BP));
}

#endif //_WIN32
