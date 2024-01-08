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

//This file contains the "main" function that drives the CUDA BP implementation

#include <array>
#include <vector>
#include <thread>
#include <string_view>
#include "../ParameterFiles/bpStructsAndEnums.h"

//specify that running optimized CUDA run (used in RunAndEvaluateBpResults.h)
#define OPTIMIZED_CUDA_RUN

//option to optimize parallel parameters by running BP w/ multiple parallel parameters options by
//finding the parallel parameters with the lowest runtime, and then setting the parallel parameters
//to the best found parallel parameters in the final run
constexpr bool OPTIMIZE_PARALLEL_PARAMS{true};

//default setting is to use the allow different thread block dimensions on kernels in same run
//testing on has found that using different parallel parameters (corresponding to thread block dimensions)
//in different kernels in the optimized CUDA implementation can decrease runtime
constexpr beliefprop::OptParallelParamsSetting optParallelParamsSetting{beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN};

//parallel parameter options to run to retrieve optimized parallel parameters in CUDA implementation
//parallel parameter corresponds to thread block dimensions in CUDA implementation
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{  {16, 1}, {32, 1}, {32, 2}, {32, 3}, {32, 4}, {32, 5},
  {32, 6},{32, 8}, {64, 1}, {64, 2}, {64, 3}, {64, 4}, {128, 1}, {128, 2}, {256, 1}, {32, 10}, {32, 12}, {32, 14}, {32, 16},
  {64, 5}, {64, 6}, {64, 7}, {64, 8}, {128, 3}, {128, 4}, {256, 2}};
constexpr std::array<unsigned int, 2> PARALLEL_PARAMS_DEFAULT{{32, 4}};

//name of processor used for running optimized implementation
constexpr std::string_view PROCESSOR_NAME{""};

//functions in RunAndEvaluateBpResults use above constants
#include "RunAndEvaluateBpResults.h"

int main(int argc, char** argv)
{
  RunAndEvaluateBpResults::runBpOnStereoSets<beliefprop::AccSetting::CUDA>();
  return 0;
}
