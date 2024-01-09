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

//This file contains the "main" function that drives the optimized CPU BP implementation

#include <memory>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <numeric>
#include <algorithm>
#include "../ParameterFiles/bpStructsAndEnums.h"

#ifdef COMPILING_FOR_ARM
#include <arm_neon.h> //needed for float16_t type
#endif

//specify that running original code only (used in RunAndEvaluateBpResults.h)
#define ORIG_CODE_CPU_RUN

//option to optimize parallel parameters by running BP w/ multiple parallel parameters options by
//finding the parallel parameters with the lowest runtime, and then setting the parallel parameters
//to the best found parallel parameters in the final run
constexpr bool OPTIMIZE_PARALLEL_PARAMS{true};

//default setting is to use the same parallel parameters for all kernels in run
//testing on i7-11800H has found that using different parallel parameters (corresponding to OpenMP thread counts)
//in different kernels in the optimized CPU implementation can increase runtime (may want to test on additional processors)
constexpr beliefprop::OptParallelParamsSetting optParallelParamsSetting{beliefprop::OptParallelParamsSetting::SAME_PARALLEL_PARAMS_ALL_KERNELS_IN_RUN};

//parallel parameter options to run to retrieve optimized parallel parameters in optimized CPU implementation
//parallel parameter corresponds to number of OpenMP threads in optimized CPU implementation
const unsigned int NUM_THREADS_CPU{std::thread::hardware_concurrency()};
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS{
\s\s{ NUM_THREADS_CPU, 1}, { (3 * NUM_THREADS_CPU) / 4 , 1}, { NUM_THREADS_CPU / 2, 1}/*,
\s\s{ NUM_THREADS_CPU / 4, 1}, { NUM_THREADS_CPU / 8, 1}*/};
const std::vector<std::array<unsigned int, 2>> PARALLEL_PARAMETERS_OPTIONS_ADDITIONAL_PARAMS{};
const std::array<unsigned int, 2> PARALLEL_PARAMS_DEFAULT{{NUM_THREADS_CPU, 1}};

//functions in RunAndEvaluateBpResults use above constants and function
#include "RunAndEvaluateBpResults.h"

int main(int argc, char** argv)
{
\s\sRunAndEvaluateBpResults::runBpOnStereoSets();
\s\sreturn 0;
}