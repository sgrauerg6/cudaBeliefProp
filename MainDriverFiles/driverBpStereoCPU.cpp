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

#include "../SingleThreadCPU/stereo.h"

//needed to run the optimized implementation a stereo set using CPU
#include "../OptimizeCPU/RunBpStereoOptimizedCPU.h"

#include "RunAndEvaluateBpResults.h"
#include <memory>
#include <fstream>

int main(int argc, char** argv)
{
	std::ofstream resultsStream("output.txt", std::ofstream::out);
	//std::ostream resultsStream(std::cout.rdbuf());

	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>, 2> bpProcess_stereoSet0_float = {
			std::make_unique<RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>()
	};

	resultsStream << "DataType: FLOAT" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(resultsStream, bpProcess_stereoSet0_float, 0);
	resultsStream << std::endl << std::endl;

	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>, 2> bpProcess_stereoSet1_float = {
			std::make_unique<RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>()
	};

	resultsStream << "DataType: FLOAT" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(resultsStream, bpProcess_stereoSet1_float, 1);
	resultsStream << std::endl << std::endl;

	std::array<std::unique_ptr<RunBpStereoSet<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>, 2> bpProcess_stereoSet2_float = {
			std::make_unique<RunBpStereoOptimizedCPU<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>()
	};

	resultsStream << "DataType: FLOAT" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<float, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(resultsStream, bpProcess_stereoSet2_float, 2);

	std::array<std::unique_ptr<RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>, 2> bpProcess_stereoSet0_half = {
			std::make_unique<RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>>()
	};

	resultsStream << "DataType: HALF" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[0]>(resultsStream, bpProcess_stereoSet0_half, 0);
	resultsStream << std::endl << std::endl;

	std::array<std::unique_ptr<RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>, 2> bpProcess_stereoSet1_half = {
			std::make_unique<RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>>()
	};

	resultsStream << "DataType: HALF" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[1]>(resultsStream, bpProcess_stereoSet1_half, 1);
	resultsStream << std::endl << std::endl;

	std::array<std::unique_ptr<RunBpStereoSet<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>, 2> bpProcess_stereoSet2_half = {
			std::make_unique<RunBpStereoOptimizedCPU<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>(),
			std::make_unique<RunBpStereoCPUSingleThread<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>>()
	};

	resultsStream << "DataType: HALF" << std::endl;
	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare<short, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[2]>(resultsStream, bpProcess_stereoSet2_half, 2);
}
