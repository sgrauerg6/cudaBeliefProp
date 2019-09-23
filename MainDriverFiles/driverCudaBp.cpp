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

//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "ParameterFiles/bpStereoCudaParameters.h"
#include "SingleThreadCPU/stereo.h"

//needed to run the implementation a stereo set using CUDA
#include "OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
#include "RunAndEvaluateBpResults.h"
#include <memory>
#include <fstream>

void retrieveDeviceProperties(int numDevice, std::ostream& resultsStream)
{
	cudaDeviceProp prop;
	cudaGetDeviceProperties( &prop, numDevice);
	int cudaDriverVersion;
	cudaDriverGetVersion(&cudaDriverVersion);

	resultsStream << "Device " << numDevice << ": " << prop.name << " with " << prop.multiProcessorCount << " multiprocessors\n";
	resultsStream << "Cuda version: " << cudaDriverVersion << "\n";
}


int main(int argc, char** argv)
{
	//std::ofstream resultsStream("output.txt", std::ofstream::out);
	std::ostream resultsStream(std::cout.rdbuf());

	RunAndEvaluateBpResults::printParameters(resultsStream);
	resultsStream << "USE_SHARED_MEMORY: " << USE_SHARED_MEMORY << "\n";
	resultsStream << "DISP_INDEX_START_REG_LOCAL_MEM: " << DISP_INDEX_START_REG_LOCAL_MEM << "\n";
	retrieveDeviceProperties(0, resultsStream);

	int cudaRuntimeVersion;
	cudaRuntimeGetVersion(&cudaRuntimeVersion);
	resultsStream << "Cuda Runtime Version: " << cudaRuntimeVersion << "\n";

	std::array<std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>, 2> bpProcessingImps = {
				std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>(new RunBpStereoSetOnGPUWithCUDA<beliefPropProcessingDataType>()),
				std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>(new RunBpStereoCPUSingleThread<beliefPropProcessingDataType>())
	};

	RunAndEvaluateBpResults::runStereoTwoImpsAndCompare(resultsStream, bpProcessingImps);
}
