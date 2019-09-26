#pragma once

#include <utility>
#include <string>

//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "./ParameterFiles/bpStereoParameters.h"
#include "./ParameterFiles/bpRunSettings.h"
#include "./ParameterFiles/bpStructsAndEnums.h"

//needed to run the each of the bp implementations
#include "SingleThreadCPU/stereo.h"
#include "OptimizeCPU/RunBpStereoOptimizedCPU.h"
#include "OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
#include "./FileProcessing/BpFileHandling.h"
#include "./ParameterFiles/bpRunSettings.h"
#include "./BpAndSmoothProcessing/RunBpStereoSet.h"


#ifdef USE_FILESYSTEM
#include <filesystem>
typedef std::filesystem::path filepathtype;
#else
typedef std::string filepathtype;
#endif //USE_FILESYSTEM

//needed to set number of threads for OpenMP
#include <omp.h>

enum bpImplementation { NAIVE_CPU, OPTIMIZED_CPU, OPTIMIZED_CUDA };

class GuiProcessStereoSet
{
public:
	static BPsettings initializeAndReturnBPSettings()
	{
		BPsettings startBPSettings;

		return startBPSettings;
	}

	//process stereo set using input implementationToRun; return pair with runtime first
	//and file path of computed disparity map second
	static std::pair<float, std::string> processStereoSet(bpImplementation implementationToRun)
	{
		std::ofstream resultsStream("output.txt", std::ofstream::out);
		//std::ostream resultsStream(std::cout.rdbuf());

		BpFileHandling bpFileSettings(bp_params::STEREO_SET);
		filepathtype refImagePath = bpFileSettings.getRefImagePath();
		filepathtype testImagePath = bpFileSettings.getTestImagePath();
		filepathtype outputDisparityMapFile = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();

		FILE* resultsFile = fopen("output.txt", "w");
		RunBpStereoSet<float>* runBp;
		if (implementationToRun == bpImplementation::NAIVE_CPU)
		{
			runBp = new RunBpStereoCPUSingleThread<float>();
		}
		else if (implementationToRun == bpImplementation::OPTIMIZED_CPU)
		{
			runBp = new RunBpStereoOptimizedCPU<float>();
			omp_set_dynamic(1);
			omp_set_num_threads(4);
		}
		else
		{
			runBp = new RunBpStereoSetOnGPUWithCUDA<float>();
		}

		ProcessStereoSetOutput processStereoOutput = (*runBp)(refImagePath, testImagePath, initializeAndReturnBPSettings(), resultsStream);
		processStereoOutput.outDisparityMap.saveDisparityMap(outputDisparityMapFile, bp_params::SCALE_BP);
		delete runBp;

		//initialize and return output pair of implementation runtime and file path of output disparity map
		return std::pair<float, std::string>(processStereoOutput.runTime, outputDisparityMapFile);
	}
};