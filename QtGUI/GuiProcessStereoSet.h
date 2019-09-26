#pragma once

#include <utility>
#include <string>

//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "bpStereoParameters.h"

//needed to run the each of the bp implementations
#include "SingleThreadCPU/stereo.h"
#include "OptimizeCPU/RunBpStereoOptimizedCPU.h"
#include "OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"

//needed to set number of threads for OpenMP
#include <omp.h>

enum bpImplementation { NAIVE_CPU, OPTIMIZED_CPU, OPTIMIZED_CUDA };

class GuiProcessStereoSet
{
public:
	static BPsettings initializeAndReturnBPSettings()
	{
		BPsettings startBPSettings;

		startBPSettings.smoothingSigma = SIGMA_BP;
		startBPSettings.numLevels = LEVELS_BP;
		startBPSettings.numIterations = ITER_BP;
		startBPSettings.lambda_bp = LAMBDA_BP;
		startBPSettings.data_k_bp = DATA_K_BP;
		startBPSettings.disc_k_bp = DISC_K_BP;

		//height/width determined when image read from file
		startBPSettings.widthImages = 0;
		startBPSettings.heightImages = 0;

		return startBPSettings;
	}

	//process stereo set using input implementationToRun; return pair with runtime first
	//and file path of computed disparity map second
	static std::pair<float, std::string> processStereoSet(bpImplementation implementationToRun)
	{
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

		float runTime = (*runBp)(DEFAULT_REF_IMAGE_PATH, DEFAULT_TEST_IMAGE_PATH, initializeAndReturnBPSettings(), SAVE_DISPARITY_IMAGE_PATH_1, resultsFile);
		delete runBp;

		//initialize and return output pair of implementation runtime and file path of output disparity map
		return std::pair<float, std::string>(runTime, SAVE_DISPARITY_IMAGE_PATH_1);
	}
};