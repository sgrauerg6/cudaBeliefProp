/*
 * BpFileHandlingConsts.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLINGCONSTS_H_
#define BPFILEHANDLINGCONSTS_H_
#include <filesystem>
#include <string>

#define USE_FILESYSTEM

namespace bp_file_handling
{
	const std::string REF_IMAGE_NAME = "refImage";
	const std::string TEST_IMAGE_NAME = "testImage";
	const std::string IN_IMAGE_POSS_EXTENSIONS[] = {"pgm", "ppm"};
	const std::string GROUND_TRUTH_DISP_FILE = "groundTruthDisparity.pgm";
	const std::string OUT_DISP_IMAGE_NAME_BASE = "computedDisparity";

#ifdef USE_FILESYSTEM
#ifdef _WIN32
	//const std::string EXE_PATH_PATH = "C:/Users/sgrau/Documents/GitHub/cudaBeliefProp";
	//SOLUTION_DIR is defined in preprocessor of Visual Studio Project
	const std::string EXE_PATH_PATH = SOLUTION_DIR;// "C:/Users/sgrau/source/repos/beliefProp";
	const std::string STEREO_SETS_PATH = EXE_PATH_PATH + "/StereoSets";
#else
	const std::filesystem::path EXE_PATH_PATH = std::filesystem::current_path();
	const std::filesystem::path STEREO_SETS_PATH = EXE_PATH_PATH / "StereoSets";
#endif //USE_FILESYSTEM
#else
#ifdef _WIN32
	//const std::string EXE_PATH_PATH = "C:/Users/sgrau/Documents/GitHub/cudaBeliefProp";
	//SOLUTION_DIR is defined in preprocessor of Visual Studio Project
	const std::string EXE_PATH_PATH = SOLUTION_DIR;// "C:/Users/sgrau/source/repos/beliefProp";
	const std::string STEREO_SETS_PATH = EXE_PATH_PATH + "/StereoSets";
#else
	const std::string EXE_PATH_PATH = "/home/scott/cudaBeliefProp";
	const std::string STEREO_SETS_PATH = EXE_PATH_PATH + "/StereoSets";
#endif
#endif //USE_FILESYSTEM
}


#endif /* BPFILEHANDLINGCONSTS_H_ */
