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

namespace bp_file_handling
{
	const std::string REF_IMAGE_NAME = "refImage";
	const std::string TEST_IMAGE_NAME = "testImage";
	const std::string IN_IMAGE_POSS_EXTENSIONS[] = {"pgm", "ppm"};
	const std::string GROUND_TRUTH_DISP_FILE = "groundTruthDisparity.pgm";
	const std::string OUT_DISP_IMAGE_NAME_BASE = "computedDisparity";

#ifdef _WIN32
	//SOLUTION_DIR is set to $(SolutionDir) in preprocessor of Visual Studio Project
	const std::string EXE_PATH_PATH = SOLUTION_DIR;
	const std::string STEREO_SETS_PATH = EXE_PATH_PATH + "/StereoSets";
#else
	//assuming that executable is created in main cudaBeliefProp directory when using g++
	const std::filesystem::path EXE_PATH_PATH = std::filesystem::current_path();
	const std::filesystem::path STEREO_SETS_PATH = EXE_PATH_PATH / "StereoSets";
#endif //_WIN32
}

#endif /* BPFILEHANDLINGCONSTS_H_ */
