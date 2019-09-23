/*
 * BpFileHandlingConsts.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLINGCONSTS_H_
#define BPFILEHANDLINGCONSTS_H_
#include <filesystem>

namespace bp_file_handling
{
	const std::string REF_IMAGE_NAME = "refImage";
	const std::string TEST_IMAGE_NAME = "testImage";
	const std::string IN_IMAGE_POSS_EXTENSIONS[] = {"pgm", "ppm"};
	const std::string GROUND_TRUTH_DISP_FILE = "groundTruthDisparity.pgm";
	const std::string OUT_DISP_IMAGE_NAME_BASE = "computedDisparity";

	const std::string EXE_PATH_PATH = "/home/scott/cudaBeliefProp";
	const std::string STEREO_SETS_PATH = EXE_PATH_PATH + "/StereoSets";
}





#endif /* BPFILEHANDLINGCONSTS_H_ */
