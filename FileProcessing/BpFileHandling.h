/*
 * BpFileHandling.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef BPFILEHANDLING_H_
#define BPFILEHANDLING_H_

#include <filesystem>
#include "BpFileHandlingConsts.h"
#include <iostream>

class BpFileHandling {
public:

	//constructor takes stereo set name as input, which must match the directory name of the stereo set
	BpFileHandling(const std::string& stereo_set_name) : stereo_set_path_(std::filesystem::path(bp_file_handling::STEREO_SETS_PATH) / stereo_set_name), num_out_disp_map_(1)
	{ }
	//virtual ~BpFileHandling();

	//return path to reference image with valid extension if found, otherwise returns empty path
	const std::filesystem::path getRefImagePath()
	{
		//check if ref image exists for each possible extension (currently pgm and ppm) and return path if so
		for (const auto& extension : bp_file_handling::IN_IMAGE_POSS_EXTENSIONS)
		{
			if (std::filesystem::exists((stereo_set_path_ / (bp_file_handling::REF_IMAGE_NAME + "." + extension))))
			{
				return stereo_set_path_ / (bp_file_handling::REF_IMAGE_NAME + "." + extension);
			}
		}

		throw std::filesystem::filesystem_error("Reference image not found", std::error_code());
	}

	//return path to test image with valid extension if found, otherwise returns empty path
	const std::filesystem::path getTestImagePath()
	{
		//check if test image exists for each possible extension (currently pgm and ppm) and return path if so
		for (const auto& extension : bp_file_handling::IN_IMAGE_POSS_EXTENSIONS)
		{
			if (std::filesystem::exists(
					(stereo_set_path_
							/ (bp_file_handling::TEST_IMAGE_NAME + "."
									+ extension))))
			{
				return stereo_set_path_
						/ (bp_file_handling::TEST_IMAGE_NAME + "." + extension);
			}
		}

		throw std::filesystem::filesystem_error("Test image not found", std::error_code());
	}

	//return path to use for current output disparity and then increment (to support multiple computed output disparity maps)
	const std::filesystem::path getCurrentOutputDisparityFilePathAndIncrement()
	{
		return stereo_set_path_ / (bp_file_handling::OUT_DISP_IMAGE_NAME_BASE + std::to_string(num_out_disp_map_++) + ".pgm");

	}

	//get file path to ground truth disparity map
	const std::filesystem::path getGroundTruthDisparityFilePath()
	{
		return stereo_set_path_ / (bp_file_handling::GROUND_TRUTH_DISP_FILE);
	}

private:
	std::filesystem::path stereo_set_path_;
	unsigned int num_out_disp_map_;
};

#endif /* BPFILEHANDLING_H_ */
