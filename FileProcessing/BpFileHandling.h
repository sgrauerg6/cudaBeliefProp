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

class BpFileHandling {
public:
	BpFileHandling(const std::string& stereo_set_name) : stereo_set_path_(std::filesystem::path(bp_file_handling::STEREO_SETS_PATH) / stereo_set_name), num_out_disp_map_(1) {}
	virtual ~BpFileHandling();

	std::filesystem::path getRefImagePath()
	{
		for (const auto& extension : bp_file_handling::IN_IMAGE_POSS_EXTENSIONS)
		{
			if (std::filesystem::exists((stereo_set_path_ / (bp_file_handling::REF_IMAGE_NAME + "." + extension))))
			{
				return stereo_set_path_ / (bp_file_handling::REF_IMAGE_NAME + "." + extension);
			}
		}

		return std::filesystem::path();
	}

	std::filesystem::path getTestImagePath()
	{
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

		return std::filesystem::path();
	}

	std::filesystem::path getCurrentOutputDisparityFilePathAndIncrement()
	{
		return stereo_set_path_ / (bp_file_handling::OUT_DISP_IMAGE_NAME_BASE + std::to_string(num_out_disp_map_++) + ".pgm");

	}

	std::filesystem::path getGroundTruthDisparityFilePath()
	{
		return stereo_set_path_ / (bp_file_handling::GROUND_TRUTH_DISP_FILE);
	}

private:
	std::filesystem::path stereo_set_path_;
	unsigned int num_out_disp_map_;

};

#endif /* BPFILEHANDLING_H_ */
