/*
 * RunAndEvaluateBpResults.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNANDEVALUATEBPRESULTS_H_
#define RUNANDEVALUATEBPRESULTS_H_

#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include <memory>
#include <array>
#include "../FileProcessing/BpFileHandling.h"
#include "../ParameterFiles/bpRunSettings.h"

typedef std::filesystem::path filepathtype;

class RunAndEvaluateBpResults {
public:
	static void printParameters(std::ostream& resultsStream)
	{
		bool optLevel = USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT;
		resultsStream << "Stereo Set: " << bp_params::STEREO_SET << "\n";
		resultsStream << "Memory Optimization Level: " << optLevel << "\n";
		resultsStream << "Indexing Optimization Level: "
					<< OPTIMIZED_INDEXING_SETTING << "\n";
		resultsStream << "BP Processing Data Type: "
					<< BELIEF_PROP_PROCESSING_DATA_TYPE_STRING << "\n";
		resultsStream << "Num Possible Disparity Values: "
					<< bp_params::NUM_POSSIBLE_DISPARITY_VALUES << "\n";
		resultsStream << "Num BP Levels: " << bp_params::LEVELS_BP << "\n";
		resultsStream << "Num BP Iterations: " << bp_params::ITER_BP << "\n";
		resultsStream << "DISC_K_BP: " << bp_params::DISC_K_BP << "\n";
		resultsStream << "DATA_K_BP: " << bp_params::DATA_K_BP << "\n";
		resultsStream << "LAMBDA_BP: " << bp_params::LAMBDA_BP << "\n";
		resultsStream << "SIGMA_BP: " << bp_params::SIGMA_BP << "\n";
		resultsStream << "CPU_OPTIMIZATION_LEVEL: " << static_cast<int>(CPU_OPTIMIZATION_SETTING)
					<< "\n";
		resultsStream << "BYTES_ALIGN_MEMORY: " << bp_params::BYTES_ALIGN_MEMORY << "\n";
		resultsStream << "NUM_DATA_ALIGN_WIDTH: " << bp_params::NUM_DATA_ALIGN_WIDTH << "\n";
	}

	//compare resulting disparity map with a ground truth (or some other disparity map...)
	//this function takes as input the file names of a two disparity maps and the factor
	//that each disparity was scaled by in the generation of the disparity map image
	static void compareDispMaps(const DisparityMap<float>& outputDisparityMap, const DisparityMap<float>& groundTruthDisparityMap, std::ostream& resultsStream)
	{
		const OutputEvaluationResults<float> outputEvalResults =
				outputDisparityMap.getOuputComparison(groundTruthDisparityMap, OutputEvaluationParameters<float>());
		resultsStream << outputEvalResults;
	}

	//run the CUDA stereo implementation on the default reference and test images with the result saved to the default
	//saved disparity map file as defined in bpStereoCudaParameters.cuh
	//static void runStereoTwoImpsAndCompare(std::ostream& outStream, const std::array<RunBpStereoSet<beliefPropProcessingDataType>*, 2>& bpProcessingImps)
	static void runStereoTwoImpsAndCompare(std::ostream& outStream, const std::array<std::unique_ptr<RunBpStereoSet<beliefPropProcessingDataType>>, 2>& bpProcessingImps)
	{
		BpFileHandling bpFileSettings(bp_params::STEREO_SET);
		const filepathtype refImagePath{bpFileSettings.getRefImagePath()};
		const filepathtype testImagePath{bpFileSettings.getTestImagePath()};
		std::array<filepathtype, 2> output_disp;
		for (unsigned int i=0; i < 2u; i++) {
			output_disp[i] = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
		}

		const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};

		//load all the BP default settings as set in bpStereoCudaParameters.cuh
		const BPsettings algSettings;

		std::cout << "Running belief propagation on reference image " << refImagePath << " and test image " << testImagePath << " on " <<
				     bpProcessingImps[0]->getBpRunDescription() << " and " << bpProcessingImps[1]->getBpRunDescription() << std::endl;
		std::array<ProcessStereoSetOutput, 2> run_output;

		for (unsigned int i = 0; i < 2u; i++) {
			run_output[i] = bpProcessingImps[i]->operator()(refImagePath.string(), testImagePath.string(), algSettings, outStream);
			run_output[i].outDisparityMap.saveDisparityMap(output_disp[i].string(), bp_params::SCALE_BP);
			outStream << "Median " << bpProcessingImps[i]->getBpRunDescription() << " runtime (including transfer time): " <<
					     run_output[i].runTime << std::endl;
		}

		for (unsigned int i = 0; i < 2u; i++) {
			std::cout << "Output disparity map from run at " << output_disp[i] << std::endl;
		}

		DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::SCALE_BP);
		for (unsigned int i = 0; i < 2u; i++) {
			outStream << std::endl << bpProcessingImps[i]->getBpRunDescription() << " output vs. Ground Truth result:\n";
			compareDispMaps(run_output[i].outDisparityMap, groundTruthDisparityMap, outStream);
		}

		outStream << std::endl << bpProcessingImps[0]->getBpRunDescription() << " output vs. " << bpProcessingImps[1]->getBpRunDescription() << " result:\n";
		compareDispMaps(run_output[0].outDisparityMap, run_output[1].outDisparityMap, outStream);

		std::cout << "More info including input parameters, detailed timings, and output disparity maps comparison to ground truth in output.txt.\n";
	}
};

#endif /* RUNANDEVALUATEBPRESULTS_H_ */
