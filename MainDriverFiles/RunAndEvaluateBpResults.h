/*
 * RunAndEvaluateBpResults.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNANDEVALUATEBPRESULTS_H_
#define RUNANDEVALUATEBPRESULTS_H_

#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
//#include "../OptimizeCPU/RunBpStereoOptimizedCPU.h"
#include <memory>
#include <array>
#include <fstream>
#include <map>
#include <vector>
#include "../FileProcessing/BpFileHandling.h"
#include "../ParameterFiles/bpRunSettings.h"

typedef std::filesystem::path filepathtype;

class RunAndEvaluateBpResults {
public:
	static std::pair<std::map<std::string, std::string>, std::vector<std::string>> getResultsMappingFromFile(const std::string& fileName) {
		std::map<std::string, std::string> resultsMapping;
		std::vector<std::string> headersInOrder;
		std::ifstream resultsFile(fileName);

		std::string line;
		constexpr char delim{':'};
		while (std::getline(resultsFile, line))
		{
		    //get "header" and corresponding result that are divided by ":"
			std::stringstream ss(line);
			std::string header, result;
			std::getline(ss, header, delim);
			std::getline(ss, result, delim);
			if (header.size() > 0) {
				unsigned int i{0u};
				const std::string origHeader{header};
				while (resultsMapping.count(header) > 0) {
					i++;
					header = origHeader + "_" + std::to_string(i);
				}
				resultsMapping[header] = result;
				headersInOrder.push_back(header);
			}
		}

		return {resultsMapping, headersInOrder};
	}

	static void printParameters(const unsigned int numStereoSet, std::ostream& resultsStream)
	{
		bool optLevel = USE_OPTIMIZED_GPU_MEMORY_MANAGEMENT;
		resultsStream << "Stereo Set: " << bp_params::STEREO_SET[numStereoSet] << "\n";
		resultsStream << "Memory Optimization Level: " << optLevel << "\n";
		resultsStream << "Indexing Optimization Level: "
					<< OPTIMIZED_INDEXING_SETTING << "\n";
		//resultsStream << "BP Processing Data Type: "
		//			<< BELIEF_PROP_PROCESSING_DATA_TYPE_STRING << "\n";
		resultsStream << "Num Possible Disparity Values: "
					<< bp_params::NUM_POSSIBLE_DISPARITY_VALUES[numStereoSet] << "\n";
		resultsStream << "Num BP Levels: " << bp_params::LEVELS_BP << "\n";
		resultsStream << "Num BP Iterations: " << bp_params::ITER_BP << "\n";
		resultsStream << "DISC_K_BP: " << bp_params::DISC_K_BP[numStereoSet] << "\n";
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
	template<typename T, unsigned int DISP_VALS>
	static void runStereoTwoImpsAndCompare(std::ostream& outStream,
			const std::array<std::unique_ptr<RunBpStereoSet<T, DISP_VALS>>, 2>& bpProcessingImps,
			const unsigned int numStereoSet, const beliefprop::BPsettings& algSettings, bool runOptImpOnly = false)
	{
		const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
		printParameters(numStereoSet, outStream);
		outStream << "DISP_VALS_TEMPLATED: " << "YES" << std::endl;
		BpFileHandling bpFileSettings(bp_params::STEREO_SET[numStereoSet]);
		const std::array<filepathtype, 2> refTestImagePath{bpFileSettings.getRefImagePath(), bpFileSettings.getTestImagePath()};
		std::array<filepathtype, 2> output_disp;
		for (unsigned int i=0; i < numImpsRun; i++) {
			output_disp[i] = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
		}

		std::cout << "Running belief propagation on reference image " << refTestImagePath[0] << " and test image " << refTestImagePath[1] << " on " <<
				     bpProcessingImps[0]->getBpRunDescription();
		if (!runOptImpOnly) {
			std::cout << " and " << bpProcessingImps[1]->getBpRunDescription();
		}
		std::cout << std::endl;

		std::array<ProcessStereoSetOutput, 2> run_output;
		for (unsigned int i = 0; i < numImpsRun; i++) {
			run_output[i] = bpProcessingImps[i]->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, outStream);
			run_output[i].outDisparityMap.saveDisparityMap(output_disp[i].string(), bp_params::SCALE_BP[numStereoSet]);
			outStream << "Median " << bpProcessingImps[i]->getBpRunDescription() << " runtime (including transfer time): " <<
					     run_output[i].runTime << std::endl;
		}

		for (unsigned int i = 0; i < numImpsRun; i++) {
			std::cout << "Output disparity map from " << bpProcessingImps[i]->getBpRunDescription() << " run at " << output_disp[i] << std::endl;
		}
		std::cout << std::endl;

		const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};
		DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::SCALE_BP[numStereoSet]);
		for (unsigned int i = 0; i < numImpsRun; i++) {
			outStream << std::endl << bpProcessingImps[i]->getBpRunDescription() << " output vs. Ground Truth result:\n";
			compareDispMaps(run_output[i].outDisparityMap, groundTruthDisparityMap, outStream);
		}

	    if (!runOptImpOnly) {
			outStream << std::endl << bpProcessingImps[0]->getBpRunDescription() << " output vs. " << bpProcessingImps[1]->getBpRunDescription() << " result:\n";
			compareDispMaps(run_output[0].outDisparityMap, run_output[1].outDisparityMap, outStream);
		}
	}

	template<typename T, unsigned int DISP_VALS>
		static void runStereoTwoImpsAndCompare(std::ostream& outStream,
		const std::unique_ptr<RunBpStereoSet<T, 0>>& optimizedImp,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS>>& singleThreadCPUImp,
		const unsigned int numStereoSet, const beliefprop::BPsettings& algSettings,
		bool runOptImpOnly = false)
	{
		const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
		printParameters(numStereoSet, outStream);
		outStream << "DISP_VALS_TEMPLATED: " << "NO" << std::endl;
		BpFileHandling bpFileSettings(bp_params::STEREO_SET[numStereoSet]);
		const std::array<filepathtype, 2> refTestImagePath{bpFileSettings.getRefImagePath(), bpFileSettings.getTestImagePath()};
		std::array<filepathtype, 2> output_disp;
		for (unsigned int i=0; i < numImpsRun; i++) {
			output_disp[i] = bpFileSettings.getCurrentOutputDisparityFilePathAndIncrement();
		}

		std::cout << "Running belief propagation on reference image " << refTestImagePath[0] << " and test image " << refTestImagePath[1] << " on " <<
				optimizedImp->getBpRunDescription();
		if (!runOptImpOnly) {
			std::cout << " and " << singleThreadCPUImp->getBpRunDescription();
		}
		std::cout << std::endl;
		std::array<ProcessStereoSetOutput, 2> run_output;

		run_output[0] = optimizedImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, outStream);
		run_output[0].outDisparityMap.saveDisparityMap(output_disp[0].string(), bp_params::SCALE_BP[numStereoSet]);
		outStream << "Median " << optimizedImp->getBpRunDescription() << " runtime (including transfer time): " <<
				run_output[0].runTime << std::endl;

        if (!runOptImpOnly) {
			run_output[1] = singleThreadCPUImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, outStream);
			run_output[1].outDisparityMap.saveDisparityMap(output_disp[1].string(), bp_params::SCALE_BP[numStereoSet]);
			outStream << "Median " << singleThreadCPUImp->getBpRunDescription() << " runtime (including transfer time): " <<
					run_output[1].runTime << std::endl;
		}

		for (unsigned int i = 0; i < numImpsRun; i++) {
			const std::string runDesc{(i == 0) ? optimizedImp->getBpRunDescription() : singleThreadCPUImp->getBpRunDescription()};
			std::cout << "Output disparity map from " << runDesc << " run at " << output_disp[i] << std::endl;
		}
		std::cout << std::endl;

		const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};
		DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::SCALE_BP[numStereoSet]);
		outStream << std::endl << optimizedImp->getBpRunDescription() << " output vs. Ground Truth result:\n";
		compareDispMaps(run_output[0].outDisparityMap, groundTruthDisparityMap, outStream);
        if (!runOptImpOnly) {
			outStream << std::endl << singleThreadCPUImp->getBpRunDescription() << " output vs. Ground Truth result:\n";
			compareDispMaps(run_output[1].outDisparityMap, groundTruthDisparityMap, outStream);

			outStream << std::endl << optimizedImp->getBpRunDescription() << " output vs. " << singleThreadCPUImp->getBpRunDescription() << " result:\n";
			compareDispMaps(run_output[0].outDisparityMap, run_output[1].outDisparityMap, outStream);
		}
	}
};

#endif /* RUNANDEVALUATEBPRESULTS_H_ */
