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
#include <numeric>
#include <algorithm>

typedef std::filesystem::path filepathtype;
const std::string OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};

namespace RunAndEvaluateBpResults {
	std::pair<std::map<std::string, std::string>, std::vector<std::string>> getResultsMappingFromFile(const std::string& fileName) {
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

	void printParameters(const unsigned int numStereoSet, std::ostream& resultsStream)
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
	void compareDispMaps(const DisparityMap<float>& outputDisparityMap, const DisparityMap<float>& groundTruthDisparityMap, std::ostream& resultsStream)
	{
		const OutputEvaluationResults<float> outputEvalResults =
				outputDisparityMap.getOuputComparison(groundTruthDisparityMap, OutputEvaluationParameters<float>());
		resultsStream << outputEvalResults;
	}

	//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
	//on the reference and test images specified by numStereoSet
	//run only optimized implementation if runOptImpOnly is true
	template<typename T, unsigned int DISP_VALS_OPTIMIZED, unsigned int DISP_VALS_SINGLE_THREAD>
		void runStereoTwoImpsAndCompare(std::ostream& outStream,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_OPTIMIZED>>& optimizedImp,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_SINGLE_THREAD>>& singleThreadCPUImp,
		const unsigned int numStereoSet, const beliefprop::BPsettings& algSettings,
		const beliefprop::ParallelParameters& parallelParams,
		bool runOptImpOnly = false)
	{
		const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
		printParameters(numStereoSet, outStream);
		if (DISP_VALS_OPTIMIZED == 0) {
			outStream << "DISP_VALS_TEMPLATED: " << "NO" << std::endl;
		}
		else {
			outStream << "DISP_VALS_TEMPLATED: " << "YES" << std::endl;
		}
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

		run_output[0] = optimizedImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, outStream, parallelParams);
		run_output[0].outDisparityMap.saveDisparityMap(output_disp[0].string(), bp_params::SCALE_BP[numStereoSet]);
		outStream << OPTIMIZED_RUNTIME_HEADER << ":" << run_output[0].runTime << std::endl;

        if (!runOptImpOnly) {
			run_output[1] = singleThreadCPUImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, outStream, parallelParams);
			run_output[1].outDisparityMap.saveDisparityMap(output_disp[1].string(), bp_params::SCALE_BP[numStereoSet]);
			outStream << OPTIMIZED_RUNTIME_HEADER << ":" << run_output[1].runTime << std::endl;
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

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, unsigned int DISP_VALS_TEMPLATE_SINGLE_THREAD>
	void runBpOnSetAndUpdateResults(std::array<std::map<std::string, std::vector<std::string>>, 2>& resDefParParamsFinal,
								const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_OPTIMIZED>>& optimizedImp,
								const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_SINGLE_THREAD>>& singleThreadCPUImp)
	{
		//load all the BP default settings
		beliefprop::BPsettings algSettings;
		algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

		//parallel parameters initialized with default thread count dimensions at every level
		beliefprop::ParallelParameters parallelParams(algSettings.numLevels_, PARALLEL_PARAMS_DEFAULT);

		//if optimizing parallel parameters, parallelParamsVect contains parallel parameter settings to run (and is empty if not)
		std::vector<std::array<unsigned int, 2>> parallelParamsVect{OPTIMIZE_PARALLEL_PARAMS ? PARALLEL_PARAMETERS_OPTIONS : std::vector<std::array<unsigned int, 2>>()};
		
		//CUDA implementation only (for now):
		//add additional parallel parameters if first stereo set, second stereo set but not double, or not using templated disparity
		//otherwise the additional parallel parameters with more than 256 can fail to launch due to resource limitations (likely related to registers)
		if (((NUM_SET == 0) || ((NUM_SET == 1) && (typeid(T) != typeid(double)))) || (DISP_VALS_TEMPLATE_OPTIMIZED == 0)) {
			parallelParamsVect.insert(parallelParamsVect.end(), PARALLEL_PARAMETERS_OPTIONS_ADDITIONAL_PARAMS.begin(), PARALLEL_PARAMETERS_OPTIONS_ADDITIONAL_PARAMS.end());
		}
		
		//mapping of parallel parameters to runtime for each kernel at each level and total runtime
		std::array<std::vector<std::map<std::array<unsigned int, 2>, double>>, (beliefprop::NUM_KERNELS + 1)> pParamsToRunTimeEachKernel;
		for (unsigned int i=0; i < beliefprop::NUM_KERNELS; i++) {
			//set to vector length for each kernel to corresponding vector length of kernel in parallelParams.parallelDimsEachKernel_
			pParamsToRunTimeEachKernel[i] = std::vector<std::map<std::array<unsigned int, 2>, double>>(parallelParams.parallelDimsEachKernel_[i].size()); 
		}
		pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS] = std::vector<std::map<std::array<unsigned int, 2>, double>>(1); 
		
		//if optimizing parallel parameters, run BP for each parallel parameters option, retrieve best parameters for each kernel or overall for the run,
		//and then run BP with best found parallel parameters
		//if not optimizing parallel parameters, run BP once using default parallel parameters
		for (unsigned int runNum=0; runNum <= parallelParamsVect.size(); runNum++) {
			const std::array<unsigned int, 2>* pParamsCurrRun = (runNum < parallelParamsVect.size()) ? (&(parallelParamsVect[runNum])) : nullptr;
			std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);
			if (runNum < parallelParamsVect.size()) {
				//set parallel parameters to current parameters for each BP processing level
				parallelParams.setParallelDims(*pParamsCurrRun, algSettings.numLevels_);
			}

			resultsStream << "DataType:" << DATA_SIZE_TO_NAME_MAP.at(sizeof(T)) << std::endl;
			if constexpr (OPTIMIZE_PARALLEL_PARAMS && (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN)) {
				//show parallel parameters for each kernel if allowing different parallel parameters for each kernel in the same run
				resultsStream << "Blur Images Parallel Dimensions:" << 
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][0] << " x " <<
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::BLUR_IMAGES][0][1] << std::endl;
				resultsStream << "Init Message Values Parallel Dimensions:" << 
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][0] << " x " <<
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][1] << std::endl;
				for (unsigned int level=0; level < algSettings.numLevels_; level++) {
					resultsStream << "Level " << std::to_string(level) << " Data Costs Parallel Dimensions:" << 
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][0] << " x " <<
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][1] << std::endl;
				}
				for (unsigned int level=0; level < algSettings.numLevels_; level++) {
					resultsStream << "Level " << std::to_string(level) << " BP Thread Parallel Dimensions:" << 
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][level][0] << " x " <<
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::BP_AT_LEVEL][level][1] << std::endl;
				}
				for (unsigned int level=0; level < algSettings.numLevels_; level++) {
					resultsStream << "Level " << std::to_string(level) << " Copy Thread Parallel Dimensions:" << 
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][level][0] << " x " <<
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::COPY_AT_LEVEL][level][1] << std::endl;
				}
				resultsStream << "Get Output Disparity Parallel Dimensions:" << 
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][0] << " x " <<
								parallelParams.parallelDimsEachKernel_[beliefprop::BpKernel::OUTPUT_DISP][0][1] << std::endl;
			}

			//run optimized implementation only (and not single-threaded CPU implementation) if not final run or run is using default parameter parameters
			//final run and run using default parallel parameters are only runs that are output in final results
			const bool runOptImpOnly{(!((runNum == parallelParamsVect.size()) || ((*pParamsCurrRun) == PARALLEL_PARAMS_DEFAULT)))};
			runStereoTwoImpsAndCompare<T, DISP_VALS_TEMPLATE_OPTIMIZED, DISP_VALS_TEMPLATE_SINGLE_THREAD>(
				resultsStream, optimizedImp, singleThreadCPUImp, NUM_SET, algSettings, parallelParams, runOptImpOnly);
			resultsStream.close();

			//get results including runtimes for each kernel (if allowing different parallel parameters for each kernel) and
			//total runtime for current run
			const auto resultsCurrentRun = getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).first;
			if (runNum < parallelParamsVect.size()) {
				if constexpr (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
					for (unsigned int level=0; level < algSettings.numLevels_; level++) {
						pParamsToRunTimeEachKernel[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][*pParamsCurrRun] =
							std::stod(resultsCurrentRun.at("Level " + std::to_string(level) + " Data Costs (" +
									std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
						pParamsToRunTimeEachKernel[beliefprop::BpKernel::BP_AT_LEVEL][level][*pParamsCurrRun] = 
							std::stod(resultsCurrentRun.at("Level " + std::to_string(level) + " BP Runtime (" + 
									std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
						pParamsToRunTimeEachKernel[beliefprop::BpKernel::COPY_AT_LEVEL][level][*pParamsCurrRun] =
							std::stod(resultsCurrentRun.at("Level " + std::to_string(level) + " Copy Runtime (" + 
														std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
					}
					pParamsToRunTimeEachKernel[beliefprop::BpKernel::BLUR_IMAGES][0][*pParamsCurrRun] =
							std::stod(resultsCurrentRun.at("Smoothing Runtime (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
					pParamsToRunTimeEachKernel[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][*pParamsCurrRun] =
							std::stod(resultsCurrentRun.at("Time to init message values (kernel portion only) (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
					pParamsToRunTimeEachKernel[beliefprop::BpKernel::OUTPUT_DISP][0][*pParamsCurrRun] =
							std::stod(resultsCurrentRun.at("Time get output disparity (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
				}
				//get total runtime
				pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0][*pParamsCurrRun] =
						std::stod(resultsCurrentRun.at(OPTIMIZED_RUNTIME_HEADER));

				if (runNum == (parallelParamsVect.size() - 1)) {
					if constexpr (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
						for (unsigned int numKernelSet = 0; numKernelSet < pParamsToRunTimeEachKernel.size(); numKernelSet++) {
							//retrieve and set optimized parallel parameters for final run
							//std::min_element used to retrieve parallel parameters corresponding to lowest runtime from previous runs
							std::transform(pParamsToRunTimeEachKernel[numKernelSet].begin(),
										pParamsToRunTimeEachKernel[numKernelSet].end(), 
										parallelParams.parallelDimsEachKernel_[numKernelSet].begin(),
										[](const auto& tDimsToRunTimeCurrLevel) -> std::array<unsigned int, 2> { 
											return (std::min_element(tDimsToRunTimeCurrLevel.begin(), tDimsToRunTimeCurrLevel.end(),
													[](const auto& a, const auto& b) { return a.second < b.second; }))->first; });
						}
					}
					else {
						//set optimized parallel parameters for all kernels to parallel parameters that got the best runtime across all kernels
						//seems like setting different parallel parameters for different kernels on GPU decrease runtime but increases runtime on CPU
						const auto bestParallelParams = std::min_element(pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0].begin(),
																		pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0].end(),
																		[](const auto& a, const auto& b) { return a.second < b.second; })->first;
						parallelParams.setParallelDims(bestParallelParams, algSettings.numLevels_);
					}
				}
			}
			if ((runNum == parallelParamsVect.size()) || ((*pParamsCurrRun) == PARALLEL_PARAMS_DEFAULT))
			{
				//set output for runs using default parallel parameters and final run (which is the same run if not optimizing parallel parameters)
				auto& resultUpdate = (runNum == parallelParamsVect.size()) ? resDefParParamsFinal[1] : resDefParParamsFinal[0];
				for (const auto& currRunResult : resultsCurrentRun) {
					if (resultUpdate.count(currRunResult.first)) {
						resultUpdate[currRunResult.first].push_back(currRunResult.second);
					}
					else {
						resultUpdate[currRunResult.first] = std::vector{currRunResult.second};
					}
				}
			}
		}
	}

	//get average and median speedup using optimized parallel parameters compared to default parallel parameters
	void getAverageMedianSpeedup(std::array<std::map<std::string, std::vector<std::string>>, 2>& resDefParParamsFinal) {
		std::vector<double> speedupsVect;
		resDefParParamsFinal[1]["Speedup Over Default Parallel Parameters"] = std::vector<std::string>();
		for (unsigned int i=0; i < resDefParParamsFinal[0].at(OPTIMIZED_RUNTIME_HEADER).size(); i++) {
			speedupsVect.push_back(std::stod(resDefParParamsFinal[0].at(OPTIMIZED_RUNTIME_HEADER).at(i)) / 
								std::stod(resDefParParamsFinal[1].at(OPTIMIZED_RUNTIME_HEADER).at(i)));
			resDefParParamsFinal[1]["Speedup Over Default Parallel Parameters"].push_back(std::to_string(speedupsVect.back()));
		}
		std::sort(speedupsVect.begin(), speedupsVect.end());
		std::cout << "Average speedup: " << 
			(std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)resDefParParamsFinal[0].at(OPTIMIZED_RUNTIME_HEADER).size()) << std::endl;
		const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
			(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
			speedupsVect[(speedupsVect.size() / 2)];
		std::cout << "Median speedup: " << medianSpeedup << std::endl;
	}
};

#endif /* RUNANDEVALUATEBPRESULTS_H_ */
