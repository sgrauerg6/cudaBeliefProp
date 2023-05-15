/*
 * RunAndEvaluateBpResults.h
 *
 *  Created on: Sep 23, 2019
 *      Author: scott
 */

#ifndef RUNANDEVALUATEBPRESULTS_H_
#define RUNANDEVALUATEBPRESULTS_H_

#include <memory>
#include <array>
#include <fstream>
#include <map>
#include <vector>
#include <numeric>
#include <algorithm>
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../FileProcessing/BpFileHandling.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../SingleThreadCPU/stereo.h"

typedef std::filesystem::path filepathtype;

//check if optimized CPU run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CPU_RUN
//needed to run the optimized implementation a stereo set using CPU
#include "../OptimizeCPU/RunBpStereoOptimizedCPU.h"
//set RunBpOptimized alias to correspond to optimized CPU implementation
template <typename T, unsigned int DISP_VALS>
using RunBpOptimized = RunBpStereoOptimizedCPU<T, DISP_VALS>;
//set data type used for half-precision
#ifdef COMPILING_FOR_ARM
using halftype = float16_t;
#else
using halftype = short;
#endif //COMPILING_FOR_ARM
#endif //OPTIMIZED_CPU_RUN

//check if CUDA run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CUDA_RUN
//needed for the current BP parameters for the costs and also the CUDA parameters such as thread block size
#include "../ParameterFiles/bpStereoCudaParameters.h"
//needed to run the implementation a stereo set using CUDA
#include "../OptimizeCUDA/RunBpStereoSetOnGPUWithCUDA.h"
//set RunBpOptimized alias to correspond to CUDA implementation
template <typename T, unsigned int DISP_VALS>
using RunBpOptimized = RunBpStereoSetOnGPUWithCUDA<T, DISP_VALS>;
//set data type used for half-precision with CUDA
using halftype = half;
#endif //OPTIMIZED_CUDA_RUN

namespace RunAndEvaluateBpResults {
	//constants for output results for individual and sets of runs
	const std::string BP_RUN_OUTPUT_FILE{"output.txt"};
	const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{"outputResults.csv"};
	const std::string BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE{"outputResultsDefaultParallelParams.csv"};
	const std::string OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};
	std::vector<std::string> headersInOrder;

	std::pair<std::map<std::string, std::string>, std::vector<std::string>> getResultsMappingFromFile(const std::string& fileName) {
		std::map<std::string, std::string> resultsMapping;
		std::vector<std::string> currResultHeadersInOrder;
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
				currResultHeadersInOrder.push_back(header);
			}
		}

		return {resultsMapping, currResultHeadersInOrder};
	}

	//compare resulting disparity map with a ground truth (or some other disparity map...)
	//this function takes as input the file names of a two disparity maps and the factor
	//that each disparity was scaled by in the generation of the disparity map image
	void compareDispMaps(const DisparityMap<float>& outputDisparityMap, const DisparityMap<float>& groundTruthDisparityMap,
						 std::ostream& resultsStream)
	{
		const OutputEvaluationResults<float> outputEvalResults =
			outputDisparityMap.getOuputComparison(groundTruthDisparityMap, OutputEvaluationParameters<float>());
		resultsStream << outputEvalResults;
	}

	//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
	//on the reference and test images specified by numStereoSet
	//run only optimized implementation if runOptImpOnly is true
	template<typename T, unsigned int DISP_VALS_OPTIMIZED, unsigned int DISP_VALS_SINGLE_THREAD>
		beliefprop::Status runStereoTwoImpsAndCompare(std::ostream& outStream,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_OPTIMIZED>>& optimizedImp,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_SINGLE_THREAD>>& singleThreadCPUImp,
		const unsigned int numStereoSet, const beliefprop::BPsettings& algSettings,
		const beliefprop::ParallelParameters& parallelParams,
		bool runOptImpOnly = false)
	{
		const unsigned int numImpsRun{runOptImpOnly ? 1u : 2u};
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
		//run optimized implementation and retrieve structure with runtime and output disparity map
		run_output[0] = optimizedImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, parallelParams, outStream);
		//check if error in run
		if ((run_output[0].runTime == 0.0) || (run_output[0].outDisparityMap.getHeight() == 0)) {
        	return beliefprop::Status::ERROR;
		}
		//save resulting disparity map
		run_output[0].outDisparityMap.saveDisparityMap(output_disp[0].string(), bp_params::SCALE_BP[numStereoSet]);
		outStream << OPTIMIZED_RUNTIME_HEADER << ":" << run_output[0].runTime << std::endl;

        if (!runOptImpOnly) {
			//run single-threaded implementation and retrieve structure with runtime and output disparity map
			run_output[1] = singleThreadCPUImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, parallelParams, outStream);
			run_output[1].outDisparityMap.saveDisparityMap(output_disp[1].string(), bp_params::SCALE_BP[numStereoSet]);
			outStream << OPTIMIZED_RUNTIME_HEADER << ":" << run_output[1].runTime << std::endl;
		}

		for (unsigned int i = 0; i < numImpsRun; i++) {
			const std::string runDesc{(i == 0) ? optimizedImp->getBpRunDescription() : singleThreadCPUImp->getBpRunDescription()};
			std::cout << "Output disparity map from " << runDesc << " run at " << output_disp[i] << std::endl;
		}
		std::cout << std::endl;

        //compare resulting disparity maps with ground truth and to each other
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

		return beliefprop::Status::NO_ERROR;
	}

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED>
	void addInputAndParamsToStream(const beliefprop::BPsettings& algSettings, std::ofstream& resultsStream) {
		resultsStream << "DataType: " << beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)) << std::endl;
		resultsStream << "Stereo Set: " << bp_params::STEREO_SET[NUM_SET] << "\n";
		resultsStream << algSettings;
		beliefprop::writeRunSettingsToStream(resultsStream);
		const std::string dispValsTemplatedStr{(DISP_VALS_TEMPLATE_OPTIMIZED == 0) ? "NO" : "YES"};
		resultsStream << "DISP_VALS_TEMPLATED: " << dispValsTemplatedStr << std::endl;
	}

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, unsigned int DISP_VALS_TEMPLATE_SINGLE_THREAD>
	beliefprop::Status runBpOnSetAndUpdateResults(std::array<std::map<std::string, std::vector<std::string>>, 2>& resDefParParamsFinal,
								const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_OPTIMIZED>>& optimizedImp,
								const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_SINGLE_THREAD>>& singleThreadCPUImp)
	{
		//load all the BP default settings
		beliefprop::BPsettings algSettings;
		algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

		//parallel parameters initialized with default thread count dimensions at every level
		beliefprop::ParallelParameters parallelParams(algSettings.numLevels_, PARALLEL_PARAMS_DEFAULT);

		//if optimizing parallel parameters, parallelParamsVect contains parallel parameter settings to run
		//(and contains only the default parallel parameters if not)
		std::vector<std::array<unsigned int, 2>> parallelParamsVect{
			OPTIMIZE_PARALLEL_PARAMS ? PARALLEL_PARAMETERS_OPTIONS : std::vector<std::array<unsigned int, 2>>{PARALLEL_PARAMS_DEFAULT}};
		
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
		for (unsigned int runNum=0; runNum < (OPTIMIZE_PARALLEL_PARAMS ? (parallelParamsVect.size() + 1) : parallelParamsVect.size()); runNum++) {
			std::ofstream resultsStream(BP_RUN_OUTPUT_FILE, std::ofstream::out);
			addInputAndParamsToStream<T, NUM_SET, DISP_VALS_TEMPLATE_OPTIMIZED>(algSettings, resultsStream);

			//get and set parallel parameters for current run if not final run that uses optimized parameters
			//final run uses default parameters if not optimizing parallel parameters
			const std::array<unsigned int, 2>* pParamsCurrRun = (runNum < parallelParamsVect.size()) ? (&(parallelParamsVect[runNum])) : nullptr;
			if (pParamsCurrRun) {
				//set parallel parameters to current parameters for each BP processing level
				parallelParams.setParallelDims(*pParamsCurrRun, algSettings.numLevels_);
			}

			if constexpr (OPTIMIZE_PARALLEL_PARAMS &&
				(optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN))
			{
				//write parallel parameters for each kernel to output stream if allowing different parallel parameters for each kernel in the same run
				parallelParams.writeToStream(resultsStream);
			}

			//run optimized implementation only (and not single-threaded CPU implementation) if not final run or run is using default parameter parameters
			//final run and run using default parallel parameters are only runs that are output in final results
			const bool runOptImpOnly{(!((runNum == parallelParamsVect.size()) || ((*pParamsCurrRun) == PARALLEL_PARAMS_DEFAULT)))};
			const auto errCode = runStereoTwoImpsAndCompare<T, DISP_VALS_TEMPLATE_OPTIMIZED, DISP_VALS_TEMPLATE_SINGLE_THREAD>(
				resultsStream, optimizedImp, singleThreadCPUImp, NUM_SET, algSettings, parallelParams, runOptImpOnly);
			resultsStream.close();

			//if error using default parameters, exit function with error
			if (pParamsCurrRun && (((*pParamsCurrRun) == PARALLEL_PARAMS_DEFAULT) && (errCode != beliefprop::Status::NO_ERROR))) {
				return beliefprop::Status::ERROR;
			}

			//retrieve and store results from current run if using default parallel parameters or is final run w/ optimized parallel parameters
			if ((runNum == parallelParamsVect.size()) || ((*pParamsCurrRun) == PARALLEL_PARAMS_DEFAULT))
			{
				//set output for runs using default parallel parameters and final run (which is the same run if not optimizing parallel parameters)
				const auto resultsCurrentRun = getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).first;
				//set output for run using default parallel parameters or final run if run was valid
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

            if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				//retrieve and store results including runtimes for each kernel if allowing different parallel parameters for each kernel and
				//total runtime for current run
				//if error in run, don't add results for current parallel parameters to results set
				if (errCode == beliefprop::Status::NO_ERROR) {
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
					}
				}

				//get optimized parallel parameters if next run is final run that uses optimized parallel parameters
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
		}
		
		return beliefprop::Status::NO_ERROR;
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

	template<typename T, unsigned int NUM_SET, bool TEMPLATED_DISP_IN_OPT_IMP>
	beliefprop::Status runBpOnSetAndUpdateResults(std::array<std::map<std::string, std::vector<std::string>>, 2>& resDefParParamsFinal) {
		std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>> runBpStereoSingleThread =
			std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
		//RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
		if constexpr (TEMPLATED_DISP_IN_OPT_IMP) {
			std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>> runBpOptimizedImp = 
				std::make_unique<RunBpOptimized<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
			const auto errCode = RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET>(resDefParParamsFinal, runBpOptimizedImp, runBpStereoSingleThread);
			if (errCode != beliefprop::Status::NO_ERROR) {
				return beliefprop::Status::ERROR;
			}
		}
		else {
			std::unique_ptr<RunBpStereoSet<T, 0>> runBpOptimizedImp = 
				std::make_unique<RunBpOptimized<T, 0>>();
			const auto errCode = RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET>(resDefParParamsFinal, runBpOptimizedImp, runBpStereoSingleThread);
			if (errCode != beliefprop::Status::NO_ERROR) {
				return beliefprop::Status::ERROR;
			}
		}
		//retrieve and store headers in output if not already retrieved to use for final output
		if (headersInOrder.size() == 0) {
			headersInOrder = RunAndEvaluateBpResults::getResultsMappingFromFile(BP_RUN_OUTPUT_FILE).second;
		}
		return beliefprop::Status::NO_ERROR;
	}

	void runBpOnStereoSets() {
		std::array<std::map<std::string, std::vector<std::string>>, 2> resDefParParamsFinal;
		std::vector<beliefprop::Status> runSuccess;
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 0, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 0, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 1, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 1, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 2, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 2, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 3, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 3, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 4, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 4, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 5, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 5, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 6, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<float, 6, false>(resDefParParamsFinal));
#ifdef DOUBLE_PRECISION_SUPPORTED
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 0, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 0, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 1, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 1, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 2, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 2, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 3, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 3, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 4, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 4, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 5, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 5, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 6, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<double, 6, false>(resDefParParamsFinal));
#endif //DOUBLE_PRECISION_SUPPORTED
#ifdef HALF_PRECISION_SUPPORTED
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 0, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 0, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 1, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 1, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 2, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 2, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 3, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 3, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 4, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 4, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 5, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 5, false>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 6, true>(resDefParParamsFinal));
		runSuccess.push_back(runBpOnSetAndUpdateResults<halftype, 6, false>(resDefParParamsFinal));
#endif //HALF_PRECISION_SUPPORTED

        if (headersInOrder.size() > 0) {
			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				//retrieve and print average and median speedup using optimized
				//parallel parameters compared to default
				RunAndEvaluateBpResults::getAverageMedianSpeedup(resDefParParamsFinal);
			}

			//write results from default and optimized parallel parameters runs to csv file
			std::array<std::ofstream, 2> resultsStreamDefaultTBFinal{std::ofstream(OPTIMIZE_PARALLEL_PARAMS ? BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE : BP_ALL_RUNS_OUTPUT_CSV_FILE),
																	OPTIMIZE_PARALLEL_PARAMS ? std::ofstream(BP_ALL_RUNS_OUTPUT_CSV_FILE) : std::ofstream()};
			for (const auto& currHeader : headersInOrder) {
				resultsStreamDefaultTBFinal[0] << currHeader << ",";
				if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
					resultsStreamDefaultTBFinal[1] << currHeader << ",";
				}
			}

			resultsStreamDefaultTBFinal[0] << std::endl;
			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				resultsStreamDefaultTBFinal[1] << "Speedup Over Default Parallel Parameters" << ",";
				resultsStreamDefaultTBFinal[1] << std::endl;
			}

			for (unsigned int i=0; i < (OPTIMIZE_PARALLEL_PARAMS ? resultsStreamDefaultTBFinal.size() : 1); i++) {
				unsigned int numRunWOutput{0};
				for (unsigned int j=0; j < runSuccess.size(); j++) {
					for (const auto& currHeader : headersInOrder) {
						if (runSuccess[j] == beliefprop::Status::ERROR) {
							resultsStreamDefaultTBFinal[i] << "Error in run" << ",";
						}
						else {
							resultsStreamDefaultTBFinal[i] << resDefParParamsFinal[i].at(currHeader).at(numRunWOutput) << ",";
						}
					}
					if ((runSuccess[j] == beliefprop::Status::NO_ERROR) && ((i == 1) && OPTIMIZE_PARALLEL_PARAMS)) {
						resultsStreamDefaultTBFinal[1] << resDefParParamsFinal[i].at("Speedup Over Default Parallel Parameters").at(numRunWOutput) << ",";
					}
					resultsStreamDefaultTBFinal[i] << std::endl;
					if (runSuccess[j] == beliefprop::Status::NO_ERROR) {
						numRunWOutput++;
					}
				}
			}
			resultsStreamDefaultTBFinal[0].close();
			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				resultsStreamDefaultTBFinal[1].close();
			}

			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (optimized parallel parameters) in "
						<< BP_ALL_RUNS_OUTPUT_CSV_FILE << std::endl;
				std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (default parallel parameters) in "
						<< BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE << std::endl;
			}
			else {
				std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run using default parallel parameters in "
						<< BP_ALL_RUNS_OUTPUT_CSV_FILE << std::endl;
			}
		}
		else {
			std::cout << "Error, no runs completed successfully" << std::endl;
		}
	}
};

#endif /* RUNANDEVALUATEBPRESULTS_H_ */
