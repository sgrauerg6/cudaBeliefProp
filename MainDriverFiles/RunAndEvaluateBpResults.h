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
#include <set>
#include "../ParameterFiles/bpRunSettings.h"
#include "../ParameterFiles/bpStereoParameters.h"
#include "../FileProcessing/BpFileHandling.h"
#include "../BpAndSmoothProcessing/RunBpStereoSet.h"
#include "../SingleThreadCPU/stereo.h"
#include "../OutputEvaluation/RunData.h"

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
	const std::string BP_RUN_OUTPUT_RESULTS_FILE{"outputResultsForRun.txt"};
	const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE{"outputResults.csv"};
	const std::string BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE{"outputResultsDefaultParallelParams.csv"};
	const std::string OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};
	const std::string SPEEDUP_HEADER{"Speedup Over Default Parallel Parameters"};

	std::vector<std::array<std::string, 2>> getResultsMappingFromFile(const std::string& fileName) {
		std::vector<std::array<std::string, 2>> dataWHeaders;
		std::set<std::string> headersSet;
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
				while (headersSet.count(header) > 0) {
					i++;
					header = origHeader + "_" + std::to_string(i);
				}
				headersSet.insert(header);
				dataWHeaders.push_back({header, result});
			}
		}

		return dataWHeaders;
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

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED>
	RunData inputAndParamsRunData(const beliefprop::BPsettings& algSettings) {
		RunData currRunData;
		currRunData.addDataWHeader("DataType", beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)));
		currRunData.addDataWHeader("Stereo Set", bp_params::STEREO_SET[NUM_SET]);
		currRunData.appendData(algSettings.runData());
		currRunData.appendData(beliefprop::runSettings());
		const std::string dispValsTemplatedStr{(DISP_VALS_TEMPLATE_OPTIMIZED == 0) ? "NO" : "YES"};
		currRunData.addDataWHeader("DISP_VALS_TEMPLATED", dispValsTemplatedStr);
		return currRunData;
	}

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, unsigned int DISP_VALS_TEMPLATE_SINGLE_THREAD>
	std::pair<beliefprop::Status, std::vector<RunData>> runBpOnSetAndUpdateResults(
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_OPTIMIZED>>& optimizedImp,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_SINGLE_THREAD>>& singleThreadCPUImp)
	{
		std::vector<RunData> outRunData(OPTIMIZE_PARALLEL_PARAMS ? 2 : 1);
		enum class RunType { ONLY_RUN, DEFAULT_PARAMS, OPTIMIZED_RUN, TEST_PARAMS };
		std::array<std::array<std::map<std::string, std::string>, 2>, 2> inParamsResultsDefOptRuns;
		//load all the BP default settings
		beliefprop::BPsettings algSettings;
		algSettings.numDispVals_ = bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET];

		//parallel parameters initialized with default thread count dimensions at every level
		beliefprop::ParallelParameters parallelParams(algSettings.numLevels_, PARALLEL_PARAMS_DEFAULT);

		//if optimizing parallel parameters, parallelParamsVect contains parallel parameter settings to run
		//(and contains only the default parallel parameters if not)
		std::vector<std::array<unsigned int, 2>> parallelParamsVect{
			OPTIMIZE_PARALLEL_PARAMS ? PARALLEL_PARAMETERS_OPTIONS : std::vector<std::array<unsigned int, 2>>()};
		
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
		for (unsigned int runNum=0; runNum < (parallelParamsVect.size() + 1); runNum++) {
			//initialize current run type to specify if current run is only run, run with default params, test params run, or final run with optiized params
			RunType currRunType{RunType::TEST_PARAMS};
			if constexpr (!OPTIMIZE_PARALLEL_PARAMS) {
				currRunType = RunType::ONLY_RUN;
			}
			else if (runNum == parallelParamsVect.size()) {
				currRunType = RunType::OPTIMIZED_RUN;
			}

			//get and set parallel parameters for current run if not final run that uses optimized parameters
			std::array<unsigned int, 2> pParamsCurrRun{PARALLEL_PARAMS_DEFAULT};
			if (currRunType == RunType::ONLY_RUN) {
				parallelParams.setParallelDims(PARALLEL_PARAMS_DEFAULT, algSettings.numLevels_);
			}
			else if (currRunType == RunType::TEST_PARAMS) {
				//set parallel parameters to parameters corresponding to current run for each BP processing level
				pParamsCurrRun = parallelParamsVect[runNum];
				parallelParams.setParallelDims(pParamsCurrRun, algSettings.numLevels_);
				if (pParamsCurrRun == PARALLEL_PARAMS_DEFAULT) {
					//set run type to default parameters if current run uses default parameters
					currRunType = RunType::DEFAULT_PARAMS;
				}
			}

			//store input params data if using default parallel parameters or final run with optimized parameters
			RunData currRunData;
			if (currRunType != RunType::TEST_PARAMS) {
				currRunData.appendData(inputAndParamsRunData<T, NUM_SET, DISP_VALS_TEMPLATE_OPTIMIZED>(algSettings));
				if constexpr (OPTIMIZE_PARALLEL_PARAMS &&
					(optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN))
				{
					//add parallel parameters for each kernel to current input data if allowing different parallel parameters for each kernel in the same run
					currRunData.appendData(parallelParams.runData());
				}
			}

			//run optimized implementation only if not final run or run is using default parameter parameters
			const bool runOptImpOnly{currRunType == RunType::TEST_PARAMS};
			std::ofstream resultsStream(BP_RUN_OUTPUT_RESULTS_FILE, std::ofstream::out);
			//run belief propagation implementation(s) and return whether or not error in run
			//detailed results stored to file that is generated using stream
			const auto errCode = runStereoTwoImpsAndCompare<T, DISP_VALS_TEMPLATE_OPTIMIZED, DISP_VALS_TEMPLATE_SINGLE_THREAD>(
				resultsStream, optimizedImp, singleThreadCPUImp, NUM_SET, algSettings, parallelParams, runOptImpOnly);
  			currRunData.addDataWHeader("Run Success", (errCode == beliefprop::Status::NO_ERROR) ? "Yes" : "No");
			resultsStream.close();

			//if error in run and run is any type other than for testing parameters, exit function with error
			if ((errCode != beliefprop::Status::NO_ERROR) && (currRunType != RunType::TEST_PARAMS)) {
				return {beliefprop::Status::ERROR, {currRunData}};
			}

			//retrieve results from current run
			const auto resultsDataCurrRun = getResultsMappingFromFile(BP_RUN_OUTPUT_RESULTS_FILE);
			for (const auto& currResultWHeader : resultsDataCurrRun) {
				currRunData.addDataWHeader(currResultWHeader[0], currResultWHeader[1]);
			}

            //add current run results for output if using default parallel parameters or is final run w/ optimized parallel parameters
			if (currRunType != RunType::TEST_PARAMS) {
				//set output for runs using default parallel parameters and final run (which is the same run if not optimizing parallel parameters)
				if (currRunType == RunType::OPTIMIZED_RUN) {
					outRunData[1] = currRunData;
				}
				else {
					outRunData[0] = currRunData;
				}
			}

            if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				//retrieve and store results including runtimes for each kernel if allowing different parallel parameters for each kernel and
				//total runtime for current run
				//if error in run, don't add results for current parallel parameters to results set
				if (errCode == beliefprop::Status::NO_ERROR) {
					if (currRunType != RunType::OPTIMIZED_RUN) {
						if constexpr (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
							for (unsigned int level=0; level < algSettings.numLevels_; level++) {
								pParamsToRunTimeEachKernel[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][pParamsCurrRun] =
									std::stod(currRunData.getData("Level " + std::to_string(level) + " Data Costs (" +
											std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
								pParamsToRunTimeEachKernel[beliefprop::BpKernel::BP_AT_LEVEL][level][pParamsCurrRun] = 
									std::stod(currRunData.getData("Level " + std::to_string(level) + " BP Runtime (" + 
											std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
								pParamsToRunTimeEachKernel[beliefprop::BpKernel::COPY_AT_LEVEL][level][pParamsCurrRun] =
									std::stod(currRunData.getData("Level " + std::to_string(level) + " Copy Runtime (" + 
																std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
							}
							pParamsToRunTimeEachKernel[beliefprop::BpKernel::BLUR_IMAGES][0][pParamsCurrRun] =
									std::stod(currRunData.getData("Smoothing Runtime (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
							pParamsToRunTimeEachKernel[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][pParamsCurrRun] =
									std::stod(currRunData.getData("Time to init message values (kernel portion only) (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
							pParamsToRunTimeEachKernel[beliefprop::BpKernel::OUTPUT_DISP][0][pParamsCurrRun] =
									std::stod(currRunData.getData("Time get output disparity (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings) "));
						}
						//get total runtime
						pParamsToRunTimeEachKernel[beliefprop::NUM_KERNELS][0][pParamsCurrRun] =
								std::stod(currRunData.getData(OPTIMIZED_RUNTIME_HEADER));
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
		
		return {beliefprop::Status::NO_ERROR, outRunData};
	}

	//get average and median speedup using optimized parallel parameters compared to default parallel parameters
	void getAverageMedianSpeedup(std::vector<std::pair<beliefprop::Status, std::vector<RunData>>>& runOutput) {
		std::vector<double> speedupsVect;
		for (unsigned int i=0; i < runOutput.size(); i++) {
			if (runOutput[i].first == beliefprop::Status::NO_ERROR) {
				speedupsVect.push_back(std::stod(runOutput[i].second[0].getData(OPTIMIZED_RUNTIME_HEADER)) / 
								       std::stod(runOutput[i].second[1].getData(OPTIMIZED_RUNTIME_HEADER)));
				runOutput[i].second[1].addDataWHeader(SPEEDUP_HEADER, std::to_string(speedupsVect.back()));
			}
		}
		if (speedupsVect.size() > 0) {
			std::sort(speedupsVect.begin(), speedupsVect.end());
			std::cout << "Average speedup: " << 
				(std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size()) << std::endl;
			const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
				(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
				speedupsVect[(speedupsVect.size() / 2)];
			std::cout << "Median speedup: " << medianSpeedup << std::endl;
		}
	}

	template<typename T, unsigned int NUM_SET, bool TEMPLATED_DISP_IN_OPT_IMP>
	std::pair<beliefprop::Status, std::vector<RunData>> runBpOnSetAndUpdateResults() {
		std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>> runBpStereoSingleThread =
			std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
		//RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
		if constexpr (TEMPLATED_DISP_IN_OPT_IMP) {
			std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>> runBpOptimizedImp = 
				std::make_unique<RunBpOptimized<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
			return RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET>(runBpOptimizedImp, runBpStereoSingleThread);
		}
		else {
			std::unique_ptr<RunBpStereoSet<T, 0>> runBpOptimizedImp = 
				std::make_unique<RunBpOptimized<T, 0>>();
			return RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET>(runBpOptimizedImp, runBpStereoSingleThread);
		}
	}

	void runBpOnStereoSets() {
		std::vector<std::pair<beliefprop::Status, std::vector<RunData>>> runOutput;
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 0, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 0, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 1, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 1, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 2, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 2, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 3, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 3, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 4, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 4, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 5, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 5, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 6, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<float, 6, false>());
#ifdef DOUBLE_PRECISION_SUPPORTED
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 0, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 0, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 1, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 1, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 2, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 2, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 3, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 3, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 4, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 4, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 5, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 5, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 6, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<double, 6, false>());
#endif //DOUBLE_PRECISION_SUPPORTED
#ifdef HALF_PRECISION_SUPPORTED
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 0, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 0, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 1, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 1, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 2, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 2, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 3, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 3, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 4, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 4, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 5, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 5, false>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 6, true>());
		runOutput.push_back(runBpOnSetAndUpdateResults<halftype, 6, false>());
#endif //HALF_PRECISION_SUPPORTED

        //get iterator to first run with success
		const auto firstSuccessRun = std::find_if(runOutput.begin(), runOutput.end(), [](const auto& runResult)
			{ return (runResult.first == beliefprop::Status::NO_ERROR); } );
		
		//check if there was at least one successful run
		if (firstSuccessRun != runOutput.end()) {
			//get headers from first successful run
			const auto headersInOrder = firstSuccessRun->second[0].getHeadersInOrder();

			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				//retrieve and print average and median speedup using optimized
				//parallel parameters compared to default
				RunAndEvaluateBpResults::getAverageMedianSpeedup(runOutput);
			}

			//write results from default and optimized parallel parameters runs to csv file
			std::array<std::ofstream, 2> resultsStreamDefaultTBFinal{std::ofstream(OPTIMIZE_PARALLEL_PARAMS ? BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE : BP_ALL_RUNS_OUTPUT_CSV_FILE),
															 		OPTIMIZE_PARALLEL_PARAMS ? std::ofstream(BP_ALL_RUNS_OUTPUT_CSV_FILE) : std::ofstream()};
			for (const auto& currHeader : headersInOrder) {
				resultsStreamDefaultTBFinal[0] << currHeader << ",";
			}
			resultsStreamDefaultTBFinal[0] << std::endl;

			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				for (const auto& currHeader : headersInOrder) {
					resultsStreamDefaultTBFinal[1] << currHeader << ",";
				}
				resultsStreamDefaultTBFinal[1] << SPEEDUP_HEADER << "," << std::endl;
			}

			for (unsigned int i=0; i < (OPTIMIZE_PARALLEL_PARAMS ? resultsStreamDefaultTBFinal.size() : 1); i++) {
				for (unsigned int runNum=0; runNum < runOutput.size(); runNum++) {
					//if run not successful only have single set of output data from run
					const unsigned int runResultIdx = (runOutput[runNum].first == beliefprop::Status::NO_ERROR) ? i : 0;
					for (const auto& currHeader : headersInOrder) {
						if (!(runOutput[runNum].second[runResultIdx].isData(currHeader))) {
							resultsStreamDefaultTBFinal[i] << "No Data" << ",";
						}
						else {
							resultsStreamDefaultTBFinal[i] << runOutput[runNum].second[runResultIdx].getData(currHeader) << ",";
						}
					}
					if ((runOutput[runNum].first == beliefprop::Status::NO_ERROR) && ((i == 1) && OPTIMIZE_PARALLEL_PARAMS)) {
						resultsStreamDefaultTBFinal[1] << runOutput[runNum].second[i].getData(SPEEDUP_HEADER) << ",";
					}
					resultsStreamDefaultTBFinal[i] << std::endl;
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
