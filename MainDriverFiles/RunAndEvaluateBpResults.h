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

#define SMALLER_SETS_ONLY

//check if optimized CPU run defined and make any necessary additions to support it
#ifdef OPTIMIZED_CPU_RUN
//needed to run the optimized implementation a stereo set using CPU
#include "../OptimizeCPU/RunBpStereoOptimizedCPU.h"
//set RunBpOptimized alias to correspond to optimized CPU implementation
template <typename T, unsigned int DISP_VALS, beliefprop::AccSetting ACCELERATION>
using RunBpOptimized = RunBpStereoOptimizedCPU<T, DISP_VALS, ACCELERATION>;
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
template <typename T, unsigned int DISP_VALS, beliefprop::AccSetting ACCELERATION>
using RunBpOptimized = RunBpStereoSetOnGPUWithCUDA<T, DISP_VALS, beliefprop::AccSetting::CUDA>;
//set data type used for half-precision with CUDA
using halftype = half;
#endif //OPTIMIZED_CUDA_RUN

using MultRunData = std::vector<std::pair<beliefprop::Status, std::vector<RunData>>>;
using MultRunSpeedup = std::pair<std::string, std::pair<double, double>>;

namespace RunAndEvaluateBpResults {
	//constants for output results for individual and sets of runs
	const std::string BP_RUN_OUTPUT_RESULTS_FILE{"outputResultsForRun.txt"};
	const std::string BP_ALL_RUNS_OUTPUT_CSV_FILE_NAME_START{"outputResults"};
	const std::string BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE_START{"outputResultsDefaultParallelParams"};
	const std::string CSV_FILE_EXTENSION{".csv"};
	const std::string OPTIMIZED_RUNTIME_HEADER{"Median Optimized Runtime (including transfer time)"};
	const std::string SPEEDUP_OPT_PAR_PARAMS_HEADER{"Speedup Over Default OMP Thread Count / CUDA Thread Block Dimensions"};
	const std::string SPEEDUP_DOUBLE{"Speedup using double-precision relative to float (actually slowdown)"};
	const std::string SPEEDUP_HALF{"Speedup using half-precision relative to float"};
	const std::string SPEEDUP_DISP_COUNT_TEMPLATE{"Speedup w/ templated disparity count (known at compile-time)"};
	const std::string SPEEDUP_VECTORIZATION{"Speedup using CPU vectorization"};
#ifdef SMALLER_SETS_ONLY
	const std::string BASELINE_RUNTIMES_FILE_PATH{"../baselineRuntimes/baselineRuntimesSmallerSetsOnly.txt"};
#else
	const std::string BASELINE_RUNTIMES_FILE_PATH{"../baselineRuntimes/baselineRuntimes.txt"};
#endif //SMALLER_SETS_ONLY

	std::pair<std::string, std::vector<double>> getBaselineRuntimeData() {
		std::ifstream baselineData(BASELINE_RUNTIMES_FILE_PATH);
		std::string line;
		//first line of data is string with baseline processor description and all subsequent data is runtimes
		//on that processor in same order as runtimes from runBpOnStereoSets() function
		std::pair<std::string, std::vector<double>> baselineNameData;
		bool firstLine{true};
		while (std::getline(baselineData, line))
		{
			if (firstLine) {
				baselineNameData.first = line;
				firstLine = false;
			}
			else {
				baselineNameData.second.push_back(std::stod(line));
			}
		}

		return baselineNameData;
	}

	std::vector<MultRunSpeedup> getAvgMedSpeedupOverBaseline(MultRunData& runOutput,
		const std::string& dataTypeStr) {
		std::vector<double> speedupsVect;
		const auto baselineRunData = getBaselineRuntimeData();
		const std::string speedupHeader = "Speedup relative to " + baselineRunData.first + " - " + dataTypeStr + " only";
		const auto baselineRuntimes = baselineRunData.second;
		std::vector<MultRunSpeedup> speedupData;
		for (unsigned int i=0; i < runOutput.size(); i++) {
			if (runOutput[i].first == beliefprop::Status::NO_ERROR) {
				speedupsVect.push_back(baselineRuntimes[i] / 
								       std::stod(runOutput[i].second[1].getData(OPTIMIZED_RUNTIME_HEADER)));
				runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
				runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
			}
		}
		if (speedupsVect.size() > 0) {
			const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
			std::cout << "Average " << speedupHeader << ": " << averageSpeedup << std::endl;
			std::sort(speedupsVect.begin(), speedupsVect.end());
			const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
				(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
				speedupsVect[(speedupsVect.size() / 2)];
			std::cout << "Median " << speedupHeader << ": " << medianSpeedup << std::endl;
			speedupData.push_back({speedupHeader, {averageSpeedup, medianSpeedup}});
		}
		else {
			return {MultRunSpeedup()};
		}
		//if processing floats, also get results for 3 smallest and 3 largest stereo sets
		if (dataTypeStr == beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(float))) {
			speedupsVect.clear();
			std::string speedupHeader = "Speedup relative to " + baselineRunData.first + " on 3 smallest stereo sets - " + dataTypeStr + " only";
			for (unsigned int i=0; i < 6; i++) {
				if (runOutput[i].first == beliefprop::Status::NO_ERROR) {
					speedupsVect.push_back(baselineRuntimes[i] / 
										std::stod(runOutput[i].second[1].getData(OPTIMIZED_RUNTIME_HEADER)));
					runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
					runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
				}
			}
			if (speedupsVect.size() > 0) {
				const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
				std::cout << "Average " << speedupHeader << ": " << averageSpeedup << std::endl;
				std::sort(speedupsVect.begin(), speedupsVect.end());
				const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
					(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
					speedupsVect[(speedupsVect.size() / 2)];
				std::cout << "Median " << speedupHeader << ": " << medianSpeedup << std::endl;
				speedupData.push_back({speedupHeader, {averageSpeedup, medianSpeedup}});
			}
#ifndef SMALLER_SETS_ONLY
			speedupHeader = "Speedup relative to " + baselineRunData.first + " on 3 largest stereo sets - " + dataTypeStr + " only";
			for (unsigned int i=9; i < 15; i++) {
#else
			speedupHeader = "Speedup relative to " + baselineRunData.first + " on largest stereo set - " + dataTypeStr + " only";
			speedupsVect.clear();
			for (unsigned int i=9; i < 11; i++) {
#endif //SMALLER_SETS_ONLY
				if (runOutput[i].first == beliefprop::Status::NO_ERROR) {
					speedupsVect.push_back(baselineRuntimes[i] / 
										std::stod(runOutput[i].second[1].getData(OPTIMIZED_RUNTIME_HEADER)));
					runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
					runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
				}
			}
			if (speedupsVect.size() > 0) {
				const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
				std::cout << "Average " << speedupHeader << ": " << averageSpeedup << std::endl;
				std::sort(speedupsVect.begin(), speedupsVect.end());
				const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
					(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
					speedupsVect[(speedupsVect.size() / 2)];
				std::cout << "Median " << speedupHeader << ": " << medianSpeedup << std::endl;
				speedupData.push_back({speedupHeader, {averageSpeedup, medianSpeedup}});
			}
		}
		return speedupData;
	}

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
	RunData compareDispMaps(const DisparityMap<float>& outputDisparityMap, const DisparityMap<float>& groundTruthDisparityMap)
	{
		const OutputEvaluationResults<float> outputEvalResults =
			outputDisparityMap.getOuputComparison(groundTruthDisparityMap, OutputEvaluationParameters<float>());
		return outputEvalResults.runData();
	}

	//run and compare output disparity maps using the given optimized and single-threaded stereo implementations
	//on the reference and test images specified by numStereoSet
	//run only optimized implementation if runOptImpOnly is true
	template<typename T, unsigned int DISP_VALS_OPTIMIZED, unsigned int DISP_VALS_SINGLE_THREAD, beliefprop::AccSetting OPT_IMP_ACCEL>
		std::pair<beliefprop::Status, RunData> runStereoTwoImpsAndCompare(
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_OPTIMIZED, OPT_IMP_ACCEL>>& optimizedImp,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_SINGLE_THREAD, beliefprop::AccSetting::NONE>>& singleThreadCPUImp,
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
		RunData runData;
		//run optimized implementation and retrieve structure with runtime and output disparity map
		run_output[0] = optimizedImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, parallelParams);
		//check if error in run
		if ((run_output[0].runTime == 0.0) || (run_output[0].outDisparityMap.getHeight() == 0)) {
        	return {beliefprop::Status::ERROR, runData};
		}
		runData.appendData(run_output[0].runData);
		//save resulting disparity map
		run_output[0].outDisparityMap.saveDisparityMap(output_disp[0].string(), bp_params::SCALE_BP[numStereoSet]);
		runData.addDataWHeader(OPTIMIZED_RUNTIME_HEADER, std::to_string(run_output[0].runTime));

        if (!runOptImpOnly) {
			//run single-threaded implementation and retrieve structure with runtime and output disparity map
			run_output[1] = singleThreadCPUImp->operator()({refTestImagePath[0].string(), refTestImagePath[1].string()}, algSettings, parallelParams);
			run_output[1].outDisparityMap.saveDisparityMap(output_disp[1].string(), bp_params::SCALE_BP[numStereoSet]);
	 		runData.appendData(run_output[1].runData);
		}

		for (unsigned int i = 0; i < numImpsRun; i++) {
			const std::string runDesc{(i == 0) ? optimizedImp->getBpRunDescription() : singleThreadCPUImp->getBpRunDescription()};
			std::cout << "Output disparity map from " << runDesc << " run at " << output_disp[i] << std::endl;
		}
		std::cout << std::endl;

        //compare resulting disparity maps with ground truth and to each other
		const filepathtype groundTruthDisp{bpFileSettings.getGroundTruthDisparityFilePath()};
		DisparityMap<float> groundTruthDisparityMap(groundTruthDisp.string(), bp_params::SCALE_BP[numStereoSet]);
		runData.addDataWHeader(optimizedImp->getBpRunDescription() + " output vs. Ground Truth result", std::string());
		runData.appendData(compareDispMaps(run_output[0].outDisparityMap, groundTruthDisparityMap));
        if (!runOptImpOnly) {
			runData.addDataWHeader(singleThreadCPUImp->getBpRunDescription() + " output vs. Ground Truth result", std::string());
			runData.appendData(compareDispMaps(run_output[1].outDisparityMap, groundTruthDisparityMap));

			runData.addDataWHeader(optimizedImp->getBpRunDescription() + " output vs. " + singleThreadCPUImp->getBpRunDescription() + " result", std::string());
			runData.appendData(compareDispMaps(run_output[0].outDisparityMap, run_output[1].outDisparityMap));
		}

		return {beliefprop::Status::NO_ERROR, runData};
	}

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, beliefprop::AccSetting ACC_SETTING>
	void addInputAndParamsToStream(const beliefprop::BPsettings& algSettings, std::ofstream& resultsStream) {
		resultsStream << "DataType: " << beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)) << std::endl;
		resultsStream << "Stereo Set: " << bp_params::STEREO_SET[NUM_SET] << "\n";
		resultsStream << algSettings;
		beliefprop::writeRunSettingsToStream<ACC_SETTING>(resultsStream);
		const std::string dispValsTemplatedStr{(DISP_VALS_TEMPLATE_OPTIMIZED == 0) ? "NO" : "YES"};
		resultsStream << "DISP_VALS_TEMPLATED: " << dispValsTemplatedStr << std::endl;
	}

	template<typename T, unsigned int NUM_SET, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, beliefprop::AccSetting ACC_SETTING>
	RunData inputAndParamsRunData(const beliefprop::BPsettings& algSettings) {
		RunData currRunData;
		currRunData.addDataWHeader("DataType", beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)));
		currRunData.addDataWHeader("Stereo Set", bp_params::STEREO_SET[NUM_SET]);
		currRunData.appendData(algSettings.runData());
		currRunData.appendData(beliefprop::runSettings<ACC_SETTING>());
		const std::string dispValsTemplatedStr{(DISP_VALS_TEMPLATE_OPTIMIZED == 0) ? "NO" : "YES"};
		currRunData.addDataWHeader("DISP_VALS_TEMPLATED", dispValsTemplatedStr);
		return currRunData;
	}

	template<typename T, unsigned int NUM_SET, beliefprop::AccSetting OPT_IMP_ACCEL, unsigned int DISP_VALS_TEMPLATE_OPTIMIZED, unsigned int DISP_VALS_TEMPLATE_SINGLE_THREAD>
	std::pair<beliefprop::Status, std::vector<RunData>> runBpOnSetAndUpdateResults(
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_OPTIMIZED, OPT_IMP_ACCEL>>& optimizedImp,
		const std::unique_ptr<RunBpStereoSet<T, DISP_VALS_TEMPLATE_SINGLE_THREAD, beliefprop::AccSetting::NONE>>& singleThreadCPUImp)
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
				currRunData.appendData(inputAndParamsRunData<T, NUM_SET, DISP_VALS_TEMPLATE_OPTIMIZED, OPT_IMP_ACCEL>(algSettings));
				if constexpr (OPTIMIZE_PARALLEL_PARAMS &&
					(optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN))
				{
					//add parallel parameters for each kernel to current input data if allowing different parallel parameters for each kernel in the same run
					currRunData.appendData(parallelParams.runData());
				}
			}

			//run optimized implementation only if not final run or run is using default parameter parameters
			const bool runOptImpOnly{currRunType == RunType::TEST_PARAMS};
			//run belief propagation implementation(s) and return whether or not error in run
			//detailed results stored to file that is generated using stream
			const auto runImpsECodeData = runStereoTwoImpsAndCompare<T, DISP_VALS_TEMPLATE_OPTIMIZED, DISP_VALS_TEMPLATE_SINGLE_THREAD, OPT_IMP_ACCEL>(
				optimizedImp, singleThreadCPUImp, NUM_SET, algSettings, parallelParams, runOptImpOnly);
  			currRunData.addDataWHeader("Run Success", (runImpsECodeData.first == beliefprop::Status::NO_ERROR) ? "Yes" : "No");

			//if error in run and run is any type other than for testing parameters, exit function with error
			if ((runImpsECodeData.first != beliefprop::Status::NO_ERROR) && (currRunType != RunType::TEST_PARAMS)) {
				return {beliefprop::Status::ERROR, {currRunData}};
			}

			//retrieve results from current run
			currRunData.appendData(runImpsECodeData.second);

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
				if (runImpsECodeData.first == beliefprop::Status::NO_ERROR) {
					if (currRunType != RunType::OPTIMIZED_RUN) {
						if constexpr (optParallelParamsSetting == beliefprop::OptParallelParamsSetting::ALLOW_DIFF_KERNEL_PARALLEL_PARAMS_IN_SAME_RUN) {
							for (unsigned int level=0; level < algSettings.numLevels_; level++) {
								pParamsToRunTimeEachKernel[beliefprop::BpKernel::DATA_COSTS_AT_LEVEL][level][pParamsCurrRun] =
									std::stod(currRunData.getData("Level " + std::to_string(level) + " Data Costs (" +
											std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"));
								pParamsToRunTimeEachKernel[beliefprop::BpKernel::BP_AT_LEVEL][level][pParamsCurrRun] = 
									std::stod(currRunData.getData("Level " + std::to_string(level) + " BP Runtime (" + 
											std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"));
								pParamsToRunTimeEachKernel[beliefprop::BpKernel::COPY_AT_LEVEL][level][pParamsCurrRun] =
									std::stod(currRunData.getData("Level " + std::to_string(level) + " Copy Runtime (" + 
																std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"));
							}
							pParamsToRunTimeEachKernel[beliefprop::BpKernel::BLUR_IMAGES][0][pParamsCurrRun] =
									std::stod(currRunData.getData("Smoothing Runtime (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"));
							pParamsToRunTimeEachKernel[beliefprop::BpKernel::INIT_MESSAGE_VALS][0][pParamsCurrRun] =
									std::stod(currRunData.getData("Time to init message values (kernel portion only) (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"));
							pParamsToRunTimeEachKernel[beliefprop::BpKernel::OUTPUT_DISP][0][pParamsCurrRun] =
									std::stod(currRunData.getData("Time get output disparity (" + std::to_string(bp_params::NUM_BP_STEREO_RUNS) + " timings)"));
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
	MultRunSpeedup getAvgMedSpeedupOptPParams(MultRunData& runOutput,
		const std::string& speedupHeader) {
		std::vector<double> speedupsVect;
		for (unsigned int i=0; i < runOutput.size(); i++) {
			if (runOutput[i].first == beliefprop::Status::NO_ERROR) {
				speedupsVect.push_back(std::stod(runOutput[i].second[0].getData(OPTIMIZED_RUNTIME_HEADER)) / 
								       std::stod(runOutput[i].second[1].getData(OPTIMIZED_RUNTIME_HEADER)));
				runOutput[i].second[0].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
				runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
			}
		}
		if (speedupsVect.size() > 0) {
			const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
			std::cout << "Average " << speedupHeader << ": " << averageSpeedup << std::endl;
			std::sort(speedupsVect.begin(), speedupsVect.end());
			const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
				(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
				speedupsVect[(speedupsVect.size() / 2)];
			std::cout << "Median " << speedupHeader << ": " << medianSpeedup << std::endl;
			return {speedupHeader, {averageSpeedup, medianSpeedup}};
		}
		return MultRunSpeedup();
	}

	MultRunSpeedup getAvgMedSpeedup(MultRunData& runOutputBase, MultRunData& runOutputTarget,
		const std::string& speedupHeader) {
		std::vector<double> speedupsVect;
		for (unsigned int i=0; i < runOutputBase.size(); i++) {
			if ((runOutputBase[i].first == beliefprop::Status::NO_ERROR) && (runOutputTarget[i].first == beliefprop::Status::NO_ERROR))  {
				speedupsVect.push_back(std::stod(runOutputBase[i].second.back().getData(OPTIMIZED_RUNTIME_HEADER)) / 
								       std::stod(runOutputTarget[i].second.back().getData(OPTIMIZED_RUNTIME_HEADER)));
				runOutputBase[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
				runOutputTarget[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
			}
		}
		if (speedupsVect.size() > 0) {
			const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
			std::cout << "Average " << speedupHeader << ": " << averageSpeedup << std::endl;
			std::sort(speedupsVect.begin(), speedupsVect.end());
			const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
				(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
				speedupsVect[(speedupsVect.size() / 2)];
			std::cout << "Median " << speedupHeader << ": " << medianSpeedup << std::endl;
			return {speedupHeader, {averageSpeedup, medianSpeedup}};
		}
		return {speedupHeader, {0.0, 0.0}};
	}

	MultRunSpeedup getAvgMedSpeedupDispValsInTemplate(MultRunData& runOutput,
		const std::string& speedupHeader) {
		std::vector<double> speedupsVect;
		//assumine that runs with and without disparity count given in template parameter are consectutive with the run with the
		//disparity count given in template being first
		for (unsigned int i=0; (i+1) < runOutput.size(); i+=2) {
			if ((runOutput[i].first == beliefprop::Status::NO_ERROR) && (runOutput[i+1].first == beliefprop::Status::NO_ERROR))  {
				speedupsVect.push_back(std::stod(runOutput[i+1].second.back().getData(OPTIMIZED_RUNTIME_HEADER)) / 
								       std::stod(runOutput[i].second.back().getData(OPTIMIZED_RUNTIME_HEADER)));
				runOutput[i].second[1].addDataWHeader(speedupHeader, std::to_string(speedupsVect.back()));
			}
		}
		if (speedupsVect.size() > 0) {
			const double averageSpeedup = (std::accumulate(speedupsVect.begin(), speedupsVect.end(), 0.0) / (double)speedupsVect.size());
			std::cout << "Average " << speedupHeader << ": " << averageSpeedup << std::endl;
			std::sort(speedupsVect.begin(), speedupsVect.end());
			const double medianSpeedup = ((speedupsVect.size() % 2) == 0) ? 
				(speedupsVect[(speedupsVect.size() / 2) - 1] + speedupsVect[(speedupsVect.size() / 2)]) / 2.0 : 
				speedupsVect[(speedupsVect.size() / 2)];
			std::cout << "Median " << speedupHeader << ": " << medianSpeedup << std::endl;
			return {speedupHeader, {averageSpeedup, medianSpeedup}};
		}
		return MultRunSpeedup();
	}

	template<typename T, unsigned int NUM_SET, bool TEMPLATED_DISP_IN_OPT_IMP, beliefprop::AccSetting OPT_IMP_ACCEL>
	std::pair<beliefprop::Status, std::vector<RunData>> runBpOnSetAndUpdateResults() {
		std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], beliefprop::AccSetting::NONE>> runBpStereoSingleThread =
			std::make_unique<RunBpStereoCPUSingleThread<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET]>>();
		//RunBpOptimized set to optimized belief propagation implementation (currently optimized CPU and CUDA implementations supported)
		if constexpr (TEMPLATED_DISP_IN_OPT_IMP) {
			std::unique_ptr<RunBpStereoSet<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], OPT_IMP_ACCEL>> runBpOptimizedImp = 
				std::make_unique<RunBpOptimized<T, bp_params::NUM_POSSIBLE_DISPARITY_VALUES[NUM_SET], OPT_IMP_ACCEL>>();
			return RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET, OPT_IMP_ACCEL>(runBpOptimizedImp, runBpStereoSingleThread);
		}
		else {
			std::unique_ptr<RunBpStereoSet<T, 0, OPT_IMP_ACCEL>> runBpOptimizedImp = 
				std::make_unique<RunBpOptimized<T, 0, OPT_IMP_ACCEL>>();
			return RunAndEvaluateBpResults::runBpOnSetAndUpdateResults<T, NUM_SET, OPT_IMP_ACCEL>(runBpOptimizedImp, runBpStereoSingleThread);
		}
	}

	//write data for file corresponding to runs for a specified data type or across all data type
	//includes results for each run as well as average and median speedup data across multiple runs
	template <beliefprop::AccSetting OPT_IMP_ACCEL, bool MULT_DATA_TYPES, typename T = void>
	void writeRunOutput(const std::pair<MultRunData, std::vector<MultRunSpeedup>>& runOutput) {
		//get iterator to first run with success
		const auto firstSuccessRun = std::find_if(runOutput.first.begin(), runOutput.first.end(), [](const auto& runResult)
			{ return (runResult.first == beliefprop::Status::NO_ERROR); } );
		
		//check if there was at least one successful run
		if (firstSuccessRun != runOutput.first.end()) {
			//write results from default and optimized parallel parameters runs to csv file
	        const std::string dataTypeStr = MULT_DATA_TYPES ? "MULT_DATA_TYPES" : beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T));
			const auto accelStr = beliefprop::accelerationString<OPT_IMP_ACCEL>();
			const std::string optResultsFileName{BP_ALL_RUNS_OUTPUT_CSV_FILE_NAME_START + "_" + 
				(PROCESSOR_NAME.size() > 0 ? PROCESSOR_NAME + "_" : "") + dataTypeStr + "_" + accelStr + CSV_FILE_EXTENSION};
			const std::string defaultParamsResultsFileName{BP_ALL_RUNS_OUTPUT_DEFAULT_PARALLEL_PARAMS_CSV_FILE_START + "_" +
				(PROCESSOR_NAME.size() > 0 ? PROCESSOR_NAME + "_" : "") + dataTypeStr + "_" + accelStr + CSV_FILE_EXTENSION};
			std::array<std::ofstream, 2> resultsStreamDefaultTBFinal{std::ofstream(OPTIMIZE_PARALLEL_PARAMS ? defaultParamsResultsFileName : optResultsFileName),
															 		 OPTIMIZE_PARALLEL_PARAMS ? std::ofstream(optResultsFileName) : std::ofstream()};
			//get headers from first successful run
			const auto headersInOrder = firstSuccessRun->second.back().getHeadersInOrder();
			for (const auto& currHeader : headersInOrder) {
				resultsStreamDefaultTBFinal[0] << currHeader << ",";
			}
			resultsStreamDefaultTBFinal[0] << std::endl;

			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				for (const auto& currHeader : headersInOrder) {
					resultsStreamDefaultTBFinal[1] << currHeader << ",";
				}
			}
			resultsStreamDefaultTBFinal[1] << std::endl;

			for (unsigned int i=0; i < (OPTIMIZE_PARALLEL_PARAMS ? resultsStreamDefaultTBFinal.size() : 1); i++) {
				for (unsigned int runNum=0; runNum < runOutput.first.size(); runNum++) {
					//if run not successful only have single set of output data from run
					const unsigned int runResultIdx = (runOutput.first[runNum].first == beliefprop::Status::NO_ERROR) ? i : 0;
					for (const auto& currHeader : headersInOrder) {
						if (!(runOutput.first[runNum].second[runResultIdx].isData(currHeader))) {
							resultsStreamDefaultTBFinal[i] << "No Data" << ",";
						}
						else {
							resultsStreamDefaultTBFinal[i] << runOutput.first[runNum].second[runResultIdx].getData(currHeader) << ",";
						}
					}
					resultsStreamDefaultTBFinal[i] << std::endl;
				}
			}

			//write speedup results
			const unsigned int indexBestResults{(OPTIMIZE_PARALLEL_PARAMS ? resultsStreamDefaultTBFinal.size() - 1 : 0)};
			resultsStreamDefaultTBFinal[indexBestResults] << std::endl << ",Average Speedup,Median Speedup" << std::endl;
			for (const auto& speedup : runOutput.second) {
				resultsStreamDefaultTBFinal[indexBestResults] << speedup.first;
				if (speedup.second.first > 0) {
					resultsStreamDefaultTBFinal[indexBestResults] << "," << speedup.second.first << "," << speedup.second.second;
				}
				resultsStreamDefaultTBFinal[indexBestResults] << std::endl;			
			}

			resultsStreamDefaultTBFinal[0].close();
			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				resultsStreamDefaultTBFinal[1].close();
			}

			if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
				std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (optimized parallel parameters) in "
						<< optResultsFileName << std::endl;
				std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run (default parallel parameters) in "
						<< defaultParamsResultsFileName << std::endl;
			}
			else {
				std::cout << "Input stereo set/parameter info, detailed timings, and computed disparity map evaluation for each run using default parallel parameters in "
						<< optResultsFileName << std::endl;
			}
		}
		else {
			std::cout << "Error, no runs completed successfully" << std::endl;
		}
	}

	//perform runs on multiple data sets using specified data type and acceleration method
	template <typename T, beliefprop::AccSetting OPT_IMP_ACCEL>
	std::pair<MultRunData, std::vector<MultRunSpeedup>> runBpOnStereoSets() {
		MultRunData runData;
		runData.push_back(runBpOnSetAndUpdateResults<T, 0, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 0, false, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 1, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 1, false, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 2, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 2, false, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 3, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 3, false, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 4, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 4, false, OPT_IMP_ACCEL>());
#ifndef SMALLER_SETS_ONLY
		runData.push_back(runBpOnSetAndUpdateResults<T, 5, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 5, false, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 6, true, OPT_IMP_ACCEL>());
		runData.push_back(runBpOnSetAndUpdateResults<T, 6, false, OPT_IMP_ACCEL>());
#endif //SMALLER_SETS_ONLY

		//initialize speedup results
		std::vector<MultRunSpeedup> speedupResults;

		//get speedup info for using optimized parallel parameters and disparity count as template parameter
		if (sizeof(T) == sizeof(float)) {
			const auto speedupOverBaseline = RunAndEvaluateBpResults::getAvgMedSpeedupOverBaseline(runData, beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)));
			speedupResults.insert(speedupResults.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
		}
		if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
			speedupResults.push_back(RunAndEvaluateBpResults::getAvgMedSpeedupOptPParams(runData, SPEEDUP_OPT_PAR_PARAMS_HEADER + " - " +
				beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)) + " only"));
		}
		speedupResults.push_back(RunAndEvaluateBpResults::getAvgMedSpeedupDispValsInTemplate(runData, SPEEDUP_DISP_COUNT_TEMPLATE + " - " +
			beliefprop::DATA_SIZE_TO_NAME_MAP.at(sizeof(T)) + " only"));
		
		//write output corresponding to results for current data type
		constexpr bool MULT_DATA_TYPES{false};
		writeRunOutput<OPT_IMP_ACCEL, MULT_DATA_TYPES, T>({runData, speedupResults});

		//return data for each run and multiple average and median speedup results across the data
		return {runData, speedupResults};
	}

	//perform runs without CPU vectorization and get speedup for each run and overall when using vectorization
	//CPU vectorization does not apply to CUDA acceleration so "NO_DATA" output is returned in that case
	template <typename T, beliefprop::AccSetting OPT_IMP_ACCEL>
	std::pair<std::pair<MultRunData, std::vector<MultRunSpeedup>>, MultRunSpeedup> getNoVectDataVectSpeedup(
		const std::string& speedupHeader, MultRunData& runOutput) {
		if constexpr ((OPT_IMP_ACCEL == beliefprop::AccSetting::CUDA) || (OPT_IMP_ACCEL == beliefprop::AccSetting::NONE)) {
			return {std::pair<MultRunData, std::vector<MultRunSpeedup>>(),
					{speedupHeader, {0.0, 0.0}}};
		}
		else {
			auto runOutputNoVect = runBpOnStereoSets<T, beliefprop::AccSetting::NONE>();
			auto speedupWVectorization = RunAndEvaluateBpResults::getAvgMedSpeedup(runOutputNoVect.first, runOutput, speedupHeader);
			return {runOutputNoVect, speedupWVectorization};
		}
	}

	template <beliefprop::AccSetting OPT_IMP_ACCEL>
	void runBpOnStereoSets() {
		//perform runs with and without vectorization using floating point
		//initially store output for floating-point runs separate from output using doubles and halfs
		auto runOutput = runBpOnStereoSets<float, OPT_IMP_ACCEL>();
		//get results and speedup for not using vectorization (only applies to CPU)
		const auto noVectDataVectSpeedupFl = getNoVectDataVectSpeedup<float, OPT_IMP_ACCEL>(SPEEDUP_VECTORIZATION + "- FLOAT only", runOutput.first);
		//get run data portion of results 
		auto runOutputNoVect = noVectDataVectSpeedupFl.first;

		//perform runs with and without vectorization using double-precision
		auto runOutputDouble = runBpOnStereoSets<double, OPT_IMP_ACCEL>();
		const auto doublesSpeedup = RunAndEvaluateBpResults::getAvgMedSpeedup(runOutput.first, runOutputDouble.first, SPEEDUP_DOUBLE);
		const auto noVectDataVectSpeedupDbl = getNoVectDataVectSpeedup<double, OPT_IMP_ACCEL>(SPEEDUP_VECTORIZATION + " - DOUBLE only", runOutputDouble.first);
		for (const auto& runData : noVectDataVectSpeedupDbl.first.first) {
			runOutputNoVect.first.push_back(runData);
		}
#ifdef HALF_PRECISION_SUPPORTED
		//perform runs with and without vectorization using half-precision
		auto runOutputHalf = runBpOnStereoSets<halftype, OPT_IMP_ACCEL>();
		const auto halfSpeedup = RunAndEvaluateBpResults::getAvgMedSpeedup(runOutput.first, runOutputHalf.first, SPEEDUP_HALF);
		const auto noVectDataVectSpeedupHalf = getNoVectDataVectSpeedup<halftype, OPT_IMP_ACCEL>(SPEEDUP_VECTORIZATION + " - HALF only", runOutputHalf.first);
		for (const auto& runData : noVectDataVectSpeedupHalf.first.first) {
			runOutputNoVect.first.push_back(runData);
		}
#endif //HALF_PRECISION_SUPPORTED
		//add output for double and half precision runs to output of floating-point runs to write
		//final output with all data
		runOutput.first.insert(runOutput.first.end(), runOutputDouble.first.begin(), runOutputDouble.first.end());
		runOutput.first.insert(runOutput.first.end(), runOutputHalf.first.begin(), runOutputHalf.first.end());
		//get speedup using vectorization across all runs
		const auto vectorizationSpeedupAll = RunAndEvaluateBpResults::getAvgMedSpeedup(runOutputNoVect.first, runOutput.first, SPEEDUP_VECTORIZATION + " - All Runs");
		//add speedup data from double and half precision runs to overall data so they are included in final results
		runOutput.second.push_back(noVectDataVectSpeedupFl.second);
		runOutput.second.insert(runOutput.second.end(), runOutputDouble.second.begin(), runOutputDouble.second.end());
		runOutput.second.push_back(noVectDataVectSpeedupDbl.second);
#ifdef HALF_PRECISION_SUPPORTED
		runOutput.second.insert(runOutput.second.end(), runOutputHalf.second.begin(), runOutputHalf.second.end());
		runOutput.second.push_back(noVectDataVectSpeedupHalf.second);
#endif //HALF_PRECISION_SUPPORTED

		//get speedup info for using optimized parallel parameters and disparity count as template parameter across all data types
		const auto speedupOverBaseline = RunAndEvaluateBpResults::getAvgMedSpeedupOverBaseline(runOutput.first, "All Runs");
		runOutput.second.insert(runOutput.second.end(), speedupOverBaseline.begin(), speedupOverBaseline.end());
		if constexpr (OPTIMIZE_PARALLEL_PARAMS) {
			runOutput.second.push_back(RunAndEvaluateBpResults::getAvgMedSpeedupOptPParams(runOutput.first, SPEEDUP_OPT_PAR_PARAMS_HEADER + " - All Runs"));
		}
		//add more speedup data to overall data for inclusion in final results
		runOutput.second.push_back(RunAndEvaluateBpResults::getAvgMedSpeedupDispValsInTemplate(runOutput.first, SPEEDUP_DISP_COUNT_TEMPLATE + " - All Runs"));
		runOutput.second.push_back(vectorizationSpeedupAll);
		runOutput.second.push_back(doublesSpeedup);
		runOutput.second.push_back(halfSpeedup);

		//write output corresponding to results for all data types
		constexpr bool MULT_DATA_TYPES{true};
		writeRunOutput<OPT_IMP_ACCEL, MULT_DATA_TYPES>(runOutput);
	}
};

#endif /* RUNANDEVALUATEBPRESULTS_H_ */
