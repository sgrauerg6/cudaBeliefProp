/*
 * EvaluateImpResults.cpp
 *
 *  Created on: Feb 20, 2024
 *      Author: scott
 * 
 *  Function definitions for class to evaluate implementation results.
 */

#include "EvaluateImpResults.h"
#include "EvaluateAcrossRuns.h"
#include <fstream>
#include <numeric>
#include <sstream>

//evaluate results for implementation runs on multiple inputs with all the runs having the same data type and acceleration method
//return run data with speedup from evaluation of implementation runs using multiple inputs with runs
//having the same data type and acceleration method
std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> EvaluateImpResults::EvalResultsSingDataTypeAcc(
  const MultRunData& run_results,
  const run_environment::RunImpSettings run_imp_settings,
  size_t data_size) const
{
  //initialize and add speedup results over baseline data if available for current input
  auto run_imp_opt_results = run_results;
  const auto speedup_over_baseline = GetSpeedupOverBaseline(run_imp_settings, run_imp_opt_results, data_size);
  const auto speedup_over_baseline_subsets = GetSpeedupOverBaselineSubsets(run_imp_settings, run_imp_opt_results, data_size);
  
  //initialize implementation run speedups
  std::vector<RunSpeedupAvgMedian> run_imp_speedups;
  run_imp_speedups.insert(run_imp_speedups.cend(), speedup_over_baseline.cbegin(), speedup_over_baseline.cend());
  run_imp_speedups.insert(run_imp_speedups.cend(), speedup_over_baseline_subsets.cbegin(), speedup_over_baseline_subsets.cend());

  //compute and add speedup info for using optimized parallel parameters and disparity count as template parameter to speedup results
  if (run_imp_settings.opt_parallel_params_setting.first) {
    const std::string speedup_header_optimized_p_params =
      std::string(run_eval::kSpeedupOptParParamsHeader) + " - " +
      std::string(run_environment::kDataSizeToNameMap.at(data_size));
    run_imp_speedups.push_back(GetAvgMedSpeedupOptPParams(
      run_imp_opt_results, speedup_header_optimized_p_params));
  }
  if (run_imp_settings.templated_iters_setting == run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated) {
    const std::string speedup_header_loop_iters_templated =
      std::string(run_eval::kSpeedupLoopItersCountTemplate) + " - " +
      std::string(run_environment::kDataSizeToNameMap.at(data_size));
    run_imp_speedups.push_back(GetAvgMedSpeedupLoopItersInTemplate(
      run_imp_opt_results, speedup_header_loop_iters_templated));
  }

  //return run data with speedup from evaluation of implementation runs using multiple inputs with runs
  //having the same data type and acceleration method
  return {run_imp_opt_results, run_imp_speedups};
}

//evaluate results for all implementation runs on multiple inputs with the runs
//potentially having different data types and acceleration methods and
//write run result and speedup outputs to files
void EvaluateImpResults::EvalAllResultsWriteOutput(
  const std::unordered_map<size_t, MultRunDataWSpeedupByAcc>& run_results_mult_runs,
  const run_environment::RunImpSettings run_imp_settings,
  run_environment::AccSetting opt_imp_acc) const
{
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> run_result_mult_runs_opt =
    run_results_mult_runs;
  //get speedup/slowdown using alternate accelerations
  //and update optimized run result for input with result using alternate acceleration if it is faster than
  //result with "optimal" acceleration for specific input
  std::unordered_map<size_t, std::vector<RunSpeedupAvgMedian>> alt_imp_speedup;
  std::unordered_map<size_t, RunSpeedupAvgMedian> alt_datatype_speedup;
  for (const size_t data_size : run_eval::kDataTypesEvalSizes) {
    alt_imp_speedup[data_size] = GetAltAccelSpeedups(
      run_result_mult_runs_opt[data_size], run_imp_settings, data_size, opt_imp_acc);
    if (data_size != sizeof(float)) {
      //get speedup or slowdown using alternate data type (double or half) compared with float
      alt_datatype_speedup[data_size] = GetAvgMedSpeedup(
        run_result_mult_runs_opt[sizeof(float)][opt_imp_acc].first,
        run_result_mult_runs_opt[data_size][opt_imp_acc].first,
        (data_size > sizeof(float)) ? run_eval::kSpeedupDouble : run_eval::kSpeedupHalf);
    }
  }

  //initialize overall results to float results using fastest acceleration and add double and half-type results to it
  auto results_w_speedups = run_result_mult_runs_opt[sizeof(float)][opt_imp_acc];
  if (run_result_mult_runs_opt.contains(sizeof(double))) {
    results_w_speedups.first.insert(results_w_speedups.first.cend(),
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].first.cbegin(),
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].first.cend());
  }
  if (run_result_mult_runs_opt.contains(sizeof(halftype))) {
    results_w_speedups.first.insert(results_w_speedups.first.cend(),
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].first.cbegin(),
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].first.cend());
  }

  //add speedup data from double and half precision runs to speedup results
  results_w_speedups.second.insert(results_w_speedups.second.cend(),
    alt_imp_speedup[sizeof(float)].cbegin(),
    alt_imp_speedup[sizeof(float)].cend());
  if (run_result_mult_runs_opt.contains(sizeof(double))) {
    results_w_speedups.second.insert(results_w_speedups.second.cend(),
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].second.cbegin(),
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].second.cend());
    results_w_speedups.second.insert(results_w_speedups.second.cend(), 
      alt_imp_speedup[sizeof(double)].cbegin(),
      alt_imp_speedup[sizeof(double)].cend());
  }
  if (run_result_mult_runs_opt.contains(sizeof(halftype))) {
    results_w_speedups.second.insert(results_w_speedups.second.cend(),
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].second.cbegin(),
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].second.cend());
    results_w_speedups.second.insert(results_w_speedups.second.cend(),
      alt_imp_speedup[sizeof(halftype)].cbegin(),
      alt_imp_speedup[sizeof(halftype)].cend());
  }

  //get speedup over baseline runtimes...can only compare with baseline runtimes that are
  //generated using same templated iterations setting as current run
  if ((run_imp_settings.base_opt_single_thread_runtime_for_template_setting) &&
      (run_imp_settings.base_opt_single_thread_runtime_for_template_setting.value().second == run_imp_settings.templated_iters_setting)) {
      const auto speedup_over_baseline = GetAvgMedSpeedupOverBaseline(
        results_w_speedups.first, run_eval::kAllRunsStr,
        run_imp_settings.base_opt_single_thread_runtime_for_template_setting.value().first);
      results_w_speedups.second.insert(
        results_w_speedups.second.cend(),
        speedup_over_baseline.cbegin(),
        speedup_over_baseline.cend());
  }

  //get speedup info for using optimized parallel parameters
  if (run_imp_settings.opt_parallel_params_setting.first) {
    results_w_speedups.second.push_back(
      GetAvgMedSpeedupOptPParams(
        results_w_speedups.first,
        std::string(run_eval::kSpeedupOptParParamsHeader) + " - " + std::string(run_eval::kAllRunsStr)));
  }

  //get speedup when using templated number for loop iteration count
  if (run_imp_settings.templated_iters_setting == run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated) {
    results_w_speedups.second.push_back(
      GetAvgMedSpeedupLoopItersInTemplate(
        results_w_speedups.first,
        std::string(run_eval::kSpeedupLoopItersCountTemplate) + " - " + std::string(run_eval::kAllRunsStr)));
  }

  //add speedups when using doubles and half precision compared to float to end of speedup data
  //if speedup data exists
  for (const auto& alt_speedup : alt_datatype_speedup) {
    results_w_speedups.second.push_back(alt_speedup.second);
  }

  //write output corresponding to results and speedups for all data types
  WriteRunOutput(results_w_speedups, run_imp_settings, opt_imp_acc);
}

//write data for file corresponding to runs for a specified data type or across all data type
//includes results for each run as well as average and median speedup data across multiple runs
void EvaluateImpResults::WriteRunOutput(
  const std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>& run_output,
  const run_environment::RunImpSettings& run_imp_settings,
  run_environment::AccSetting acceleration_setting) const
{
  //get iterator to first run with success
  const auto first_success_run =
    std::find_if(run_output.first.cbegin(), run_output.first.cend(),
      [](const auto& run_result) { return run_result; } );

  //check if there was at least one successful run
  if (first_success_run != run_output.first.cend()) {
    //write results from default and optimized parallel parameters runs to csv file
    //file name contains info about data type, parameter settings, and processor name if available
    //only show data type string and acceleration string for runs using a single data type that are
    //used for debugging (not multidata type results) 
    //get directory to store implementation results for specific implementation
    const auto imp_results_fp = GetImpResultsPath();
    //create any results directories if needed
    if (!(std::filesystem::is_directory(imp_results_fp / run_eval::kImpResultsRunDataFolderName))) {
      std::filesystem::create_directory(imp_results_fp / run_eval::kImpResultsRunDataFolderName);
    }
    if (!(std::filesystem::is_directory(imp_results_fp / run_eval::kImpResultsRunDataWSpeedupsFolderName))) {
      std::filesystem::create_directory(imp_results_fp / run_eval::kImpResultsRunDataWSpeedupsFolderName);
    }
    if (!(std::filesystem::is_directory(imp_results_fp / run_eval::kImpResultsSpeedupsFolderName))) {
      std::filesystem::create_directory(imp_results_fp / run_eval::kImpResultsSpeedupsFolderName);
    }
    //get file paths for run result and speedup files for implementation run
    const std::filesystem::path opt_results_file_path{imp_results_fp / run_eval::kImpResultsRunDataFolderName /
      std::filesystem::path(((run_imp_settings.run_name) ? std::string(run_imp_settings.run_name.value()) + "_" : "") + 
      std::string(run_eval::kRunResultsDescFileName) + std::string(run_eval::kCsvFileExtension))};
    const std::filesystem::path opt_results_w_speedup_file_path{imp_results_fp / run_eval::kImpResultsRunDataWSpeedupsFolderName /
      std::filesystem::path(((run_imp_settings.run_name) ? std::string(run_imp_settings.run_name.value()) + "_" : "") + 
      std::string(run_eval::kRunResultsWSpeedupsDescFileName) + std::string(run_eval::kCsvFileExtension))};
    const std::filesystem::path default_params_results_file_path{imp_results_fp / run_eval::kImpResultsRunDataFolderName /
      std::filesystem::path(((run_imp_settings.run_name) ? std::string(run_imp_settings.run_name.value()) + "_" : "") + 
      std::string(run_eval::kRunResultsDescDefaultPParamsFileName) + std::string(run_eval::kCsvFileExtension))};
    const std::filesystem::path speedups_results_file_path{imp_results_fp / run_eval::kImpResultsSpeedupsFolderName /
      std::filesystem::path(((run_imp_settings.run_name) ? std::string(run_imp_settings.run_name.value()) + "_" : "") + 
      std::string(run_eval::kSpeedupsDescFileName) + std::string(run_eval::kCsvFileExtension))};
    std::array<std::ostringstream, 2> run_data_opt_default_sstr;
    std::array<std::ostringstream, 2> speedups_headers_left_top_sstr;

    //get headers from first successful run and write headers to top of output files
    const auto headers_in_order = (*first_success_run)->back().HeadersInOrder();
    for (unsigned int i=0; i < (run_imp_settings.opt_parallel_params_setting.first ? run_data_opt_default_sstr.size() : 1); i++) {
      for (const auto& curr_header : headers_in_order) {
        run_data_opt_default_sstr[i] << curr_header << ',';
      }
      run_data_opt_default_sstr[i] << std::endl;
    }

    //write output for run on each input with each data type
    for (unsigned int i=0; i < (run_imp_settings.opt_parallel_params_setting.first ? run_data_opt_default_sstr.size() : 1); i++) {
      for (unsigned int run_num=0; run_num < run_output.first.size(); run_num++) {
        //if run not successful only have single set of output data from run
        const unsigned int run_result_idx = run_output.first[run_num] ? i : 0;
        for (const auto& curr_header : headers_in_order) {
          if (!(run_output.first[run_num]->at(run_result_idx).IsData(curr_header))) {
            run_data_opt_default_sstr[i] << "No Data" << ',';
          }
          else {
            run_data_opt_default_sstr[i] << run_output.first[run_num]->at(run_result_idx).GetDataAsStr(curr_header) << ',';
          }
        }
        run_data_opt_default_sstr[i] << std::endl;
      }
    }

    //generate speedup results with headers on left side and with headers on top row
    speedups_headers_left_top_sstr[0] << "Speedup Results,Average Speedup,Median Speedup" << std::endl;
    for (const auto& speedup : run_output.second) {
      speedups_headers_left_top_sstr[0] << speedup.first;
      if (speedup.second[0] > 0) {
        speedups_headers_left_top_sstr[0] << ',' << speedup.second[0] << ',' << speedup.second[1];
      }
      speedups_headers_left_top_sstr[0] << std::endl;
    }
    speedups_headers_left_top_sstr[1] << ',';
    for (const auto& speedup : run_output.second) {
      speedups_headers_left_top_sstr[1] << speedup.first << ',';
    }
    for (const auto& speedup_desc_w_index : {std::pair<std::string_view, size_t>{"Average Speedup", 0},
                                             std::pair<std::string_view, size_t>{"Median Speedup", 1}}) {
      speedups_headers_left_top_sstr[1] << std::endl << speedup_desc_w_index.first << ',';
      for (const auto& speedup : run_output.second) {
        speedups_headers_left_top_sstr[1] << speedup.second[speedup_desc_w_index.second] << ',';
      }
    }
    
    //write run results strings to output streams
    //one results file contains only speedup results, another contains only run results,
    //and a third contains run results followed by speedups
    std::ofstream speedup_results_str{speedups_results_file_path};
    speedup_results_str << speedups_headers_left_top_sstr[1].str();
    std::array<std::ofstream, 2> results_stream_default_tb_final{
      (run_imp_settings.opt_parallel_params_setting.first ?
        std::ofstream() : std::ofstream(opt_results_file_path)),
      run_imp_settings.opt_parallel_params_setting.first ? std::ofstream(opt_results_file_path) : std::ofstream()};
    std::ofstream run_result_w_speedup_sstr(opt_results_w_speedup_file_path);
    if (!run_imp_settings.opt_parallel_params_setting.first) {
      //only write file with default params when there are not results with optimized params
      results_stream_default_tb_final[0] << run_data_opt_default_sstr[0].str();
    }
    if (run_imp_settings.opt_parallel_params_setting.first) {
      results_stream_default_tb_final[1] << run_data_opt_default_sstr[1].str();
      run_result_w_speedup_sstr << run_data_opt_default_sstr[1].str() << std::endl;
    }
    else {
      run_result_w_speedup_sstr << run_data_opt_default_sstr[0].str() << std::endl;        
    }
    //add speedups with headers on left to file containing run results and speedups
    run_result_w_speedup_sstr << speedups_headers_left_top_sstr[0].str();

    std::cout << "Input/settings/parameters info, detailed timings, and evaluation for each run and across runs in "
              << opt_results_w_speedup_file_path << std::endl;
    std::cout << "Run inputs and results in " << opt_results_file_path << std::endl;
    std::cout << "Speedup results in " << speedups_results_file_path << std::endl;

    //get vector of speedup headers to use for evaluation across runs
    std::vector<std::string> speedup_headers;
    for (const auto& speedup_header_data : run_output.second) {
      speedup_headers.push_back(speedup_header_data.first);
    }
    EvaluateAcrossRuns().operator()(
      imp_results_fp, GetCombResultsTopText(), GetInputParamsShow(), speedup_headers);
  }
  else {
    std::cout << "Error, no runs completed successfully" << std::endl;
  }
}

//process results for runs with alternate acceleration from optimal acceleration and
//get speedup for each run and overall when using optimal acceleration compared to alternate accelerations
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetAltAccelSpeedups(
  MultRunDataWSpeedupByAcc& run_imp_results_by_acc_setting,
  const run_environment::RunImpSettings& run_imp_settings,
  size_t data_type_size,
  run_environment::AccSetting fastest_acc) const
{
  //set up mapping from acceleration type to description
  const std::map<run_environment::AccSetting, std::string> acc_to_speedup_str{
    {run_environment::AccSetting::kNone, 
     std::string(run_eval::kSpeedupCPUVectorization) + " - " + std::string(run_environment::kDataSizeToNameMap.at(data_type_size))},
    {run_environment::AccSetting::kAVX256, 
     std::string(run_eval::kSpeedupVsAvx256Vectorization) + " - " + std::string(run_environment::kDataSizeToNameMap.at(data_type_size))}};

  if (run_imp_results_by_acc_setting.size() == 1) {
    //no alternate run results
    return {{acc_to_speedup_str.at(run_environment::AccSetting::kNone), {0.0, 0.0}},
            {acc_to_speedup_str.at(run_environment::AccSetting::kAVX256), {0.0, 0.0}}};
  }
  else {
    //initialize speedup/slowdown using alternate acceleration
    std::vector<RunSpeedupAvgMedian> alt_acc_speedups;
    for (auto& alt_acc_imp_results : run_imp_results_by_acc_setting) {
      if ((alt_acc_imp_results.first != fastest_acc) && (acc_to_speedup_str.contains(alt_acc_imp_results.first))) {
        //process results using alternate acceleration
        //go through each result and replace initial run data with alternate implementation run data if alternate implementation run is faster
        for (unsigned int i = 0; i < run_imp_results_by_acc_setting[fastest_acc].first.size(); i++) {
          if (run_imp_results_by_acc_setting[fastest_acc].first[i] && alt_acc_imp_results.second.first[i]) {
            const double init_result_time =
              run_imp_results_by_acc_setting[fastest_acc].first[i]->back().GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value();
            const double alt_acc_result_time =
              alt_acc_imp_results.second.first[i]->back().GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value();
            if (alt_acc_result_time < init_result_time) {
              run_imp_results_by_acc_setting[fastest_acc].first[i] = alt_acc_imp_results.second.first[i];
            }
          }
        }
        //get speedup/slowdown using alternate acceleration compared to fastest implementation and store in speedup results
        alt_acc_speedups.push_back(GetAvgMedSpeedup(alt_acc_imp_results.second.first, run_imp_results_by_acc_setting[fastest_acc].first,
          acc_to_speedup_str.at(alt_acc_imp_results.first)));
      }
    }
    return alt_acc_speedups;
  }
}

//get speedup over baseline data if data available
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetSpeedupOverBaseline(
  const run_environment::RunImpSettings& run_imp_settings,
  MultRunData& run_data_all_runs, size_t data_type_size) const
{
  //initialize speedup results
  std::vector<RunSpeedupAvgMedian> speedup_results;

  //only get speedup over baseline when processing float data type since that is run first and corresponds to the data at the top
  //of the baseline data
  if (data_type_size == sizeof(float)) {
    //get speedup over baseline runtimes...can only compare with baseline runtimes that are
    //generated using same templated iterations setting as current run
    if ((run_imp_settings.base_opt_single_thread_runtime_for_template_setting) &&
        (run_imp_settings.base_opt_single_thread_runtime_for_template_setting.value().second == run_imp_settings.templated_iters_setting))
    {
      const auto speedup_over_baseline_subsets = GetAvgMedSpeedupOverBaseline(
        run_data_all_runs, run_environment::kDataSizeToNameMap.at(data_type_size),
        run_imp_settings.base_opt_single_thread_runtime_for_template_setting.value().first);
      speedup_results.insert(speedup_results.cend(), speedup_over_baseline_subsets.cbegin(), speedup_over_baseline_subsets.cend());
    }
  }
  return speedup_results;
}

//get speedup over baseline run for subsets of smallest and largest sets if data available
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetSpeedupOverBaselineSubsets(
  const run_environment::RunImpSettings& run_imp_settings,
  MultRunData& run_data_all_runs,
  size_t data_type_size) const
{
  if ((data_type_size == sizeof(float)) &&
      (run_imp_settings.base_opt_single_thread_runtime_for_template_setting && 
        (run_imp_settings.base_opt_single_thread_runtime_for_template_setting->second == run_imp_settings.templated_iters_setting)))
  {
    return GetAvgMedSpeedupOverBaselineSubsets(run_data_all_runs, run_environment::kDataSizeToNameMap.at(data_type_size),
      run_imp_settings.base_opt_single_thread_runtime_for_template_setting->first, run_imp_settings.subset_str_indices);
  }
  //return empty vector if doesn't match settings to get speedup over baseline for subsets
  return std::vector<RunSpeedupAvgMedian>();
}

//get baseline runtime data if available...return null if baseline data not available
std::optional<std::pair<std::string, std::vector<double>>> EvaluateImpResults::GetBaselineRuntimeData(
  std::string_view baseline_data_path) const
{
  std::ifstream baseline_data_stream(std::string{baseline_data_path});
  if (!(baseline_data_stream.is_open())) {
    return {};
  }
  std::string line;
  //first line of data is string with baseline processor description and all subsequent data is runtimes
  //on that processor in same order as runtimes from runBenchmark() function
  std::string baseline_name;
  std::vector<double> baseline_data;
  bool first_line{true};
  while (std::getline(baseline_data_stream, line)) {
    if (first_line) {
      baseline_name = line;
      first_line = false;
    }
    else {
      baseline_data.push_back(std::stod(line));
    }
  }

  return std::pair<std::string, std::vector<double>>{
    baseline_name, baseline_data};
}

//get average and median speedup from vector of speedup values
std::array<double, 2> EvaluateImpResults::GetAvgMedSpeedup(const std::vector<double>& speedups_vect) const {
  const double average_speedup = (std::accumulate(speedups_vect.cbegin(), speedups_vect.cend(), 0.0) / (double)speedups_vect.size());
  auto speedups_vect_sorted = speedups_vect;
  std::ranges::sort(speedups_vect_sorted);
  const double median_speedup = ((speedups_vect_sorted.size() % 2) == 0) ? 
    (speedups_vect_sorted[(speedups_vect_sorted.size() / 2) - 1] + speedups_vect_sorted[(speedups_vect_sorted.size() / 2)]) / 2.0 : 
    speedups_vect_sorted[(speedups_vect_sorted.size() / 2)];
  return {average_speedup, median_speedup};
}

//get average and median speedup of specified subset(s) of runs compared to baseline data from file
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetAvgMedSpeedupOverBaselineSubsets(MultRunData& run_output,
  std::string_view data_type_str, const std::array<std::string_view, 2>& base_data_path_opt_single_thread,
  const std::vector<std::pair<std::string, std::vector<unsigned int>>>& subset_str_indices) const
{
  //get speedup over baseline for optimized runs
  std::vector<RunSpeedupAvgMedian> speedup_data;
  const auto baseline_run_data = GetBaselineRuntimeData(base_data_path_opt_single_thread[0]);
  if (baseline_run_data) {
    const auto baseline_runtimes = (*baseline_run_data).second;
    //retrieve speedup data for any subsets of optimized runs
    for (const auto& curr_subset_str_indices : subset_str_indices) {
      std::vector<double> speedups_vect;
      const std::string speedup_header = "Speedup relative to " + std::string((*baseline_run_data).first) + " on " +
        std::string(curr_subset_str_indices.first) + " - " + std::string(data_type_str);
      for (unsigned int i : curr_subset_str_indices.second) {
        if (run_output[i]) {
          speedups_vect.push_back(
            baseline_runtimes[i] / 
            run_output[i]->at(1).GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
          for (auto& run_data : run_output[i].value()) {
            run_data.AddDataWHeader(std::string(speedup_header), speedups_vect.back());
          }
        }
      }
      if (!(speedups_vect.empty())) {
        speedup_data.push_back({speedup_header, GetAvgMedSpeedup(speedups_vect)});
      }
    }
  }

  return speedup_data;
}

//get average and median speedup of current runs compared to baseline data from file
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetAvgMedSpeedupOverBaseline(MultRunData& run_output,
  std::string_view data_type_str, const std::array<std::string_view, 2>& baseline_path_opt_single_thread) const
{
  //get speedup over baseline for optimized runs
  std::vector<RunSpeedupAvgMedian> speedup_data;
  const auto baseline_run_data = GetBaselineRuntimeData(baseline_path_opt_single_thread[0]);
  if (baseline_run_data) {
    std::vector<double> speedups_vect;
    const std::string speedup_header = "Speedup relative to " + baseline_run_data->first + " - " + std::string(data_type_str);
    const auto baseline_runtimes = (*baseline_run_data).second;
    for (unsigned int i=0; i < run_output.size(); i++) {
      if (run_output[i]) {
        speedups_vect.push_back(
          baseline_runtimes[i] / run_output[i]->at(1).GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
        for (auto& run_data : run_output[i].value()) {
          run_data.AddDataWHeader(speedup_header, speedups_vect.back());
        }
      }
    }
    if (!(speedups_vect.empty())) {
      speedup_data.push_back({speedup_header, GetAvgMedSpeedup(speedups_vect)});
    }
  }

  //get speedup over baseline for single thread runs
  const auto baseline_run_data_single_thread = GetBaselineRuntimeData(baseline_path_opt_single_thread[1]);
  if (baseline_run_data_single_thread) {
    std::vector<double> speedups_vect;
    const std::string speedup_header = "Single-Thread (Orig Imp) speedup relative to " +
      std::string((*baseline_run_data_single_thread).first) + " - " + std::string(data_type_str);
    const auto baseline_runtimesSThread = (*baseline_run_data_single_thread).second;
    for (unsigned int i=0; i < run_output.size(); i++) {
      if (run_output[i]) {
        speedups_vect.push_back(baseline_runtimesSThread[i] /
          run_output[i]->at(1).GetDataAsDouble(run_eval::kSingleThreadRuntimeHeader).value());
        for (auto& run_data : run_output[i].value()) {
          run_data.AddDataWHeader(speedup_header, speedups_vect.back());
        }
      }
    }
    if (!(speedups_vect.empty())) {
      speedup_data.push_back({speedup_header, GetAvgMedSpeedup(speedups_vect)});
    }
  }

  return speedup_data;
}

//get average and median speedup using optimized parallel parameters compared to default parallel parameters
//and also add speedup for each run using optimized parallel parameters compared to each run with default
//parallel parameters
RunSpeedupAvgMedian EvaluateImpResults::GetAvgMedSpeedupOptPParams(
  MultRunData& run_output, std::string_view speedup_header) const
{
  std::vector<double> speedups_vect;
  for (unsigned int i=0; i < run_output.size(); i++) {
    if (run_output[i]) {
      speedups_vect.push_back(run_output[i]->at(0).GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value() / 
                              run_output[i]->at(1).GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
      run_output[i]->at(0).AddDataWHeader(std::string(speedup_header), speedups_vect.back());
      run_output[i]->at(1).AddDataWHeader(std::string(speedup_header), speedups_vect.back());
    }
  }
  if (!(speedups_vect.empty())) {
    return {std::string(speedup_header), GetAvgMedSpeedup(speedups_vect)};
  }
  return {std::string(speedup_header), {0.0, 0.0}};
}

//get average and median speedup between base and target runtime data and also add
//speedup for each target runtime data run as compared to corresponding base run
RunSpeedupAvgMedian EvaluateImpResults::GetAvgMedSpeedup(
  MultRunData& run_output_base,
  MultRunData& run_output_target,
  std::string_view speedup_header) const
{
  std::vector<double> speedups_vect;
  for (unsigned int i=0; i < run_output_base.size(); i++) {
    if (run_output_base[i] && run_output_target[i]) {
      speedups_vect.push_back(run_output_base[i]->back().GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value() / 
                             run_output_target[i]->back().GetDataAsDouble(run_eval::kOptimizedRuntimeHeader).value());
      run_output_base[i]->at(1).AddDataWHeader(std::string(speedup_header), speedups_vect.back());
      run_output_target[i]->at(1).AddDataWHeader(std::string(speedup_header), speedups_vect.back());
    }
  }
  if (!(speedups_vect.empty())) {
    return {std::string(speedup_header), GetAvgMedSpeedup(speedups_vect)};
  }
  return {std::string(speedup_header), {0.0, 0.0}};
}

//get average and median speedup when loop iterations are given at compile time as template value
//and also add speedup for each run with templated loop iterations as compared to same run without
//templated loop iterations
RunSpeedupAvgMedian EvaluateImpResults::GetAvgMedSpeedupLoopItersInTemplate(MultRunData& run_output,
  std::string_view speedup_header) const
{
  //get mapping of run inputs to runtime with index value in run output
  std::map<std::vector<std::string>, std::pair<double, size_t>> run_input_settings_to_time_w_idx;
  for (unsigned int i=0; i < run_output.size(); i++) {
    if (run_output[i]) {
      const auto input_settings_to_time = run_output[i]->back().GetParamsToRuntime(
        std::vector<std::string_view>(run_eval::kRunInputSigHeaders.cbegin(), run_eval::kRunInputSigHeaders.cend()),
        run_eval::kOptimizedRuntimeHeader);
      if (input_settings_to_time) {
        run_input_settings_to_time_w_idx.insert({input_settings_to_time->first, {input_settings_to_time->second, i}});
      }
    }
  }
  //go through all run input settings to time and get each pair that is the same in datatype and input and
  //differs in disp values templated and get speedup for each of templated compared to non-templated
  std::vector<double> speedups_vect;
  auto run_data_iter = run_input_settings_to_time_w_idx.cbegin();
  while (run_data_iter != run_input_settings_to_time_w_idx.cend()) {
    auto data_type_run = run_data_iter->first[run_eval::kRunInputDatatypeIdx];
    auto input_idx_run = run_data_iter->first[run_eval::kRunInputNumInputIdx];
    auto run_comp_1 = run_data_iter;
    auto run_comp_2 = run_data_iter;
    //find run with same datatype and input index
    while (++run_data_iter != run_input_settings_to_time_w_idx.cend()) {
      if ((run_data_iter->first[run_eval::kRunInputDatatypeIdx] == data_type_run) &&
          (run_data_iter->first[run_eval::kRunInputNumInputIdx] == input_idx_run))
      {
        run_comp_2 = run_data_iter;
        break;
      }
    }
    //if don't have two separate runs with same data type and input, erase current run from mapping and continue
    if (run_comp_1 == run_comp_2) {
      run_input_settings_to_time_w_idx.erase(run_comp_1);
      run_data_iter = run_input_settings_to_time_w_idx.cbegin();
      continue;
    }
    //retrieve which run data uses templated iteration count and which one doesn't and get speedup
    //add speedup to speedup vector and also to run data of run with templated iteration count
    if ((run_comp_1->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[1]) &&
        (run_comp_2->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[0]))
    {
      speedups_vect.push_back(run_comp_2->second.first / run_comp_1->second.first);
      run_output[run_comp_1->second.second]->back().AddDataWHeader(std::string(speedup_header), speedups_vect.back());
    }
    else if ((run_comp_1->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[0]) &&
             (run_comp_2->first[run_eval::kRunInputLoopItersTemplatedIdx] == run_eval::kBoolValFalseTrueDispStr[1]))
    {
      speedups_vect.push_back(run_comp_1->second.first / run_comp_2->second.first);
      run_output[run_comp_2->second.second]->back().AddDataWHeader(std::string(speedup_header), speedups_vect.back());
    }
    //remove runs that have been processed from mapping
    run_input_settings_to_time_w_idx.erase(run_comp_1);
    run_input_settings_to_time_w_idx.erase(run_comp_2);
    run_data_iter = run_input_settings_to_time_w_idx.cbegin();
  }
  if (!(speedups_vect.empty())) {
    return {std::string(speedup_header), GetAvgMedSpeedup(speedups_vect)};
  }
  return {std::string(speedup_header), {0.0, 0.0}};
}