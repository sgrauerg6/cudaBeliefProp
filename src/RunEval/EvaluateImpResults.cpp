/*
Copyright (C) 2024 Scott Grauer-Gray

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307 USA
*/

/**
 * @file EvaluateImpResults.cpp
 * @author Scott Grauer-Gray
 * @brief Function definitions for class to evaluate implementation results.
 * 
 * @copyright Copyright (c) 2024
 */

#include <sstream>
#include <numeric>
#include <fstream>
#include <algorithm>
#include "EvaluateAcrossRuns.h"
#include "RunResultsSpeedups.h"
#include "EvaluateImpResults.h"

//evaluate results for implementation runs on multiple inputs with all the runs
//having the same data type and acceleration method
//return run data with speedup from evaluation of implementation runs using
//multiple inputs with runs having the same data type and acceleration method
std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> EvaluateImpResults::EvalResultsSingDataTypeAcc(
  const MultRunData& run_results,
  const run_environment::RunImpSettings run_imp_settings,
  size_t data_size) const
{
  //store whether or not parallel parameters optimized in run
  const bool p_params_optimized{
    (!(run_imp_settings.p_params_default_alt_options.second.empty()))};

  //initialize and add speedup results over baseline data if available for current input
  auto run_imp_opt_results = run_results;
  const auto speedup_over_baseline = 
    GetSpeedupOverBaseline(run_imp_settings, run_imp_opt_results, data_size);
  const auto speedup_over_baseline_subsets =
    GetSpeedupOverBaselineSubsets(run_imp_settings, run_imp_opt_results, data_size);

  //initialize implementation run speedups
  std::vector<RunSpeedupAvgMedian> run_imp_speedups;
  run_imp_speedups.insert(run_imp_speedups.cend(),
                          speedup_over_baseline.cbegin(),
                          speedup_over_baseline.cend());
  run_imp_speedups.insert(run_imp_speedups.cend(),
                          speedup_over_baseline_subsets.cbegin(),
                          speedup_over_baseline_subsets.cend());

  //compute and add speedup info for using optimized parallel parameters
  //compared to default parallel parameters
  if (p_params_optimized) {
    const std::string speedup_header_optimized_p_params =
      std::string(run_eval::kSpeedupOptParParamsHeader) + " - " +
      std::string(run_environment::kDataSizeToNameMap.at(data_size));
    run_imp_speedups.push_back(GetAvgMedSpeedupOptPParams(
      run_imp_opt_results, speedup_header_optimized_p_params));
  }

  //compute and add speedup info for using templated loop iteration counts compared to
  //loop iteration counts not being known at compile time
  if (run_imp_settings.templated_iters_setting ==
      run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated)
  {
    const std::string speedup_header_loop_iters_templated =
      std::string(run_eval::kSpeedupLoopItersCountTemplate) + " - " +
      std::string(run_environment::kDataSizeToNameMap.at(data_size));
    run_imp_speedups.push_back(GetAvgMedSpeedupLoopItersInTemplate(
      run_imp_opt_results, speedup_header_loop_iters_templated));
  }

  //return run data with speedup from evaluation of implementation runs using
  //multiple inputs with runs having the same data type and acceleration method
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
  //store whether or not parallel parameters optimized in run
  const bool p_params_optimized{
    (!(run_imp_settings.p_params_default_alt_options.second.empty()))};

  //initialize optimized multi-run results from input run result
  //run results must be writable since speedup results get added to runs and
  //optimized run results may be adjusted if faster runs found in alternative
  //implementations compared to expected fastest implementation
  std::unordered_map<size_t, MultRunDataWSpeedupByAcc> run_result_mult_runs_opt =
    run_results_mult_runs;

  //get speedup/slowdown using alternate accelerations and update optimized run
  //result for input with result using alternate acceleration if it is faster than
  //result with "optimal" acceleration for specific input
  std::unordered_map<size_t, std::vector<RunSpeedupAvgMedian>> alt_imp_speedup;
  //check if setting is to run alternate optimized implementations and run and
  //evaluate alternate optimized implementations if that's the case
  if (run_imp_settings.run_alt_optimized_imps) {
    for (const size_t data_size : run_imp_settings.datatypes_eval_sizes) {
      alt_imp_speedup[data_size] = GetAltAccelSpeedups(
        run_result_mult_runs_opt[data_size],
        run_imp_settings, data_size,
        opt_imp_acc);
    }
  }

  //get speedup/slowdown using alternate datatype (double or half) compared
  //with float
  std::unordered_map<size_t, RunSpeedupAvgMedian> alt_datatype_speedup;
  for (const size_t data_size : run_imp_settings.datatypes_eval_sizes) {
    if (data_size != sizeof(float)) {
      //get speedup or slowdown using alternate data type (double or half) compared with float
      alt_datatype_speedup[data_size] = GetAvgMedSpeedupBaseVsTarget(
        run_result_mult_runs_opt[sizeof(float)][opt_imp_acc].first,
        run_result_mult_runs_opt[data_size][opt_imp_acc].first,
        (data_size > sizeof(float)) ? run_eval::kSpeedupDouble : run_eval::kSpeedupHalf,
        BaseTargetDiff::kDiffDatatype);
    }
  }
  //initialize overall results to float results using fastest acceleration and
  //add double and half-type results to it
  auto [run_results, run_speedups] =
    run_result_mult_runs_opt[sizeof(float)][opt_imp_acc];
  if (run_result_mult_runs_opt.contains(sizeof(double))) {
    run_results.merge(
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].first);
  }
  if (run_result_mult_runs_opt.contains(sizeof(halftype))) {
    run_results.merge(
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].first);
  }

  //add speedup data from double and half precision runs to speedup results
  run_speedups.insert(run_speedups.cend(),
    alt_imp_speedup[sizeof(float)].cbegin(),
    alt_imp_speedup[sizeof(float)].cend());
  if (run_result_mult_runs_opt.contains(sizeof(double))) {
    run_speedups.insert(run_speedups.cend(),
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].second.cbegin(),
      run_result_mult_runs_opt[sizeof(double)][opt_imp_acc].second.cend());
    run_speedups.insert(run_speedups.cend(), 
      alt_imp_speedup[sizeof(double)].cbegin(),
      alt_imp_speedup[sizeof(double)].cend());
  }
  if (run_result_mult_runs_opt.contains(sizeof(halftype))) {
    run_speedups.insert(run_speedups.cend(),
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].second.cbegin(),
      run_result_mult_runs_opt[sizeof(halftype)][opt_imp_acc].second.cend());
    run_speedups.insert(run_speedups.cend(),
      alt_imp_speedup[sizeof(halftype)].cbegin(),
      alt_imp_speedup[sizeof(halftype)].cend());
  }

  //get speedup over baseline runtimes
  if (run_imp_settings.baseline_runtimes_path_desc)
  {
    const auto speedup_over_baseline = GetAvgMedSpeedupOverBaseline(
      run_results, run_eval::kAllRunsStr,
      *run_imp_settings.baseline_runtimes_path_desc);
    run_speedups.insert(
      run_speedups.cend(),
      speedup_over_baseline.cbegin(),
      speedup_over_baseline.cend());
  }

  //get speedup info for using optimized parallel parameters
  if (p_params_optimized) {
    run_speedups.push_back(
      GetAvgMedSpeedupOptPParams(
        run_results,
        std::string(run_eval::kSpeedupOptParParamsHeader) + " - " + std::string(run_eval::kAllRunsStr)));
  }

  //get speedup when using templated number for loop iteration count
  if (run_imp_settings.templated_iters_setting ==
      run_environment::TemplatedItersSetting::kRunTemplatedAndNotTemplated)
  {
    run_speedups.push_back(
      GetAvgMedSpeedupLoopItersInTemplate(
        run_results,
        std::string(run_eval::kSpeedupLoopItersCountTemplate) + " - " + std::string(run_eval::kAllRunsStr)));
  }

  //add speedups when using doubles and half precision compared to float to end of speedup data
  //if speedup data exists
  for (const auto& alt_speedup : alt_datatype_speedup) {
    run_speedups.push_back(alt_speedup.second);
  }

  //write output corresponding to results and run_speedups for all data types
  WriteRunOutput({run_results, run_speedups}, run_imp_settings, opt_imp_acc);
}

//write data for file corresponding to runs for a specified data type or across all data type
//includes results for each run as well as average and median speedup data across multiple runs
void EvaluateImpResults::WriteRunOutput(
  const std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>& run_results_w_speedups,
  const run_environment::RunImpSettings& run_imp_settings,
  run_environment::AccSetting acceleration_setting) const
{
  //get references to run results and speedups
  const auto& [run_results, speedup_headers_w_data] = run_results_w_speedups;

  //store whether or not parallel parameters optimized in run
  const bool p_params_optimized{
    (!(run_imp_settings.p_params_default_alt_options.second.empty()))};

  //get iterator to first run with success
  //run_result corresponds to a std::optional object that contains run data
  //if run successful and returns false if no data (indicating run not successful)
  const auto first_success_run_iter =
    std::find_if(run_results.cbegin(), run_results.cend(),
      [](const auto& run_result) { return run_result.second; } );

  //check if there was at least one successful run
  if (first_success_run_iter != run_results.cend())
  {
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
    if (!(std::filesystem::is_directory(imp_results_fp / run_eval::kImpResultsRunDataDefaultPParamsFolderName))) {
      std::filesystem::create_directory(imp_results_fp / run_eval::kImpResultsRunDataDefaultPParamsFolderName);
    }
    if (!(std::filesystem::is_directory(imp_results_fp / run_eval::kImpResultsRunDataWSpeedupsFolderName))) {
      std::filesystem::create_directory(imp_results_fp / run_eval::kImpResultsRunDataWSpeedupsFolderName);
    }
    if (!(std::filesystem::is_directory(imp_results_fp / run_eval::kImpResultsSpeedupsFolderName))) {
      std::filesystem::create_directory(imp_results_fp / run_eval::kImpResultsSpeedupsFolderName);
    }

    //get file paths for run result and speedup files for implementation run
    //optimized run results file path
    const std::filesystem::path opt_results_file_path{
      imp_results_fp / run_eval::kImpResultsRunDataFolderName /
      std::filesystem::path(run_imp_settings.run_name + "_"  +
      std::string(run_eval::kRunResultsDescFileName) +
      std::string(run_eval::kCsvFileExtension))};

    //default parallel params results file path
    const std::filesystem::path default_params_results_file_path{
      imp_results_fp / run_eval::kImpResultsRunDataDefaultPParamsFolderName /
      std::filesystem::path(run_imp_settings.run_name + "_" +
      std::string(run_eval::kRunResultsDefaultPParamsDescFileName) +
      std::string(run_eval::kCsvFileExtension))};

    //optimized run results with speedups file path
    const std::filesystem::path opt_results_w_speedup_file_path{
      imp_results_fp / run_eval::kImpResultsRunDataWSpeedupsFolderName /
      std::filesystem::path(run_imp_settings.run_name + "_" +
      std::string(run_eval::kRunResultsWSpeedupsDescFileName) +
      std::string(run_eval::kCsvFileExtension))};

    //speedup results file path
    const std::filesystem::path speedups_results_file_path{
      imp_results_fp / run_eval::kImpResultsSpeedupsFolderName /
      std::filesystem::path(run_imp_settings.run_name + "_" +
      std::string(run_eval::kSpeedupsDescFileName) +
      std::string(run_eval::kCsvFileExtension))};
    
    //initialize parallel params setting enum and set of parallel params
    //settings that are enabled in run
    const auto parallel_param_settings =
      p_params_optimized ?
      std::set<run_environment::ParallelParamsSetting>{
        run_environment::ParallelParamsSetting::kDefault,
        run_environment::ParallelParamsSetting::kOptimized} :
      std::set<run_environment::ParallelParamsSetting>{
        run_environment::ParallelParamsSetting::kDefault};

    //get headers from first successful run and write headers to top of output files
    auto headers_in_order = first_success_run_iter->second->at(
      run_environment::ParallelParamsSetting::kOptimized).HeadersInOrder();

    //get vector of speedup headers to use for evaluation across runs
    //and also for run results
    std::vector<std::string> speedup_headers;
    for (const auto& [speedup_header, _] : speedup_headers_w_data) {
      speedup_headers.push_back(speedup_header);
    }

    //delete any speedup headers already in headers_in_order
    //since they will be added in order to end of vector
    for (const auto& speedup_header : speedup_headers) {
      if (auto speedup_header_iter = std::find(headers_in_order.cbegin(),
                                               headers_in_order.cend(),
                                               speedup_header);
          speedup_header_iter != headers_in_order.cend())
      {
        headers_in_order.erase(speedup_header_iter);
      }
    }
    
    //add speedup headers to the end of headers in order
    headers_in_order.insert(
      headers_in_order.cend(),
      speedup_headers.cbegin(),
      speedup_headers.cend());

    //initialize map of ostringstreams for run data using default and optimized parallel parameters
    std::map<run_environment::ParallelParamsSetting, std::ostringstream> run_data_sstr;
    for (const auto& p_param_setting : parallel_param_settings) {
      run_data_sstr.insert({p_param_setting, std::ostringstream()});
    }

    //write each results header in order in first row of results string
    //for each results set and then write newline to go to next line
    for (const auto& curr_header : headers_in_order) {
      for (const auto& p_param_setting : parallel_param_settings) {
        run_data_sstr.at(p_param_setting) << curr_header << ',';
      }
    }
    for (const auto& p_param_setting : parallel_param_settings) {
      run_data_sstr.at(p_param_setting) << std::endl;
    }

    //write output for run on each input with each data type
    //write data for default parallel parameters and for optimized parallel parameters
    for (const auto& p_param_setting : parallel_param_settings) {
      for (const auto& [_, run_sig_data] : run_results) {
        //if run not successful only have single set of output data from run
        //don't write data if no data for run
        if (run_sig_data) {
          for (const auto& curr_header : headers_in_order) {
            if (run_sig_data->at(p_param_setting).IsData(curr_header))
            {
              run_data_sstr.at(p_param_setting) <<
                run_sig_data->at(p_param_setting).GetDataAsStr(curr_header);
            }
            run_data_sstr.at(p_param_setting) << ',';            
          }
          run_data_sstr.at(p_param_setting) << std::endl;
        }
      }
    }

    //write run results strings to output streams corresponding to results files
    //two results files contain run results (with and without optimized
    //parallel parameters), one results file contains speedup results, and another
    //contains run results followed by speedups

    //write run results with and without optimized parallel parameters to files
    //if run without optimized parallel parameters then results with default
    //parallel parameters written to optimized parallel parameters file
    enum class OutResults { kDefaultPParams, kOptPParams, kSpeedups, kOptWSpeedups };
    std::map<OutResults, std::ofstream> results_stream;
    results_stream.insert({OutResults::kDefaultPParams,
                           std::ofstream(default_params_results_file_path)});
    results_stream.insert({OutResults::kOptPParams,
                           std::ofstream(opt_results_file_path)});
    results_stream.insert({OutResults::kSpeedups,
                           std::ofstream(speedups_results_file_path)});
    results_stream.insert({OutResults::kOptWSpeedups,
                           std::ofstream(opt_results_w_speedup_file_path)});
    //write run results file with default parallel params
    results_stream.at(OutResults::kDefaultPParams) <<
      run_data_sstr.at(run_environment::ParallelParamsSetting::kDefault).str();
    if (p_params_optimized) {
      //write run results with optimized parallel params to optimized
      //run results file stream
      results_stream.at(OutResults::kOptPParams) <<
        run_data_sstr.at(run_environment::ParallelParamsSetting::kOptimized).str();
      //add optimized run results to ostringstream for output containing run results and speedups
      results_stream.at(OutResults::kOptWSpeedups) <<
        run_data_sstr.at(run_environment::ParallelParamsSetting::kOptimized).str();
    }
    else {
      //write results with default parallel parameters to optimized results file
      //stream if parallel parameters not optimized in run
      results_stream.at(OutResults::kOptPParams) <<
        run_data_sstr.at(run_environment::ParallelParamsSetting::kDefault).str();     
      results_stream.at(OutResults::kOptWSpeedups) << 
        run_data_sstr.at(run_environment::ParallelParamsSetting::kDefault).str();        
    }

    //generate speedup results with headers on top row and write to
    //"speedup" results stream
    results_stream.at(OutResults::kSpeedups) << ',';
    for (const auto& [speedup_header, _] : speedup_headers_w_data) {
      results_stream.at(OutResults::kSpeedups) << speedup_header << ',';
    }
    for (const auto& [middle_val_desc, middle_val_enum] : 
      {std::pair<std::string_view, run_eval::MiddleValData>{"Average Speedup", run_eval::MiddleValData::kAverage},
       std::pair<std::string_view, run_eval::MiddleValData>{"Median Speedup", run_eval::MiddleValData::kMedian}})
    {
      results_stream.at(OutResults::kSpeedups) << std::endl << middle_val_desc << ',';
      for (const auto& [_, speedup_data] : speedup_headers_w_data) {
        if (speedup_data.contains(middle_val_enum)) {
          results_stream.at(OutResults::kSpeedups) << speedup_data.at(middle_val_enum) << ',';
        }
        else {
          results_stream.at(OutResults::kSpeedups) << ',';
        }
      }
    }

    //generate speedup results with headers on left side
    //and add to results stream for run results with speedups
    results_stream.at(OutResults::kOptWSpeedups) << std::endl <<
      "Speedup Results,Average Speedup,Median Speedup" << std::endl;
    for (const auto& [speedup_header, speedup_data] : speedup_headers_w_data) {
      results_stream.at(OutResults::kOptWSpeedups) << speedup_header;
      if ((speedup_data.contains(run_eval::MiddleValData::kAverage)) &&
          (speedup_data.at(run_eval::MiddleValData::kAverage) > 0)) {
        results_stream.at(OutResults::kOptWSpeedups) << ',' <<
          speedup_data.at(run_eval::MiddleValData::kAverage) << ',' <<
          speedup_data.at(run_eval::MiddleValData::kMedian);
      }
      else {
        results_stream.at(OutResults::kOptWSpeedups) << ",,";
      }
      results_stream.at(OutResults::kOptWSpeedups) << std::endl;
    }

    //close streams for writing to results files since file writing is done
    for (auto& [_, curr_results_stream] : results_stream) {
      curr_results_stream.close();
    }

    //print location of output evaluation files to standard output
    std::cout << "Input/settings/parameters info, detailed timings, and evaluation for each run and across runs in "
              << opt_results_w_speedup_file_path << std::endl;
    std::cout << "Run inputs and results in " << opt_results_file_path << std::endl;
    std::cout << "Speedup results in " << speedups_results_file_path << std::endl;
    std::cout << "Run inputs and results using default parallel parameters in "
              << default_params_results_file_path << std::endl;

    //run evaluation across current and previous runs across architectures
    //using run results and speedups saved from previous runs along with
    //current run results
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
     std::string(run_eval::kSpeedupVsAvx256Vectorization) + " - " + std::string(run_environment::kDataSizeToNameMap.at(data_type_size))},
    {run_environment::AccSetting::kAVX256_F16, 
     std::string(run_eval::kSpeedupVsAvx256Vectorization) + " - " + std::string(run_environment::kDataSizeToNameMap.at(data_type_size))}};

  if (run_imp_results_by_acc_setting.size() == 1) {
    //no alternate run results
    return {};
  }
  else {
    //initialize speedup/slowdown using alternate acceleration
    std::vector<RunSpeedupAvgMedian> alt_acc_speedups;
    for (auto& [acc_setting, acc_results] : run_imp_results_by_acc_setting) {
      if ((acc_setting != fastest_acc) && (acc_to_speedup_str.contains(acc_setting))) {
        //process results using alternate acceleration
        //go through each result and replace initial run data with alternate
        //implementation run data if alternate implementation run is faster
        for (auto& [run_input_sig, run_input_results] : run_imp_results_by_acc_setting.at(fastest_acc).first)
        {
          if (run_input_results && acc_results.first.at(run_input_sig))
          {
            //get runtime of "fastest" acceleration run that is to contain the
            //optimized results and alternate acceleration run
            //if alternate acceleration run is faster, replace the optimized
            //result for run with alternate acceleration run since it is faster
            const double init_result_time =
              *run_input_results->at(
                run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(run_eval::kOptimizedRuntimeHeader);
            const double alt_acc_result_time =
              *acc_results.first.at(run_input_sig)->at(
                run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(run_eval::kOptimizedRuntimeHeader);
            if (alt_acc_result_time < init_result_time) {
              run_input_results = acc_results.first.at(run_input_sig);
            }
          }
        }
        //get speedup/slowdown using alternate acceleration compared to
        //fastest implementation and store in speedup results
        //TODO: right now this is computed with alternate acceleration result
        //having replaced "fastest acceleration" result if it is faster, may
        //want to change that
        alt_acc_speedups.push_back(GetAvgMedSpeedupBaseVsTarget(
          acc_results.first,
          run_imp_results_by_acc_setting.at(fastest_acc).first,
          acc_to_speedup_str.at(acc_setting),
          BaseTargetDiff::kDiffAcceleration));
      }
    }
    return alt_acc_speedups;
  }
}

//get speedup over baseline data if data available
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetSpeedupOverBaseline(
  const run_environment::RunImpSettings& run_imp_settings,
  MultRunData& run_data_all_runs,
  size_t data_type_size) const
{
  //initialize speedup results
  std::vector<RunSpeedupAvgMedian> speedup_results;

  //get speedup over baseline runtimes...can only compare with baseline runtimes that are
  //generated using same templated iterations setting as current run
  if (run_imp_settings.baseline_runtimes_path_desc)
  {
    const auto speedup_over_baseline = GetAvgMedSpeedupOverBaseline(
      run_data_all_runs,
      run_environment::kDataSizeToNameMap.at(data_type_size),
      *run_imp_settings.baseline_runtimes_path_desc);
    speedup_results.insert(
      speedup_results.cend(),
      speedup_over_baseline.cbegin(),
      speedup_over_baseline.cend());
  }

  return speedup_results;
}

//get speedup over baseline run for subsets of smallest and largest sets if data available
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetSpeedupOverBaselineSubsets(
  const run_environment::RunImpSettings& run_imp_settings,
  MultRunData& run_data_all_runs,
  size_t data_type_size) const
{
  if (run_imp_settings.baseline_runtimes_path_desc)
  {
    return GetAvgMedSpeedupOverBaselineSubsets(
      run_data_all_runs,
      run_environment::kDataSizeToNameMap.at(data_type_size),
      *run_imp_settings.baseline_runtimes_path_desc,
      run_imp_settings.subset_desc_input_sig);
  }
  //return empty vector if doesn't match settings to get speedup over baseline for subsets
  return std::vector<RunSpeedupAvgMedian>();
}

//get baseline runtime data if available...return null if baseline data not available
//key for runtime data in results is different to retrieve optimized runtime compared to
//single thread runtime for baseline run
std::optional<std::pair<std::string, std::map<InputSignature, std::string>>> EvaluateImpResults::GetBaselineRuntimeData(
  const std::array<std::string_view, 2>& baseline_runtimes_path_desc,
  std::string_view key_runtime_data) const
{
  RunResultsSpeedups baseline_run_results(baseline_runtimes_path_desc.at(0));
  return std::pair<std::string, std::map<InputSignature, std::string>>{
    std::string(baseline_runtimes_path_desc.at(1)),
    baseline_run_results.InputsToKeyVal(key_runtime_data)};
}

//get average and median speedup from vector of speedup values
RunSpeedupAvgMedian::second_type EvaluateImpResults::GetAvgMedSpeedup(
  const std::vector<double>& speedups_vect) const
{
  const double average_speedup =
    (std::accumulate(speedups_vect.cbegin(), speedups_vect.cend(), 0.0) / (double)speedups_vect.size());
  auto speedups_vect_sorted = speedups_vect;
  std::ranges::sort(speedups_vect_sorted);
  const double median_speedup = ((speedups_vect_sorted.size() % 2) == 0) ? 
    (speedups_vect_sorted[(speedups_vect_sorted.size() / 2) - 1] + speedups_vect_sorted[(speedups_vect_sorted.size() / 2)]) / 2.0 : 
    speedups_vect_sorted[(speedups_vect_sorted.size() / 2)];
  return {{run_eval::MiddleValData::kAverage, average_speedup},
          {run_eval::MiddleValData::kMedian, median_speedup}};
}

//get average and median speedup of specified subset(s) of runs compared to baseline data from file
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetAvgMedSpeedupOverBaselineSubsets(
  MultRunData& run_results,
  std::string_view data_type_str,
  const std::array<std::string_view, 2>& baseline_runtimes_path_desc,
  const std::vector<std::pair<std::string, std::vector<InputSignature>>>& subset_desc_input_sig) const
{
  //get speedup over baseline for optimized runs
  std::vector<RunSpeedupAvgMedian> speedup_data;
  const auto baseline_run_data =
    GetBaselineRuntimeData(baseline_runtimes_path_desc, run_eval::kOptimizedRuntimeHeader);
  if (baseline_run_data) {
    const auto& baseline_runtimes = (*baseline_run_data).second;
    //retrieve speedup data for subsets of optimized runs over corresponding
    //run in baseline data
    for (const auto& [subset_desc, subset_inputs] : subset_desc_input_sig) {
      std::vector<double> speedups_vect;
      //get header corresponding to current subset
      const std::string speedup_header = "Speedup relative to " + std::string((*baseline_run_data).first) + " on " +
        std::string(subset_desc) + " - " + std::string(data_type_str);
      //go through each input signature of current subset
      for (const auto& subset_input_signature : subset_inputs) {
        //go through each run output and compute speedup for each run that matches
        //subset input signature
        for (auto& [run_sig, run_sig_results] : run_results)
        {
          if (run_sig_results)
          {
            //check if run signature corresponds to subset signature and retrieve
            //and process speedup if that's the case
            if (run_sig.EqualsUsingAny(subset_input_signature)) {
              speedups_vect.push_back(
                std::stod(baseline_runtimes.at(run_sig)) /
                *run_sig_results->at(
                  run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(
                    run_eval::kOptimizedRuntimeHeader));
              for (auto& run_data : *run_sig_results) {
                run_data.second.AddDataWHeader(std::string(speedup_header), speedups_vect.back());
              }
            }
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
std::vector<RunSpeedupAvgMedian> EvaluateImpResults::GetAvgMedSpeedupOverBaseline(
  MultRunData& run_results,
  std::string_view data_type_str,
  const std::array<std::string_view, 2>& baseline_runtimes_path_desc) const
{
  //define the start of the speedup info and runtime header in run data
  //that is being compared for optimized and single thread implementations
  const std::array<std::string_view, 2> baseline_opt_start_info_runtime_header{
    "Speedup relative to", run_eval::kOptimizedRuntimeHeader};
  const std::array<std::string_view, 2> baseline_s_thread_start_info_runtime_header{
    "Single-Thread (Orig Imp) speedup relative to", run_eval::kSingleThreadRuntimeHeader};
  
  //vector for speedup data over baseline for optimized and single thread
  //implementations
  std::vector<RunSpeedupAvgMedian> speedup_data;

  //get speedup of current run compared to baseline optimized and
  //single thread implementations
  for (const auto& start_info_runtime_header : 
       {baseline_opt_start_info_runtime_header,
        baseline_s_thread_start_info_runtime_header})
  {
    //get baseline run data for runs according to current runtime header
    const auto baseline_run_data =
      GetBaselineRuntimeData(baseline_runtimes_path_desc, start_info_runtime_header.at(1));
    if (baseline_run_data) {
      const auto& [baseline_name, baseline_runtimes_to_sig] = *baseline_run_data;
      std::vector<double> speedups_vect;
      const std::string speedup_header = std::string(start_info_runtime_header.at(0)) +
        " " + baseline_name + " - " + std::string(data_type_str);
      for (auto& [run_sig, run_sig_results] : run_results)
      {
        //compute speedup for each run compared to baseline runtime and add
        //to vector of speedups as well as to run data
        if (run_sig_results && (baseline_runtimes_to_sig.contains(run_sig))) {
          const auto runtime_input_sig = *run_sig_results->at(
            run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(
              start_info_runtime_header.at(1));
          const auto baseline_runtime = std::stod(baseline_runtimes_to_sig.at(run_sig));
          speedups_vect.push_back(baseline_runtime / runtime_input_sig);
          for (auto& [_, run_data] : *run_sig_results) {
            run_data.AddDataWHeader(speedup_header, speedups_vect.back());
          }
        }
      }
      if (!(speedups_vect.empty())) {
        //get average and median speedups across all runs compared to baseline runtimes
        speedup_data.push_back({speedup_header, GetAvgMedSpeedup(speedups_vect)});
      }
    }
  }

  return speedup_data;
}

//get average and median speedup using optimized parallel parameters compared to default parallel parameters
//and also add speedup for each run using optimized parallel parameters compared to each run with default
//parallel parameters
RunSpeedupAvgMedian EvaluateImpResults::GetAvgMedSpeedupOptPParams(
  MultRunData& run_results, std::string_view speedup_header) const
{
  //initialize vector of speedups across runs
  std::vector<double> speedups_vect;
  for (auto& [_, sig_run_results] : run_results)
  {
    if (sig_run_results) {
      //compute speedup of run using optimized parallel parameters compared to default parallel parameters
      //and add to vector of speedups for all runs
      speedups_vect.push_back(
        (*(sig_run_results->at(
          run_environment::ParallelParamsSetting::kDefault).GetDataAsDouble(
            run_eval::kOptimizedRuntimeHeader))) / 
        (*(sig_run_results->at(
          run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(
            run_eval::kOptimizedRuntimeHeader))));
      
      //add speedup for run to data for run in run results
      sig_run_results->at(
        run_environment::ParallelParamsSetting::kDefault).AddDataWHeader(
          std::string(speedup_header), speedups_vect.back());
      sig_run_results->at(
        run_environment::ParallelParamsSetting::kOptimized).AddDataWHeader(
          std::string(speedup_header), speedups_vect.back());
    }
  }
  if (!(speedups_vect.empty())) {
    //compute average and median speedup when using parallel parameters
    //across runs from vector containing speedups of every run
    return {RunSpeedupAvgMedian::first_type(speedup_header), GetAvgMedSpeedup(speedups_vect)};
  }
  return {RunSpeedupAvgMedian::first_type(speedup_header), {}};
}

//get average and median speedup between base and target runtime data and also add
//speedup for each target runtime data run as compared to corresponding base run
RunSpeedupAvgMedian EvaluateImpResults::GetAvgMedSpeedupBaseVsTarget(
  MultRunData& run_results_base,
  MultRunData& run_results_target,
  std::string_view speedup_header,
  BaseTargetDiff base_target_diff) const
{
  std::vector<double> speedups_vect;
  //go through all base runtime data
  for (auto& [input_sig_base, sig_run_results_base] : run_results_base)
  {
    InputSignature base_in_sig_adjusted(input_sig_base);
    //go through all target runtime data and find speedup for data
    //that corresponds with current base runtime data
    for (auto& [input_sig_target, sig_run_results_target] : run_results_target)
    {
      InputSignature target_in_sig_adjusted(input_sig_target);
      if (base_target_diff == BaseTargetDiff::kDiffDatatype) {
        //remove datatype from input signature if different datatype
        //between base and target output
        base_in_sig_adjusted.RemoveDatatypeSetting();
        target_in_sig_adjusted.RemoveDatatypeSetting();
      }
      else if (base_target_diff == BaseTargetDiff::kDiffTemplatedSetting) {
        //remove templated setting from input signature if different template
        //setting between base and target output
        base_in_sig_adjusted.RemoveTemplatedLoopIterSetting();
        target_in_sig_adjusted.RemoveTemplatedLoopIterSetting();
      }
      if (base_in_sig_adjusted == target_in_sig_adjusted) {
        //compute speedup between corresponding base and target data and
        //add it to vector of speedups across all data
        if (sig_run_results_base && 
            sig_run_results_target)
        {
          speedups_vect.push_back(
            *sig_run_results_base->at(
              run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(
                run_eval::kOptimizedRuntimeHeader) / 
            *sig_run_results_target->at(
              run_environment::ParallelParamsSetting::kOptimized).GetDataAsDouble(
                run_eval::kOptimizedRuntimeHeader));

          //add speedup data to corresponding base and target data
          sig_run_results_base->at(
            run_environment::ParallelParamsSetting::kOptimized).AddDataWHeader(
              std::string(speedup_header), speedups_vect.back());
          sig_run_results_target->at(
            run_environment::ParallelParamsSetting::kOptimized).AddDataWHeader(
              std::string(speedup_header), speedups_vect.back());
        }
      }
    }
  }
  if (!(speedups_vect.empty())) {
    //get average and median speedup from speedup vector with all speedup
    //data and return as pair with header describing speedup
    return {RunSpeedupAvgMedian::first_type(speedup_header), GetAvgMedSpeedup(speedups_vect)};
  }
  return {RunSpeedupAvgMedian::first_type(speedup_header), {}};
}

//get average and median speedup when loop iterations are given at compile time as template value
//and also add speedup for each run with templated loop iterations as compared to same run without
//templated loop iterations
RunSpeedupAvgMedian EvaluateImpResults::GetAvgMedSpeedupLoopItersInTemplate(
  MultRunData& run_results,
  std::string_view speedup_header) const
{
  //get all data entries corresponding to templated loop iterations and not templated loop iterations
  //in separate structures
  std::array<MultRunData, 2> templated_non_templated_loops_data;
  for (const auto data_index : {0, 1}) {
    std::copy_if(run_results.cbegin(), run_results.cend(),
      std::inserter(templated_non_templated_loops_data[data_index],
                    templated_non_templated_loops_data[data_index].end()),
        [data_index](const auto& input_sig_run_results) {
          if (input_sig_run_results.first.TemplatedLoopIters()) {
            //data index 0 -> return true if templated loop iters
            //data index 1 -> return true if not templated loop iters
            if (data_index == 0) {
              return *input_sig_run_results.first.TemplatedLoopIters();
            }
            if (data_index == 1) {
              return (!(*input_sig_run_results.first.TemplatedLoopIters()));
            }
            else {
              return false;
            }
          }
          else {
            return false;
          }
      });
  }

  //get speedup using templated loop iteration counts with non-templated loop
  //iters being base data and templated loop iterations counts being target data
  const RunSpeedupAvgMedian speedup_header_data =
    GetAvgMedSpeedupBaseVsTarget(
      templated_non_templated_loops_data[1],
      templated_non_templated_loops_data[0],
      speedup_header,
      BaseTargetDiff::kDiffTemplatedSetting);
  
  //update run output data with templated and non templated data results
  //after processing for speedup
  //reason is that speedup data added to each data entry showing speedup
  //compared to corresponding data with or without templated loops
  for (const auto& run_data_updated : templated_non_templated_loops_data) {
    for (const auto& curr_run_data : run_data_updated) {
      run_results.at(curr_run_data.first) = curr_run_data.second;
    }
  }

  //return speedup data with header
  return speedup_header_data;
}