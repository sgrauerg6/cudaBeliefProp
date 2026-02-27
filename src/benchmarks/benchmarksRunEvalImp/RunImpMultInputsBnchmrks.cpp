/*
Copyright (C) 2026 Scott Grauer-Gray

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
 * @file RunImpMultInputsBnchmrks.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2026
 */

#include "RunImpOnInputBnchmrks.h"
#include "RunImpMultInputsBnchmrks.h"

MultRunData RunImpMultInputsBnchmrks::RunEvalImpMultDataSets(
  const run_environment::RunImpSettings& run_imp_settings,
  size_t data_type_size) const 
{
  if (this->opt_imp_accel_ == run_environment::AccSetting::kCUDA) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kCUDA>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kAVX512) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kAVX512>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kAVX512_F16) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kAVX512_F16>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kAVX256) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kAVX256>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kAVX256_F16) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kAVX256_F16>(run_imp_settings, data_type_size);
  }
  else if (this->opt_imp_accel_ == run_environment::AccSetting::kNEON) {
    return RunEvalImpMultDataSets<run_environment::AccSetting::kNEON>(run_imp_settings, data_type_size);
  }
  //else (this->opt_imp_accel_ == run_environment::AccSetting::kNone)
  return RunEvalImpMultDataSets<run_environment::AccSetting::kNone>(run_imp_settings, data_type_size);
}

template <run_environment::AccSetting OPT_IMP_ACCEL>
MultRunData RunImpMultInputsBnchmrks::RunEvalImpMultDataSets(
  const run_environment::RunImpSettings& run_imp_settings,
  size_t data_type_size) const
{
  if (data_type_size == sizeof(float)) {
    return RunEvalImpMultDataSets<float, OPT_IMP_ACCEL>(run_imp_settings);
  }
  else if (data_type_size == sizeof(double)) {
    return RunEvalImpMultDataSets<double, OPT_IMP_ACCEL>(run_imp_settings);
  }
  else {
    //if using x86 CPU that supports AVX vectorization, need to check if using
    //float16 vectorization data stored as _Float16 if using float16
    //vectorization and as short if not
#if (defined(OPTIMIZED_CPU_RUN) && (!(defined(COMPILING_FOR_ARM))))
    if constexpr ((OPT_IMP_ACCEL == run_environment::AccSetting::kAVX512) ||
                  (OPT_IMP_ACCEL == run_environment::AccSetting::kAVX256))
    {
      return RunEvalImpMultDataSets<short, OPT_IMP_ACCEL>(run_imp_settings);
    }
#if defined(FLOAT16_VECTORIZATION)
    else if constexpr ((OPT_IMP_ACCEL == run_environment::AccSetting::kAVX512_F16) ||
                       (OPT_IMP_ACCEL == run_environment::AccSetting::kAVX256_F16) ||
                       (OPT_IMP_ACCEL == run_environment::AccSetting::kNone))
    {
      std::cout << "FLOAT16 USED" << std::endl;
      return RunEvalImpMultDataSets<_Float16, OPT_IMP_ACCEL>(run_imp_settings);
    }
#endif //FLOAT16_VECTORIZATION
#endif //(defined(OPTIMIZED_CPU_RUN) && (!(defined(COMPILING_FOR_ARM))))
    return RunEvalImpMultDataSets<halftype, OPT_IMP_ACCEL>(run_imp_settings);
  }
}

//run and evaluate optimized bp implementation on evaluation stereo sets
template <RunData_t T, run_environment::AccSetting OPT_IMP_ACCEL>
MultRunData RunImpMultInputsBnchmrks::RunEvalImpMultDataSets(
  const run_environment::RunImpSettings& run_imp_settings) const
{
  //run and evaluate bp implementation on all stereo sets used for benchmarking
  std::vector<MultRunData> run_results;
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_32)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_64)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_128)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_256)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_512)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_1024)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_2048)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_4096)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_6144)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_8192)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_12288)>().operator()(
    run_imp_settings));
  run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_16384)>().operator()(
    run_imp_settings));
  /*run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_20480)>().operator()(
    run_imp_settings));*/
  /*run_results.push_back(RunImpOnInputBnchmrks<
    T, OPT_IMP_ACCEL, static_cast<size_t>(benchmarks::MtrxWH::kMtrxWH_24576)>().operator()(
    run_imp_settings));*/

  //merge results for each input to overall results
  MultRunData run_data_all_runs;
  for (auto& run_result : run_results) {
    run_data_all_runs.merge(run_result);
  }
 
  return run_data_all_runs;
}
