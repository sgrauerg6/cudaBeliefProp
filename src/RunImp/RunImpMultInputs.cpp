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
 * @file RunImpMultInputs.cpp
 * @author Scott Grauer-Gray
 * @brief 
 * 
 * @copyright Copyright (c) 2024
 */

#include "RunImpMultInputs.h"

//run and evaluate runs on one or more input of benchmark implementation
//using multiple settings
std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>> RunImpMultInputs::operator()(
  const run_environment::RunImpSettings& run_imp_settings,
  size_t data_type_size,
  std::unique_ptr<EvaluateImpResults>& evalResults) const
{
  //run belief propagation implementation on multiple datasets and
  //return run data for all runs
  MultRunData run_data_all_runs =
    RunEvalImpMultDataSets(run_imp_settings, data_type_size);

  //evaluate results
  //return data for each run and average and median
  //speedup results across the data
  return evalResults->EvalResultsSingDataTypeAcc(
    run_data_all_runs, run_imp_settings, data_type_size);
}
