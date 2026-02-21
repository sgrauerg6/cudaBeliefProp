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
 * @file DetailedTimingBnchmrksConsts.h
 * @author Scott Grauer-Gray
 * @brief Constants for timing benchmarks implementation
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef DETAILED_TIMING_BNCHMRKS_CONSTS_H_
#define DETAILED_TIMING_BNCHMRKS_CONSTS_H_

#include <string>
#include <unordered_map>

namespace benchmarks {

/** @brief Enum for each runtime segment in benchmark implementation shown in
 * timing outputs */
enum class Runtime_Type {
  kDataSetUp, kTotalTime, kTotalNoTransfer, kTotalWithTransfer};

/** @brief Mapping of runtime segment enum to header describing timing of the
 * segment */
const std::unordered_map<Runtime_Type, std::string_view> kTimingNames{
  {Runtime_Type::kDataSetUp, "Time to initialize data"}, 
  {Runtime_Type::kTotalTime, "Total Benchmarks Runtime"}, 
  {Runtime_Type::kTotalNoTransfer, "Total Runtime (not including data transfer time)"},
  {Runtime_Type::kTotalWithTransfer, "Total Runtime (including data transfer time)"}};
};

#endif /* DETAILED_TIMING_BNCHMRKS_CONSTS_H_ */
