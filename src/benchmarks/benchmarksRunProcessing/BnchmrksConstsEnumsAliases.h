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
 * @file BnchmrksConstsEnumsAliases.h
 * @author Scott Grauer-Gray
 * @brief File with namespace for enums, constants, structures, and
 * functions specific to benchmarks processing
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef BNCHMRKS_CONSTS_ENUMS_ALIASES_H_
#define BNCHMRKS_CONSTS_ENUMS_ALIASES_H_

#include <array>
#include <unordered_map>
#include <string_view>
#include "RunEval/RunTypeConstraints.h"

/**
 * @brief Namespace for enums, constants, structures, and
 * functions specific to benchmarks processing
 */
namespace benchmarks {

/** @brief header for benchmark name */
constexpr std::string_view kBenchmarkHeader{"Benchmark"};

/** @brief Define the benchmark options */
enum class BenchmarkRun : size_t {
  kAddOneD, kAddTwoD, kDivideOneD, kDivideTwoD, kCopyOneD, kCopyTwoD, kGemm };

/** @brief Mapping between benchmark enums and names */
const std::unordered_map<BenchmarkRun, std::string_view> kBnchmrksNames{
  {BenchmarkRun::kAddOneD, "Addition (1D Matrices)"},
  {BenchmarkRun::kAddTwoD, "Addition (2D Matrices)"},
  {BenchmarkRun::kDivideOneD, "Divide (1D Matrices)"},
  {BenchmarkRun::kDivideTwoD, "Divide (2D Matrices)"},
  {BenchmarkRun::kCopyOneD, "Copy (1D Matrices)"},
  {BenchmarkRun::kCopyTwoD, "Copy (2D Matrices)"},
  {BenchmarkRun::kGemm, "Gemm"}};
};

#endif /* BNCHMRKS_CONSTS_ENUMS_ALIASES_H_ */
