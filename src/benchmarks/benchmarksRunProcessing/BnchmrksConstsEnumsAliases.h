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
#include "RunEval/RunTypeConstraints.h"

/**
 * @brief Namespace for enums, constants, structures, and
 * functions specific to benchmarks processing
 */
namespace benchmarks {

/** @brief Define the benchmark options */
enum class BenchmarkRun : size_t { kAddOneD, kAddTwoD, kCopyTwoD, kCopyOneD, kGemm };

};

#endif /* BNCHMRKS_CONSTS_ENUMS_ALIASES_H_ */
