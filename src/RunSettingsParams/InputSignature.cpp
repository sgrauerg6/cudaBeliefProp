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
 * @file InputSignature.cpp
 * @author Scott Grauer-Gray
 * @brief Function definitions for class for defining input signature for evaluation run
 *   that consists of evaluation set number, data type, and whether or not to use
 *   templated loop iteration count.
 * 
 * @copyright Copyright (c) 2024
 */

#include <charconv>
#include "InputSignature.h"

//constructor to generate evaluation input signature from string array with
//strings corresponding to each part
InputSignature::InputSignature(const std::array<std::string_view, 3>& in_sig_strings) {
  const std::map<std::string_view, unsigned int> datatype_str_to_size{
    {run_environment::kDataSizeToNameMap.at(sizeof(float)), sizeof(float)},
    {run_environment::kDataSizeToNameMap.at(sizeof(double)), sizeof(double)},
    {run_environment::kDataSizeToNameMap.at(sizeof(short)), sizeof(short)}};
  data_type_size_ = datatype_str_to_size.at(in_sig_strings[run_eval::kRunInputDatatypeIdx]);
  int num_from_string{};
  std::from_chars(
    in_sig_strings[run_eval::kRunInputNumInputIdx].data(),
    in_sig_strings[run_eval::kRunInputNumInputIdx].data() + in_sig_strings[run_eval::kRunInputNumInputIdx].size(),
    num_from_string);
  eval_set_num_ = num_from_string;
  use_templated_loop_iters_ = 
    (in_sig_strings[run_eval::kRunInputLoopItersTemplatedIdx] == 
     run_eval::kBoolValFalseTrueDispStr[1]) ? true : false;      
}

//constructor to generate evaluation input signature from parameters
//corresponding to each part
InputSignature::InputSignature(
  std::optional<unsigned int> data_type_size,
  std::optional<unsigned int> eval_set_num,
  std::optional<bool> use_templated_loop_iters) :
  data_type_size_(data_type_size),
  eval_set_num_(eval_set_num),
  use_templated_loop_iters_(use_templated_loop_iters) {}

//less than operator for comparing evaluation input signatures
//so they can be ordered
//operator needed to support ordering since input signature
//is used as std::map key and also for evaluation output order
//don't use input signature parts where one of the comparison sides
//has "no value" which corresponds to "any" for the input signature
bool InputSignature::operator<(const InputSignature& rhs) const {
  if (((data_type_size_) && (rhs.data_type_size_)) &&
      ((*data_type_size_ != *rhs.data_type_size_))) {
    //compare datatype
    //order is float, double, half
    //define mapping of datatype string to value for comparison
    const std::map<unsigned int, unsigned int> datatype_size_to_order_num{
      {sizeof(float), 0},
      {sizeof(double), 1},
      {sizeof(short), 2}};
    return (datatype_size_to_order_num.at(*data_type_size_) <
            datatype_size_to_order_num.at(*rhs.data_type_size_));
  }
  else if (((eval_set_num_) && (rhs.eval_set_num_)) &&
           ((*eval_set_num_ != *rhs.eval_set_num_))) {
    //compare evaluation data number
    //ordering is as expected by numeric value (such as 0 < 1)
    return (eval_set_num_ < rhs.eval_set_num_);
  }
  else if ((use_templated_loop_iters_ && rhs.use_templated_loop_iters_) &&
           ((*use_templated_loop_iters_ != *rhs.use_templated_loop_iters_))) {
    //compare whether or not using templated iter count
    //order is using templated iter count followed by not using templated iter count
    if (*use_templated_loop_iters_ == true) { return true; /* a < b is true */ }
  }
  return false; /* a <= b is false */
}

//equality operator for comparing evaluation input signatures
//only compare inputs with valid values...no value corresponds to "any value"
//for part of input signature so only consider parts where both have valid
//values
bool InputSignature::operator==(const InputSignature& rhs) const {
  return (std::tie(data_type_size_, eval_set_num_, use_templated_loop_iters_) ==
          std::tie(rhs.data_type_size_, rhs.eval_set_num_, rhs.use_templated_loop_iters_));
}

//alternate "equal" operator where an attribute is considered "equal"
//in cases where one side is "ANY" for the attribute as indicated
//by "no value" for std::optional object
bool InputSignature::EqualsUsingAny(const InputSignature& rhs) const {
  return (std::tie(rhs.data_type_size_ ? data_type_size_ : std::optional<unsigned int>(),
                   rhs.eval_set_num_ ? eval_set_num_ : std::optional<unsigned int>(),
                   rhs.use_templated_loop_iters_ ? use_templated_loop_iters_ : std::optional<bool>()) ==
          std::tie(data_type_size_ ? rhs.data_type_size_ : std::optional<unsigned int>(),
                   eval_set_num_ ? rhs.eval_set_num_ : std::optional<unsigned int>(),
                   use_templated_loop_iters_ ? rhs.use_templated_loop_iters_ : std::optional<bool>()));
}
