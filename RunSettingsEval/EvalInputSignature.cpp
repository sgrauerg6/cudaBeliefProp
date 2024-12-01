/*
 * EvalInputSignature.cpp
 *
 *  Created on: Nov 22, 2024
 *      Author: scott
 * 
 *  Function definitions for class for defining input signature for evaluation run
 *  that consists of evaluation set number, data type, and whether or not to use
 *   templated loop iteration count.
 */

#include "EvalInputSignature.h"

//constructor to generate evaluation input signature from string array with
//strings corresponding to each part
EvalInputSignature::EvalInputSignature(const std::array<std::string_view, 3>& in_sig_strings) {
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
EvalInputSignature::EvalInputSignature(
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
bool EvalInputSignature::operator<(const EvalInputSignature& rhs) const {
  if (((data_type_size_.has_value()) && (rhs.data_type_size_.has_value())) &&
      ((data_type_size_.value() != rhs.data_type_size_.value()))) {
    //compare datatype
    //order is float, double, half
    //define mapping of datatype string to value for comparison
    const std::map<unsigned int, unsigned int> datatype_size_to_order_num{
      {sizeof(float), 0},
      {sizeof(double), 1},
      {sizeof(short), 2}};
    return (datatype_size_to_order_num.at(data_type_size_.value()) <
            datatype_size_to_order_num.at(rhs.data_type_size_.value()));
  }
  else if (((eval_set_num_.has_value()) && (rhs.eval_set_num_.has_value())) &&
           ((eval_set_num_.value() != rhs.eval_set_num_.value()))) {
    //compare evaluation data number
    //ordering is as expected by numeric value (such as 0 < 1)
    return (eval_set_num_ < rhs.eval_set_num_);
  }
  else if (((use_templated_loop_iters_.has_value()) && (rhs.use_templated_loop_iters_.has_value())) &&
           ((use_templated_loop_iters_.value() != rhs.use_templated_loop_iters_.value()))) {
    //compare whether or not using templated iter count
    //order is using templated iter count followed by not using templated iter count
    if (use_templated_loop_iters_.value() == true) { return true; /* a < b is true */ }
  }
  return false; /* a <= b is false */
}

//equality operator for comparing evaluation input signatures
//only compare inputs with valid values...no value corresponds to "any value"
//for part of input signature so only consider parts where both have valid
//values
bool EvalInputSignature::operator==(const EvalInputSignature& rhs) const {
  return (std::tie(data_type_size_, eval_set_num_, use_templated_loop_iters_) ==
          std::tie(rhs.data_type_size_, rhs.eval_set_num_, rhs.use_templated_loop_iters_));
}
