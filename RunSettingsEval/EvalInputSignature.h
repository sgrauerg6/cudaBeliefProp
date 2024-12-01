/*
 * EvalInputSignature.h
 *
 *  Created on: Nov 22, 2024
 *      Author: scott
 * 
 *  Class for defining input signature for evaluation run that consists of evaluation set number,
 *  data type, and whether or not to use templated loop iteration count.
 */

#ifndef EVAL_INPUT_SIGNATURE_H_
#define EVAL_INPUT_SIGNATURE_H_

#include <map>
#include <string_view>
#include <string>
#include <charconv>
#include <iostream>
#include <fstream>
#include "RunEvalConstsEnums.h"

//class defines input signature for evaluation run that contains evaluation set
//number, data type, and whether or not to use templated loop iteration count
class EvalInputSignature {
public:
  //constructor to generate evaluation input signature from string array with
  //strings corresponding to each part
  EvalInputSignature(const std::array<std::string_view, 3>& in_sig_strings);

  //constructor to generate evaluation input signature from parameters
  //corresponding to each part
  EvalInputSignature(
    std::optional<unsigned int> data_type_size,
    std::optional<unsigned int> eval_set_num,
    std::optional<bool> use_templated_loop_iters);

  //less than operator for comparing evaluation input signatures
  //so they can be ordered
  //operator needed to support ordering since input signature
  //is used as std::map key and also for evaluation output order
  //if any EvalInputSignature member is "no value" that property is considered
  //"any" and is ignored in the comparison
  bool operator<(const EvalInputSignature& rhs) const;

  //equality operator for comparing evaluation input signatures
  //if any EvalInputSignature member is "no value" that property is considered
  //"any" and is ignored in the comparison
  bool operator==(const EvalInputSignature& rhs) const;

  std::string DataTypeStr() const {
    if (!(data_type_size_.has_value())) {
      return "ANY";
    }
    if (data_type_size_.value() == 2) {
      return "HALF";
    }
    else if (data_type_size_.value() == 4) {
      return "FLOAT";
    }
    else if (data_type_size_.value() == 8) {
      return "DOUBLE";
    }
    return "UNKNOWN";
  }

  std::string EvalSetNumStr() const {
    if (!(eval_set_num_.has_value())) {
      return "ANY";
    }
    else {
      return std::to_string(eval_set_num_.value());
    }
  }

  std::string UseTemplatedLoopItersStr() const {
    if (!(use_templated_loop_iters_.has_value())) {
      return "BOTH";
    }
    return ((!(use_templated_loop_iters_.value())) ?
      std::string(run_eval::kBoolValFalseTrueDispStr[0]) : std::string(run_eval::kBoolValFalseTrueDispStr[1]));
  }
  
  //remove templated loop iter setting and change it to "any"
  void RemoveTemplatedLoopIterSetting() {
    use_templated_loop_iters_.reset();
  }

  void RemoveDatatypeSetting() {
    data_type_size_.reset();
  }

  //overloaded << operator to write EvalInputSignature object to stream
  friend std::ostream& operator<<(std::ostream& os, const EvalInputSignature& eval_input_sig);

private:
  std::optional<unsigned int> data_type_size_;
  std::optional<unsigned int> eval_set_num_;
  std::optional<bool> use_templated_loop_iters_;
};

//overloaded << operator to write EvalInputSignature object to stream
inline std::ostream& operator<<(std::ostream& os, const EvalInputSignature& eval_input_sig) {
  os << eval_input_sig.DataTypeStr() << " "  << eval_input_sig.EvalSetNumStr() << " " << eval_input_sig.UseTemplatedLoopItersStr();
  return os;
}

using MultRunData = std::map<EvalInputSignature, std::optional<std::map<run_environment::ParallelParamsSetting, RunData>>>;
using RunSpeedupAvgMedian = std::pair<std::string, std::array<double, 2>>;
using MultRunDataWSpeedupByAcc =
  std::unordered_map<run_environment::AccSetting, std::pair<MultRunData, std::vector<RunSpeedupAvgMedian>>>;

#endif //EVAL_INPUT_SIGNATURE_H_