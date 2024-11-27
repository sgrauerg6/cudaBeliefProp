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
    unsigned int data_type_size,
    unsigned int eval_set_num,
    bool use_templated_loop_iters);

  //less than operator for comparing evaluation input signatures
  //so they can be ordered
  //operator needed to support ordering since input signature
  //is used as std::map key and also for evaluation output order
  bool operator<(const EvalInputSignature& rhs) const;

  std::string DataTypeStr() const {
    if (data_type_size_ == 2) {
      return "HALF";
    }
    else if (data_type_size_ == 4) {
      return "FLOAT";
    }
    else if (data_type_size_ == 8) {
      return "DOUBLE";
    }
    return "UNKNOWN";
  }

  unsigned int EvalSetNum() const {
    return eval_set_num_;
  }

  std::string_view UseTemplatedLoopItersStr() const {
    return ((!use_templated_loop_iters_) ?
      run_eval::kBoolValFalseTrueDispStr[0] : run_eval::kBoolValFalseTrueDispStr[1]);
  }

  //overloaded << operator to write EvalInputSignature object to stream
  friend std::ostream& operator<<(std::ostream& os, const EvalInputSignature& eval_input_sig);

private:
  unsigned int data_type_size_;
  unsigned int eval_set_num_;
  bool use_templated_loop_iters_;
};

//overloaded << operator to write EvalInputSignature object to stream
inline std::ostream& operator<<(std::ostream& os, const EvalInputSignature& eval_input_sig) {
  os << eval_input_sig.DataTypeStr() << " "  << eval_input_sig.eval_set_num_ << " " << eval_input_sig.UseTemplatedLoopItersStr();
  return os;
}


#endif //EVAL_INPUT_SIGNATURE_H_