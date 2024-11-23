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
#include "RunEvalConstsEnums.h"

//class defines input signature for evaluation run that contains evaluation set
//number, data type, and whether or not to use templated loop iteration count
class EvalInputSignature {
public:
  //constructor to generate evaluation input signature from string array with
  //strings corresponding to each part
  EvalInputSignature(const std::array<std::string_view, 3>& in_sig_strings);

  //less than operator for comparing evaluation input signatures
  //so they can be ordered
  //operator needed to support ordering since input signature
  //is used as std::map key and also for evaluation output order
  bool operator<(const EvalInputSignature& rhs) const;

private:
  unsigned int data_type_size_;
  unsigned int eval_set_num_;
  bool use_templated_loop_iters_;
};

#endif //EVAL_INPUT_SIGNATURE_H_