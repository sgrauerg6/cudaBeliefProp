/*
Copyright (C) 2009 Scott Grauer-Gray, Chandra Kambhamettu, and Kannappan Palaniappan

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

//This header declares the kernel functions and constant/texture storage to run belief propagation on CUDA

#ifndef KERNEL_BP_STEREO_CPU_H
#define KERNEL_BP_STEREO_CPU_H

#include <math.h>
#include <omp.h>
#include <algorithm>
#include <iostream>
//TODO: switch use of printf with std::format when it is supported on compiler used for development
//#include <format>
#include "BpResultsEvaluation/BpEvaluationStereoSets.h"
#include "BpRunProcessing/BpConstsEnumsAliases.h"
#include "BpSharedFuncts/SharedBpProcessingFuncts.h"
#include "BpRunProcessing/BpParallelParams.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunImp/UtilityFuncts.h"
#include "RunImpCPU/RunCPUSettings.h"
#include "RunImpCPU/SIMDProcessing.h"

/**
 * @brief Namespace to define global kernel functions for optimized belief propagation
 * processing on the CPU using OpenMP and SIMD vectorization.
 * 
 */
namespace beliefpropCPU
{
  //initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
  template<RunData_t T, unsigned int DISP_VALS>
  void InitializeBottomLevelData(const beliefprop::BpLevelProperties& current_bp_level,
    const float* image_1_pixels_device, const float* image_2_pixels_device,
    T* data_cost_stereo_checkerboard_0, T* data_cost_stereo_checkerboard_1,
    float lambda_bp, float data_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  //initialize the "data cost" for each possible disparity at the current level using the data costs from the previous level
  template<RunData_t T, unsigned int DISP_VALS>
  void InitializeCurrentLevelData(beliefprop::CheckerboardPart checkerboard_part,
    const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& prev_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    T* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  //initialize the message values at each pixel of the current level to the default value
  template<RunData_t T, unsigned int DISP_VALS>
  void InitializeMessageValsToDefaultKernel(const beliefprop::BpLevelProperties& current_bp_level,
    T* message_u_checkerboard_0, T* message_d_checkerboard_0,
    T* message_l_checkerboard_0, T* message_r_checkerboard_0,
    T* message_u_checkerboard_1, T* message_d_checkerboard_1,
    T* message_l_checkerboard_1, T* message_r_checkerboard_1,
    unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  //run the current iteration of belief propagation using the checkerboard update method where half the pixels in the "checkerboard"
  //scheme retrieve messages from each 4-connected neighbor and then update their message based on the retrieved messages and the data cost
  template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
  void RunBPIterationUsingCheckerboardUpdates(beliefprop::CheckerboardPart checkerboard_to_update,
    const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    T* message_u_checkerboard_0, T* message_d_checkerboard_0,
    T* message_l_checkerboard_0, T* message_r_checkerboard_0,
    T* message_u_checkerboard_1, T* message_d_checkerboard_1,
    T* message_l_checkerboard_1, T* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_num_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<RunData_t T, unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions(
    beliefprop::CheckerboardPart checkerboard_part_update,
    const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    T* message_u_checkerboard_0, T* message_d_checkerboard_0,
    T* message_l_checkerboard_0, T* message_r_checkerboard_0,
    T* message_u_checkerboard_1, T* message_d_checkerboard_1,
    T* message_l_checkerboard_1, T* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  //copy the computed BP message values at the current level to the corresponding locations at the "next" level down
  //the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
  template<RunData_t T, unsigned int DISP_VALS>
  void CopyMsgDataToNextLevel(beliefprop::CheckerboardPart checkerboard_part,
    const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& next_bp_level,
    const T* message_u_prev_checkerboard_0, const T* message_d_prev_checkerboard_0,
    const T* message_l_prev_checkerboard_0, const T* message_r_prev_checkerboard_0,
    const T* message_u_prev_checkerboard_1, const T* message_d_prev_checkerboard_1,
    const T* message_l_prev_checkerboard_1, const T* message_r_prev_checkerboard_1,
    const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
    const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
    const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
    const T* message_l_checkerboard_1, const T* message_r_checkerboard_1,
    unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  //retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel
  template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
  void RetrieveOutputDisparity(const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    const T* message_u_prev_checkerboard_0, const T* message_d_prev_checkerboard_0,
    const T* message_l_prev_checkerboard_0, const T* message_r_prev_checkerboard_0,
    const T* message_u_prev_checkerboard_1, const T* message_d_prev_checkerboard_1,
    const T* message_l_prev_checkerboard_1, const T* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  //retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel using SIMD vectors
  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectors(const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    const T* message_u_prev_checkerboard_0, const T* message_d_prev_checkerboard_0,
    const T* message_l_prev_checkerboard_0, const T* message_r_prev_checkerboard_0,
    const T* message_u_prev_checkerboard_1, const T* message_d_prev_checkerboard_1,
    const T* message_l_prev_checkerboard_1, const T* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    unsigned int simd_data_size,
    const ParallelParams& opt_cpu_params);
  
  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsAVX256(
    const beliefprop::BpLevelProperties& current_bp_level,
    const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
    const float* message_u_prev_checkerboard_0, const float* message_d_prev_checkerboard_0,
    const float* message_l_prev_checkerboard_0, const float* message_r_prev_checkerboard_0,
    const float* message_u_prev_checkerboard_1, const float* message_d_prev_checkerboard_1,
    const float* message_l_prev_checkerboard_1, const float* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsAVX256(
    const beliefprop::BpLevelProperties& current_bp_level,
    const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
    const short* message_u_prev_checkerboard_0, const short* message_d_prev_checkerboard_0,
    const short* message_l_prev_checkerboard_0, const short* message_r_prev_checkerboard_0,
    const short* message_u_prev_checkerboard_1, const short* message_d_prev_checkerboard_1,
    const short* message_l_prev_checkerboard_1, const short* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsAVX256(
    const beliefprop::BpLevelProperties& current_bp_level,
    const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
    const double* message_u_prev_checkerboard_0, const double* message_d_prev_checkerboard_0,
    const double* message_l_prev_checkerboard_0, const double* message_r_prev_checkerboard_0,
    const double* message_u_prev_checkerboard_1, const double* message_d_prev_checkerboard_1,
    const double* message_l_prev_checkerboard_1, const double* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

#if (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsAVX512(
    const beliefprop::BpLevelProperties& current_bp_level,
    const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
    const float* message_u_prev_checkerboard_0, const float* message_d_prev_checkerboard_0,
    const float* message_l_prev_checkerboard_0, const float* message_r_prev_checkerboard_0,
    const float* message_u_prev_checkerboard_1, const float* message_d_prev_checkerboard_1,
    const float* message_l_prev_checkerboard_1, const float* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsAVX512(
    const beliefprop::BpLevelProperties& current_bp_level,
    const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
    const short* message_u_prev_checkerboard_0, const short* message_d_prev_checkerboard_0,
    const short* message_l_prev_checkerboard_0, const short* message_r_prev_checkerboard_0,
    const short* message_u_prev_checkerboard_1, const short* message_d_prev_checkerboard_1,
    const short* message_l_prev_checkerboard_1, const short* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsAVX512(
    const beliefprop::BpLevelProperties& current_bp_level,
    const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
    const double* message_u_prev_checkerboard_0, const double* message_d_prev_checkerboard_0,
    const double* message_l_prev_checkerboard_0, const double* message_r_prev_checkerboard_0,
    const double* message_u_prev_checkerboard_1, const double* message_d_prev_checkerboard_1,
    const double* message_l_prev_checkerboard_1, const double* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);
#endif //(CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)

  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsNEON(
    const beliefprop::BpLevelProperties& current_bp_level,
    const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
    const float* message_u_prev_checkerboard_0, const float* message_d_prev_checkerboard_0,
    const float* message_l_prev_checkerboard_0, const float* message_r_prev_checkerboard_0,
    const float* message_u_prev_checkerboard_1, const float* message_d_prev_checkerboard_1,
    const float* message_l_prev_checkerboard_1, const float* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsNEON(
    const beliefprop::BpLevelProperties& current_bp_level,
    const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
    const double* message_u_prev_checkerboard_0, const double* message_d_prev_checkerboard_0,
    const double* message_l_prev_checkerboard_0, const double* message_r_prev_checkerboard_0,
    const double* message_u_prev_checkerboard_1, const double* message_d_prev_checkerboard_1,
    const double* message_l_prev_checkerboard_1, const double* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

#ifdef COMPILING_FOR_ARM
  template<unsigned int DISP_VALS>
  void RetrieveOutputDisparityUseSIMDVectorsNEON(
    const beliefprop::BpLevelProperties& current_bp_level,
    const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
    const float16_t* message_u_prev_checkerboard_0, const float16_t* message_d_prev_checkerboard_0,
    const float16_t* message_l_prev_checkerboard_0, const float16_t* message_r_prev_checkerboard_0,
    const float16_t* message_u_prev_checkerboard_1, const float16_t* message_d_prev_checkerboard_1,
    const float16_t* message_l_prev_checkerboard_1, const float16_t* message_r_prev_checkerboard_1,
    float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);
#endif //COMPILING_FOR_ARM

  //run the current iteration of belief propagation where the input messages and data costs come in as arrays
  //and the output message values are written to output message arrays
  template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
  void RunBPIterationUpdateMsgValsUseSIMDVectors(unsigned int x_val_start_processing,
    unsigned int y_val, const beliefprop::BpLevelProperties& current_bp_level,
    const U prev_u_message[DISP_VALS], const U prev_d_message[DISP_VALS],
    const U prev_l_message[DISP_VALS], const U prev_r_message[DISP_VALS],
    const U data_message[DISP_VALS],
    T* current_u_message, T* current_d_message,
    T* current_l_message, T* current_r_message,
    const U disc_k_bp_vect, bool data_aligned);

  template<RunData_t T, RunDataVect_t U>
  void RunBPIterationUpdateMsgValsUseSIMDVectors(
    unsigned int x_val_start_processing, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const U* prev_u_message, const U* prev_d_message,
    const U* prev_l_message, const U* prev_r_message,
    const U* data_message,
    T* current_u_message, T* current_d_message,
    T* current_l_message, T* current_r_message,
    const U disc_k_bp_vect, bool data_aligned,
    unsigned int bp_settings_disp_vals);
  
  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
    float* message_u_checkerboard_0, float* message_d_checkerboard_0,
    float* message_l_checkerboard_0, float* message_r_checkerboard_0,
    float* message_u_checkerboard_1, float* message_d_checkerboard_1,
    float* message_l_checkerboard_1, float* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
    short* message_u_checkerboard_0, short* message_d_checkerboard_0,
    short* message_l_checkerboard_0, short* message_r_checkerboard_0,
    short* message_u_checkerboard_1, short* message_d_checkerboard_1,
    short* message_l_checkerboard_1, short* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
    double* message_u_checkerboard_0, double* message_d_checkerboard_0,
    double* message_l_checkerboard_0, double* message_r_checkerboard_0,
    double* message_u_checkerboard_1, double* message_d_checkerboard_1,
    double* message_l_checkerboard_1, double* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);
  
#if (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
    float* message_u_checkerboard_0, float* message_d_checkerboard_0,
    float* message_l_checkerboard_0, float* message_r_checkerboard_0,
    float* message_u_checkerboard_1, float* message_d_checkerboard_1,
    float* message_l_checkerboard_1, float* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const short* data_cost_checkerboard_0, const short* data_cost_checkerboard_1,
    short* message_u_checkerboard_0, short* message_d_checkerboard_0,
    short* message_l_checkerboard_0, short* message_r_checkerboard_0,
    short* message_u_checkerboard_1, short* message_d_checkerboard_1,
    short* message_l_checkerboard_1, short* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
    double* message_u_checkerboard_0, double* message_d_checkerboard_0,
    double* message_l_checkerboard_0, double* message_r_checkerboard_0,
    double* message_u_checkerboard_1, double* message_d_checkerboard_1,
    double* message_l_checkerboard_1, double* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);
#endif //(CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)

  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const float* data_cost_checkerboard_0, const float* data_cost_checkerboard_1,
    float* message_u_checkerboard_0, float* message_d_checkerboard_0,
    float* message_l_checkerboard_0, float* message_r_checkerboard_0,
    float* message_u_checkerboard_1, float* message_d_checkerboard_1,
    float* message_l_checkerboard_1, float* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const double* data_cost_checkerboard_0, const double* data_cost_checkerboard_1,
    double* message_u_checkerboard_0, double* message_d_checkerboard_0,
    double* message_l_checkerboard_0, double* message_r_checkerboard_0,
    double* message_u_checkerboard_1, double* message_d_checkerboard_1,
    double* message_l_checkerboard_1, double* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

#ifdef COMPILING_FOR_ARM
  template<unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const float16_t* data_cost_checkerboard_0, const float16_t* data_cost_checkerboard_1,
    float16_t* message_u_checkerboard_0, float16_t* message_d_checkerboard_0,
    float16_t* message_l_checkerboard_0, float16_t* message_r_checkerboard_0,
    float16_t* message_u_checkerboard_1, float16_t* message_d_checkerboard_1,
    float16_t* message_l_checkerboard_1, float16_t* message_r_checkerboard_1,
    float disc_k_bp, unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);
#endif //COMPILING_FOR_ARM

  template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
  void RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess(
    beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    T* message_u_checkerboard_0, T* message_d_checkerboard_0,
    T* message_l_checkerboard_0, T* message_r_checkerboard_0,
    T* message_u_checkerboard_1, T* message_d_checkerboard_1,
    T* message_l_checkerboard_1, T* message_r_checkerboard_1,
    float disc_k_bp, unsigned int simd_data_size,
    unsigned int bp_settings_disp_vals,
    const ParallelParams& opt_cpu_params);

  // compute current message
  template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
  void MsgStereoSIMD(unsigned int x_val, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const U messages_neighbor_1[DISP_VALS], const U messages_neighbor_2[DISP_VALS],
    const U messages_neighbor_3[DISP_VALS], const U data_costs[DISP_VALS],
    T* dst_message_array, const U& disc_k_bp, bool data_aligned);

  // compute current message
  template<RunData_t T, RunDataVect_t U>
  void MsgStereoSIMD(unsigned int x_val, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const U* messages_neighbor_1, const U* messages_neighbor_2,
    const U* messages_neighbor_3, const U* data_costs,
    T* dst_message_array, const U& disc_k_bp, bool data_aligned,
    unsigned int bp_settings_disp_vals);

  // compute current message
  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W>
  void MsgStereoSIMDProcessing(unsigned int x_val, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const U* messages_neighbor_1, const U* messages_neighbor_2,
    const U* messages_neighbor_3, const U* data_costs,
    T* dst_message_array, const U& disc_k_bp, bool data_aligned,
    unsigned int bp_settings_disp_vals);

  // compute current message
  template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, unsigned int DISP_VALS>
  void MsgStereoSIMDProcessing(unsigned int x_val, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const U messages_neighbor_1[DISP_VALS], const U messages_neighbor_2[DISP_VALS],
    const U messages_neighbor_3[DISP_VALS], const U data_costs[DISP_VALS],
    T* dst_message_array, const U& disc_k_bp, bool data_aligned);

  //function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  template<RunDataProcess_t T, RunDataVectProcess_t U, unsigned int DISP_VALS>
  void DtStereoSIMD(U f[DISP_VALS]);

  //function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  template<RunDataProcess_t T, RunDataVectProcess_t U>
  void DtStereoSIMD(U* f, unsigned int bp_settings_disp_vals);

  template<RunDataVectProcess_t T>
  void UpdateBestDispBestVals(T& best_disparities, T& best_vals, const T& current_disparity, const T& val_at_disp) {
    std::cout << "Data type not supported for updating best disparities and values" << std::endl;
  }

  template<RunData_t T, unsigned int DISP_VALS>
  void PrintDataAndMessageValsAtPointKernel(unsigned int x_val, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
    const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
    const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
    const T* message_l_checkerboard_1, const T* message_r_checkerboard_1);

  template<RunData_t T, unsigned int DISP_VALS>
  void PrintDataAndMessageValsToPointKernel(unsigned int x_val, unsigned int y_val,
    const beliefprop::BpLevelProperties& current_bp_level,
    const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
    const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
    const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
    const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
    const T* message_l_checkerboard_1, const T* message_r_checkerboard_1);
};

//headers to include differ depending on architecture and CPU vectorization setting
#ifdef COMPILING_FOR_ARM
#include "KernelBpStereoCPU_ARMTemplateSpFuncts.h"

#if (CPU_VECTORIZATION_DEFINE == NEON_DEFINE)
#include "KernelBpStereoCPU_NEON.h"
#endif //CPU_VECTORIZATION_DEFINE == NEON_DEFINE

#else
//needed so that template specializations are used when available
#include "KernelBpStereoCPU_TemplateSpFuncts.h"

#if (CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE)
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"
#elif (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
#include "KernelBpStereoCPU_AVX256TemplateSpFuncts.h"
#include "KernelBpStereoCPU_AVX512TemplateSpFuncts.h"
#endif //CPU_VECTORIZATION_DEFINE

#endif //COMPILING_FOR_ARM

//definitions of CPU functions declared in namespace

//initialize the "data cost" for each possible disparity between the two full-sized input images ("bottom" of the image pyramid)
//the image data is stored in the CUDA arrays image1PixelsTextureBPStereo and image2PixelsTextureBPStereo
template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::InitializeBottomLevelData(
  const beliefprop::BpLevelProperties& current_bp_level,
  const float* image_1_pixels_device, const float* image_2_pixels_device,
  T* data_cost_stereo_checkerboard_0, T* data_cost_stereo_checkerboard_1,
  float lambda_bp, float data_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel), 0})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < (current_bp_level.width_level_*current_bp_level.height_level_); val++)
#else
  for (unsigned int val = 0; val < (current_bp_level.width_level_*current_bp_level.height_level_); val++)
#endif //_WIN32
  {
    const unsigned int y_val = val / current_bp_level.width_level_;
    const unsigned int x_val = val % current_bp_level.width_level_;

    beliefprop::InitializeBottomLevelDataPixel<T, DISP_VALS>(x_val, y_val, current_bp_level,
        image_1_pixels_device, image_2_pixels_device,
        data_cost_stereo_checkerboard_0, data_cost_stereo_checkerboard_1,
        lambda_bp, data_k_bp, bp_settings_disp_vals);
  }
}

//initialize the data costs at the "next" level up in the pyramid given that the data at the lower has been set
template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::InitializeCurrentLevelData(
  beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level,
  const beliefprop::BpLevelProperties& prev_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* data_cost_current_level, unsigned int offset_num, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kDataCostsAtLevel), current_bp_level.level_num_})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#else
  for (unsigned int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#endif //_WIN32
  {
    const unsigned int y_val = val / current_bp_level.width_checkerboard_level_;
    const unsigned int x_val = val % current_bp_level.width_checkerboard_level_;

    beliefprop::InitializeCurrentLevelDataPixel<T, T, DISP_VALS>(
        x_val, y_val, checkerboard_part,
        current_bp_level, prev_bp_level,
        data_cost_checkerboard_0, data_cost_checkerboard_1,
        data_cost_current_level, offset_num, bp_settings_disp_vals);
  }
}

//initialize the message values at each pixel of the current level to the default value
template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::InitializeMessageValsToDefaultKernel(
  const beliefprop::BpLevelProperties& current_bp_level,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{
    (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kInitMessageVals), 0})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#else
  for (unsigned int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#endif //_WIN32
  {
    const unsigned int y_val = val / current_bp_level.width_checkerboard_level_;
    const unsigned int x_val_in_checkerboard = val % current_bp_level.width_checkerboard_level_;

    beliefprop::InitializeMessageValsToDefaultKernelPixel<T, DISP_VALS>(
      x_val_in_checkerboard, y_val, current_bp_level,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      bp_settings_disp_vals);
  }
}

template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions(
  beliefprop::CheckerboardPart checkerboard_part_update,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  const unsigned int width_checkerboard_run_processing = current_bp_level.width_level_ / 2;

  //in cuda kernel storing data one at a time (though it is coalesced), so simd_data_size not relevant here and set to 1
  //still is a check if start of row is aligned
  const bool data_aligned = util_functs::MemoryAlignedAtDataStart(
    0, 1, current_bp_level.num_data_align_width_, current_bp_level.div_padded_checkerboard_w_align_);

#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel), current_bp_level.level_num_})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < (width_checkerboard_run_processing * current_bp_level.height_level_); val++)
#else
  for (unsigned int val = 0; val < (width_checkerboard_run_processing * current_bp_level.height_level_); val++)
#endif //_WIN32
  {
    const unsigned int y_val = val / width_checkerboard_run_processing;
    const unsigned int x_val = val % width_checkerboard_run_processing;

    beliefprop::RunBPIterationUsingCheckerboardUpdatesKernel<T, T, DISP_VALS>(
      x_val, y_val, checkerboard_part_update, current_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disc_k_bp, 0, data_aligned, bp_settings_disp_vals);
  }
}

template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUpdateMsgValsUseSIMDVectors(
  unsigned int x_val_start_processing, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U prev_u_message[DISP_VALS], const U prev_d_message[DISP_VALS],
  const U prev_l_message[DISP_VALS], const U prev_r_message[DISP_VALS],
  const U data_message[DISP_VALS],
  T* current_u_message, T* current_d_message,
  T* current_l_message, T* current_r_message,
  const U disc_k_bp_vect, bool data_aligned)
{
  MsgStereoSIMD<T, U, DISP_VALS>(x_val_start_processing, y_val, current_bp_level,
    prev_u_message, prev_l_message, prev_r_message, data_message, current_u_message,
    disc_k_bp_vect, data_aligned);

  MsgStereoSIMD<T, U, DISP_VALS>(x_val_start_processing, y_val, current_bp_level,
    prev_d_message, prev_l_message, prev_r_message, data_message, current_d_message,
    disc_k_bp_vect, data_aligned);

  MsgStereoSIMD<T, U, DISP_VALS>(x_val_start_processing, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_r_message, data_message, current_r_message,
    disc_k_bp_vect, data_aligned);

  MsgStereoSIMD<T, U, DISP_VALS>(x_val_start_processing, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_l_message, data_message, current_l_message,
    disc_k_bp_vect, data_aligned);
}

template<RunData_t T, RunDataVect_t U>
void beliefpropCPU::RunBPIterationUpdateMsgValsUseSIMDVectors(
  unsigned int x_val_start_processing, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U* prev_u_message, const U* prev_d_message,
  const U* prev_l_message, const U* prev_r_message,
  const U* data_message,
  T* current_u_message, T* current_d_message,
  T* current_l_message, T* current_r_message,
  const U disc_k_bp_vect, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMD<T, U>(x_val_start_processing, y_val, current_bp_level,
    prev_u_message, prev_l_message, prev_r_message, data_message, current_u_message,
    disc_k_bp_vect, data_aligned, bp_settings_disp_vals);

  MsgStereoSIMD<T, U>(x_val_start_processing, y_val, current_bp_level,
    prev_d_message, prev_l_message, prev_r_message, data_message, current_d_message,
    disc_k_bp_vect, data_aligned, bp_settings_disp_vals);

  MsgStereoSIMD<T, U>(x_val_start_processing, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_r_message, data_message, current_r_message,
    disc_k_bp_vect, data_aligned, bp_settings_disp_vals);

  MsgStereoSIMD<T, U>(x_val_start_processing, y_val, current_bp_level,
    prev_u_message, prev_d_message, prev_l_message, data_message, current_l_message,
    disc_k_bp_vect, data_aligned, bp_settings_disp_vals);
}

template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsProcess(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, unsigned int simd_data_size,
  unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  const unsigned int width_checkerboard_run_processing = current_bp_level.width_level_ / 2;
  const U disc_k_bp_vect = simd_processing::createSIMDVectorSameData<U>(disc_k_bp);

  if constexpr (DISP_VALS > 0) {
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
    int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel(
      {static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel), current_bp_level.level_num_})[0]};
    #pragma omp parallel for num_threads(num_threads_kernel)
#else
    #pragma omp parallel for
#endif
#ifdef _WIN32
    for (int y_val = 1; y_val < current_bp_level.height_level_ - 1; y_val++) {
#else
    for (unsigned int y_val = 1; y_val < current_bp_level.height_level_ - 1; y_val++) {
#endif //_WIN32
      //checkerboard_adjustment used for indexing into current checkerboard to update
      const unsigned int checkerboard_adjustment =
        (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) ? ((y_val) % 2) : ((y_val + 1) % 2);
      const unsigned int start_x = (checkerboard_adjustment == 1) ? 0 : 1;
      const unsigned int end_final = std::min(current_bp_level.width_checkerboard_level_ - checkerboard_adjustment,
                                              width_checkerboard_run_processing);
      const unsigned int end_x_simd_vect_start = (end_final / simd_data_size) * simd_data_size - simd_data_size;

      for (unsigned int x_val = 0; x_val < end_final; x_val += simd_data_size) {
        unsigned int x_val_process = x_val;

        //need this check first for case where endXAvxStart is 0 and start_x is 1
        //if past the last AVX start (since the next one would go beyond the row),
        //set to simd_data_size from the final pixel so processing the last numDataInAvxVector in avx
        //may be a few pixels that are computed twice but that's OK
        if (x_val_process > end_x_simd_vect_start) {
          x_val_process = end_final - simd_data_size;
        }

        //not processing at x=0 if start_x is 1 (this will cause this processing to be less aligned than ideal for this iteration)
        x_val_process = std::max(start_x, x_val_process);

        //check if the memory is aligned for AVX instructions at x_val_process location
        const bool data_aligned_x_val = util_functs::MemoryAlignedAtDataStart(
          x_val_process, simd_data_size, current_bp_level.num_data_align_width_,
          current_bp_level.div_padded_checkerboard_w_align_);

        //initialize arrays for data and message values
        U data_message[DISP_VALS], prev_u_message[DISP_VALS], prev_d_message[DISP_VALS],
          prev_l_message[DISP_VALS], prev_r_message[DISP_VALS];

        //load using aligned instructions when possible
        if (data_aligned_x_val) {
          for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
            if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              data_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(
                x_val_process, y_val, current_disparity, current_bp_level,
                DISP_VALS, data_cost_checkerboard_0);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(
                x_val_process, y_val + 1, current_disparity, current_bp_level,
                DISP_VALS, message_u_checkerboard_1);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(
                x_val_process, y_val - 1, current_disparity, current_bp_level,
                DISP_VALS, message_d_checkerboard_1);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process + checkerboard_adjustment, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_l_checkerboard_1);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                (x_val_process + checkerboard_adjustment) - 1, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_r_checkerboard_1);
            }
            else //checkerboard_part_update == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              data_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(
                x_val_process, y_val, current_disparity, current_bp_level,
                DISP_VALS, data_cost_checkerboard_1);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(
                x_val_process, y_val + 1, current_disparity, current_bp_level,
                DISP_VALS, message_u_checkerboard_0);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(
                x_val_process, y_val - 1, current_disparity, current_bp_level,
                DISP_VALS, message_d_checkerboard_0);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process + checkerboard_adjustment, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_l_checkerboard_0);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                (x_val_process + checkerboard_adjustment) - 1, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_r_checkerboard_0);
            }
          }
        } else {
          for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
            if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              data_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process, y_val, current_disparity, current_bp_level,
                DISP_VALS, data_cost_checkerboard_0);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process, y_val + 1, current_disparity, current_bp_level,
                DISP_VALS, message_u_checkerboard_1);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process, y_val - 1, current_disparity, current_bp_level,
                DISP_VALS, message_d_checkerboard_1);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process + checkerboard_adjustment, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_l_checkerboard_1);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                (x_val_process + checkerboard_adjustment) - 1, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_r_checkerboard_1);
            }
            else //checkerboard_part_update == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              data_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process, y_val, current_disparity, current_bp_level,
                DISP_VALS, data_cost_checkerboard_1);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process, y_val + 1, current_disparity, current_bp_level,
                DISP_VALS, message_u_checkerboard_0);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process, y_val - 1, current_disparity, current_bp_level,
                DISP_VALS, message_d_checkerboard_0);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                x_val_process + checkerboard_adjustment, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_l_checkerboard_0);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(
                (x_val_process + checkerboard_adjustment) - 1, y_val, current_disparity, current_bp_level,
                DISP_VALS, message_r_checkerboard_0);
            }
          }
        }

        if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
          RunBPIterationUpdateMsgValsUseSIMDVectors<T, U, DISP_VALS>(x_val_process, y_val, current_bp_level,
            prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
            message_u_checkerboard_0, message_d_checkerboard_0,
            message_l_checkerboard_0, message_r_checkerboard_0,
            disc_k_bp_vect, data_aligned_x_val);
        }
        else {
          RunBPIterationUpdateMsgValsUseSIMDVectors<T, U, DISP_VALS>(x_val_process, y_val, current_bp_level,
            prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
            message_u_checkerboard_1, message_d_checkerboard_1,
            message_l_checkerboard_1, message_r_checkerboard_1,
            disc_k_bp_vect, data_aligned_x_val);
        }
      }
    }
  }
  else {
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
    int num_threads_kernel{
      (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kBpAtLevel), current_bp_level.level_num_})[0]};
    #pragma omp parallel for num_threads(num_threads_kernel)
#else
    #pragma omp parallel for
#endif
#ifdef _WIN32
    for (int y_val = 1; y_val < current_bp_level.height_level_ - 1; y_val++) {
#else
    for (unsigned int y_val = 1; y_val < current_bp_level.height_level_ - 1; y_val++) {
#endif //_WIN32
      //checkerboard_adjustment used for indexing into current checkerboard to update
      const unsigned int checkerboard_adjustment =
        (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) ? ((y_val) % 2) : ((y_val + 1) % 2);
      const unsigned int start_x = (checkerboard_adjustment == 1) ? 0 : 1;
      const unsigned int end_final = std::min(current_bp_level.width_checkerboard_level_ - checkerboard_adjustment,
                                              width_checkerboard_run_processing);
      const unsigned int end_x_simd_vect_start = (end_final / simd_data_size) * simd_data_size - simd_data_size;

      for (unsigned int x_val = 0; x_val < end_final; x_val += simd_data_size) {
        unsigned int x_val_process = x_val;

        //need this check first for case where endXAvxStart is 0 and start_x is 1
        //if past the last AVX start (since the next one would go beyond the row),
        //set to simd_data_size from the final pixel so processing the last numDataInAvxVector in avx
        //may be a few pixels that are computed twice but that's OK
        if (x_val_process > end_x_simd_vect_start) {
          x_val_process = end_final - simd_data_size;
        }

        //not processing at x=0 if start_x is 1 (this will cause this processing to be less aligned than ideal for this iteration)
        x_val_process = std::max(start_x, x_val_process);

        //check if the memory is aligned for AVX instructions at x_val_process location
        const bool data_aligned_x_val = util_functs::MemoryAlignedAtDataStart(x_val_process, simd_data_size, current_bp_level.num_data_align_width_,
          current_bp_level.div_padded_checkerboard_w_align_);

        //initialize arrays for data and message values
        U* data_message = new U[bp_settings_disp_vals];
        U* prev_u_message = new U[bp_settings_disp_vals];
        U* prev_d_message = new U[bp_settings_disp_vals];
        U* prev_l_message = new U[bp_settings_disp_vals];
        U* prev_r_message = new U[bp_settings_disp_vals];

        //load using aligned instructions when possible
        if (data_aligned_x_val) {
          for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
            if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              data_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_0);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val + 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_1);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val - 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_1);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_1);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_1);
            }
            else //checkerboard_part_update == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              data_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_1);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val + 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_0);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val - 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_0);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_0);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_0);
            }
          }
        } 
        else {
          for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
            if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              data_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_0);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val + 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_1);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val - 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_1);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_1);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_1);
            } 
            else //checkerboard_part_update == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              data_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_1);
              prev_u_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val + 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_0);
              prev_d_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val - 1,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_0);
              prev_l_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_0);
              prev_r_message[current_disparity] = simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_0);
            }
          }
        }

        if (checkerboard_to_update == beliefprop::CheckerboardPart::kCheckerboardPart0) {
          RunBPIterationUpdateMsgValsUseSIMDVectors<T, U>(x_val_process, y_val, current_bp_level,
            prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
            message_u_checkerboard_0, message_d_checkerboard_0,
            message_l_checkerboard_0, message_r_checkerboard_0,
            disc_k_bp_vect, data_aligned_x_val, bp_settings_disp_vals);
        }
        else {
          RunBPIterationUpdateMsgValsUseSIMDVectors<T, U>(x_val_process, y_val, current_bp_level,
            prev_u_message, prev_d_message, prev_l_message, prev_r_message, data_message,
            message_u_checkerboard_1, message_d_checkerboard_1,
            message_l_checkerboard_1, message_r_checkerboard_1,
            disc_k_bp_vect, data_aligned_x_val, bp_settings_disp_vals);
        }

        delete [] data_message;
        delete [] prev_u_message;
        delete [] prev_d_message;
        delete [] prev_l_message;
        delete [] prev_r_message;
      }
    }
  }
}

//kernel function to run the current iteration of belief propagation in parallel using
//the checkerboard update method where half the pixels in the "checkerboard" scheme
//retrieve messages from each 4-connected neighbor and then update their message based
//on the retrieved messages and the data cost
template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefpropCPU::RunBPIterationUsingCheckerboardUpdates(
  beliefprop::CheckerboardPart checkerboard_to_update, const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  float disc_k_bp, unsigned int bp_settings_num_disp_vals,
  const ParallelParams& opt_cpu_params)
{
#ifdef COMPILING_FOR_ARM
if constexpr (ACCELERATION == run_environment::AccSetting::kNEON)
  {
    if (current_bp_level.width_checkerboard_level_ > 5)
    {
      RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsNEON<DISP_VALS>(checkerboard_to_update,
        current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
    }
    else
    {
      RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions<T, DISP_VALS>(checkerboard_to_update,
        current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
    }
  }
  else
  {
    RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions<T, DISP_VALS>(checkerboard_to_update,
      current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
  }
#else
#if ((CPU_VECTORIZATION_DEFINE == AVX_256_DEFINE) || (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE))
  if constexpr (ACCELERATION == run_environment::AccSetting::kAVX256)
  {
    //only use AVX-256 if width of processing checkerboard is over 10
    if (current_bp_level.width_checkerboard_level_ > 10)
    {
      RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX256<DISP_VALS>(checkerboard_to_update,
        current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
    }
    else
    {
      RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions<T, DISP_VALS>(checkerboard_to_update,
        current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
    }
  }
#endif //CPU_VECTORIZATION_DEFINE
#if (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
  else if constexpr (ACCELERATION == run_environment::AccSetting::kAVX512)
  {
    //only use AVX-512 if width of processing checkerboard is over 20
    if (current_bp_level.width_checkerboard_level_ > 20)
    {
      RunBPIterationUsingCheckerboardUpdatesUseSIMDVectorsAVX512<DISP_VALS>(checkerboard_to_update,
        current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
    }
    else
    {
      RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions<T, DISP_VALS>(checkerboard_to_update,
        current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
    }
  }
#endif //CPU_VECTORIZATION_DEFINE
  else
  {
    RunBPIterationUsingCheckerboardUpdatesNoPackedInstructions<T, DISP_VALS>(checkerboard_to_update,
      current_bp_level, data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disc_k_bp, bp_settings_num_disp_vals, opt_cpu_params);
  }
#endif //COMPILING_FOR_ARM
}

//kernel to copy the computed BP message values at the current level to the corresponding locations at the "next" level down
//the kernel works from the point of view of the pixel at the prev level that is being copied to four different places
template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::CopyMsgDataToNextLevel(beliefprop::CheckerboardPart checkerboard_part,
  const beliefprop::BpLevelProperties& current_bp_level, const beliefprop::BpLevelProperties& next_bp_level,
  const T* message_u_prev_checkerboard_0, const T* message_d_prev_checkerboard_0,
  const T* message_l_prev_checkerboard_0, const T* message_r_prev_checkerboard_0,
  const T* message_u_prev_checkerboard_1, const T* message_d_prev_checkerboard_1,
  const T* message_l_prev_checkerboard_1, const T* message_r_prev_checkerboard_1,
  T* message_u_checkerboard_0, T* message_d_checkerboard_0,
  T* message_l_checkerboard_0, T* message_r_checkerboard_0,
  T* message_u_checkerboard_1, T* message_d_checkerboard_1,
  T* message_l_checkerboard_1, T* message_r_checkerboard_1,
  unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{(int)opt_cpu_params.OptParamsForKernel(
    {static_cast<unsigned int>(beliefprop::BpKernel::kCopyAtLevel), current_bp_level.level_num_})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#else
  for (unsigned int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#endif //_WIN32
  {
    const unsigned int y_val = val / current_bp_level.width_checkerboard_level_;
    const unsigned int x_val = val % current_bp_level.width_checkerboard_level_;

    beliefprop::CopyMsgDataToNextLevelPixel<T, DISP_VALS>(x_val, y_val,
      checkerboard_part, current_bp_level, next_bp_level,
      message_u_prev_checkerboard_0, message_d_prev_checkerboard_0,
      message_l_prev_checkerboard_0, message_r_prev_checkerboard_0,
      message_u_prev_checkerboard_1, message_d_prev_checkerboard_1,
      message_l_prev_checkerboard_1, message_r_prev_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      bp_settings_disp_vals);
  }
}

template<RunData_t T, unsigned int DISP_VALS, run_environment::AccSetting ACCELERATION>
void beliefpropCPU::RetrieveOutputDisparity(
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  const ParallelParams& opt_cpu_params)
{
  if constexpr (ACCELERATION == run_environment::AccSetting::kNone) {
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{
    (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp), 0})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#else
  for (unsigned int val = 0; val < (current_bp_level.width_checkerboard_level_*current_bp_level.height_level_); val++)
#endif //_WIN32
  {
    const unsigned int y_val = val / current_bp_level.width_checkerboard_level_;
    const unsigned int x_val = val % current_bp_level.width_checkerboard_level_;

    beliefprop::RetrieveOutputDisparityPixel<T, T, DISP_VALS>(
      x_val, y_val, current_bp_level,
      data_cost_checkerboard_0, data_cost_checkerboard_1,
      message_u_checkerboard_0, message_d_checkerboard_0,
      message_l_checkerboard_0, message_r_checkerboard_0,
      message_u_checkerboard_1, message_d_checkerboard_1,
      message_l_checkerboard_1, message_r_checkerboard_1,
      disparity_between_images_device, bp_settings_disp_vals);
    }
  }
  else {
#ifndef COMPILING_FOR_ARM
    //SIMD vectorization of output disparity
    if constexpr (ACCELERATION == run_environment::AccSetting::kAVX512) {
#if (CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
      RetrieveOutputDisparityUseSIMDVectorsAVX512<DISP_VALS>(current_bp_level,
        data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disparity_between_images_device, bp_settings_disp_vals, opt_cpu_params);
#endif //(CPU_VECTORIZATION_DEFINE == AVX_512_DEFINE)
    }
    else if constexpr (ACCELERATION == run_environment::AccSetting::kAVX256) {
      RetrieveOutputDisparityUseSIMDVectorsAVX256<DISP_VALS>(current_bp_level,
        data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disparity_between_images_device, bp_settings_disp_vals, opt_cpu_params);
    }
#else
      RetrieveOutputDisparityUseSIMDVectorsNEON<DISP_VALS>(current_bp_level,
        data_cost_checkerboard_0, data_cost_checkerboard_1,
        message_u_checkerboard_0, message_d_checkerboard_0,
        message_l_checkerboard_0, message_r_checkerboard_0,
        message_u_checkerboard_1, message_d_checkerboard_1,
        message_l_checkerboard_1, message_r_checkerboard_1,
        disparity_between_images_device, bp_settings_disp_vals, opt_cpu_params);
#endif //COMPILING_FOR_ARM
  }
}

//retrieve the best disparity estimate from image 1 to image 2 for each pixel in parallel using SIMD vectors
template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, unsigned int DISP_VALS>
void beliefpropCPU::RetrieveOutputDisparityUseSIMDVectors(
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1,
  float* disparity_between_images_device, unsigned int bp_settings_disp_vals,
  unsigned int simd_data_size,
  const ParallelParams& opt_cpu_params)
{
  const unsigned int width_checkerboard_run_processing = current_bp_level.width_level_ / 2;

  //initially get output for each checkerboard
  //set width of disparity checkerboard to be a multiple of simd_data_size so that SIMD vectors can be aligned
  unsigned int width_disp_checkerboard =
    ((current_bp_level.padded_width_checkerboard_level_ % current_bp_level.num_data_align_width_) == 0) ?
      current_bp_level.padded_width_checkerboard_level_  :
      (current_bp_level.padded_width_checkerboard_level_ + (current_bp_level.num_data_align_width_ - 
        (current_bp_level.padded_width_checkerboard_level_ % current_bp_level.num_data_align_width_)));
  const unsigned int num_data_disp_checkerboard = width_disp_checkerboard * current_bp_level.height_level_;
#ifdef _WIN32
  V* disparity_checkerboard_0 = 
    static_cast<V*>(
      _aligned_malloc(2 * num_data_disp_checkerboard * sizeof(V), current_bp_level.num_data_align_width_ * sizeof(V)));
#else
  V* disparity_checkerboard_0 =
    static_cast<V*>(std::aligned_alloc(
      current_bp_level.num_data_align_width_ * sizeof(V), 2 * num_data_disp_checkerboard * sizeof(V)));
#endif

  for (const auto checkerboardGetDispMap : {beliefprop::CheckerboardPart::kCheckerboardPart0,
                                            beliefprop::CheckerboardPart::kCheckerboardPart1})
  {
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
    int num_threads_kernel{
      (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp), 0})[0]};
    #pragma omp parallel for num_threads(num_threads_kernel)
#else
    #pragma omp parallel for
#endif
#ifdef _WIN32
    for (int y_val = 1; y_val < current_bp_level.height_level_ - 1; y_val++) {
#else
    for (unsigned int y_val = 1; y_val < current_bp_level.height_level_ - 1; y_val++) {
#endif //_WIN32
      //checkerboard_adjustment used for indexing into current checkerboard to retrieve best disparities
      const unsigned int checkerboard_adjustment = 
        (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) ? ((y_val) % 2) : ((y_val + 1) % 2);
      const unsigned int start_x = (checkerboard_adjustment == 1) ? 0 : 1;
      const unsigned int end_final = std::min(current_bp_level.width_checkerboard_level_ - checkerboard_adjustment,
                                              width_checkerboard_run_processing);
      const unsigned int end_x_simd_vect_start = (end_final / simd_data_size) * simd_data_size - simd_data_size;

      for (unsigned int x_val = 0; x_val < end_final; x_val += simd_data_size) {
        unsigned int x_val_process = x_val;

        //need this check first for case where endXAvxStart is 0 and start_x is 1
        //if past the last AVX start (since the next one would go beyond the row),
        //set to simd_data_size from the final pixel so processing the last numDataInAvxVector in avx
        //may be a few pixels that are computed twice but that's OK
        if (x_val_process > end_x_simd_vect_start) {
          x_val_process = end_final - simd_data_size;
        }

        //not processing at x=0 if start_x is 1 (this will cause this processing to be less aligned than ideal for this iteration)
        x_val_process = std::max(start_x, x_val_process);

        //get index for output into disparity map corresponding to checkerboard
        const unsigned int index_output = (y_val * width_disp_checkerboard) + x_val_process;

        //check if the memory is aligned for AVX instructions at x_val_process location
        const bool data_aligned_x_val = util_functs::MemoryAlignedAtDataStart(x_val_process, simd_data_size,
          current_bp_level.num_data_align_width_, current_bp_level.div_padded_checkerboard_w_align_);

        //declare SIMD vectors for data and message values at each disparity
        //U data_message, prev_u_message, prev_d_message, prev_l_message, prev_r_message;

        //declare SIMD vectors for current best values and best disparities
        W best_vals, best_disparities, val_at_disp;

        //load using aligned instructions when possible
        if constexpr (DISP_VALS > 0) {
          for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
            if (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              if (data_aligned_x_val) {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>(
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, DISP_VALS, message_u_checkerboard_1),
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, DISP_VALS, message_d_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_l_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_r_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, DISP_VALS, data_cost_checkerboard_0));
              }
              else {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>(
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, DISP_VALS, message_u_checkerboard_1),
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, DISP_VALS, message_d_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_l_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_r_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, DISP_VALS, data_cost_checkerboard_0));
              }
            }
            else //checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              if (data_aligned_x_val) {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>(
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, DISP_VALS, message_u_checkerboard_0),
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, DISP_VALS, message_d_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_l_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_r_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, DISP_VALS, data_cost_checkerboard_1));
              }
              else {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>(
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, DISP_VALS, message_u_checkerboard_0),
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, DISP_VALS, message_d_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_l_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, DISP_VALS, message_r_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, DISP_VALS, data_cost_checkerboard_1));
              }
            }
            if (current_disparity == 0) {
              best_vals = val_at_disp;
              //set disp at min vals to all 0
              best_disparities = simd_processing::createSIMDVectorSameData<W>(0.0f);
            }
            else {
              //update best disparity and best values
              //if value at current disparity is lower than current best value, need
              //to update best value to current value and set best disparity to current disparity
              UpdateBestDispBestVals(best_disparities, best_vals,
                simd_processing::createSIMDVectorSameData<W>((float)current_disparity), val_at_disp);
            }
          }
          if (data_aligned_x_val) {
            if (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              simd_processing::StorePackedDataAligned<V, W>(index_output, disparity_checkerboard_0, best_disparities);
            }
            else //checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              simd_processing::StorePackedDataAligned<V, W>(
                num_data_disp_checkerboard + index_output, disparity_checkerboard_0, best_disparities);
            }
          }
          else {
            if (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              simd_processing::StorePackedDataUnaligned<V, W>(index_output, disparity_checkerboard_0, best_disparities);
            }
            else //checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              simd_processing::StorePackedDataUnaligned<V, W>(
                num_data_disp_checkerboard + index_output, disparity_checkerboard_0, best_disparities);
            }
          }
        }
        else {
          for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++) {
            if (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              if (data_aligned_x_val) {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>(
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_1),
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp,
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_0));
              }
              else {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>(
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_1),
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_1));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_0));
              }
            }
            else //checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              if (data_aligned_x_val) {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>( 
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_0),
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataAligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_1));
              }
              else {
                //retrieve and get sum of message and data values
                val_at_disp = simd_processing::AddVals<U, U, W>( 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val + 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_u_checkerboard_0),
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val - 1,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_d_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process + checkerboard_adjustment, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_l_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp, 
                  simd_processing::LoadPackedDataUnaligned<T, U>((x_val_process + checkerboard_adjustment) - 1, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, message_r_checkerboard_0));
                val_at_disp = simd_processing::AddVals<W, U, W>(val_at_disp,
                  simd_processing::LoadPackedDataUnaligned<T, U>(x_val_process, y_val,
                    current_disparity, current_bp_level, bp_settings_disp_vals, data_cost_checkerboard_1));
              }
            }
            if (current_disparity == 0) {
              best_vals = val_at_disp;
              //set disp at min vals to all 0
              best_disparities = simd_processing::createSIMDVectorSameData<W>(0.0f);
            }
            else {
              //update best disparity and best values
              //if value at current disparity is lower than current best value, need
              //to update best value to current value and set best disparity to current disparity
              UpdateBestDispBestVals(best_disparities, best_vals,
                simd_processing::createSIMDVectorSameData<W>((float)current_disparity), val_at_disp);
            }
          }
          //store best disparities in checkerboard being updated
          if (data_aligned_x_val) {
            if (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              simd_processing::StorePackedDataAligned<V, W>(index_output, disparity_checkerboard_0, best_disparities);
            }
            else //checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              simd_processing::StorePackedDataAligned<V, W>(
                num_data_disp_checkerboard + index_output, disparity_checkerboard_0, best_disparities);
            }
          }
          else {
            if (checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart0) {
              simd_processing::StorePackedDataUnaligned<V, W>(
                index_output, disparity_checkerboard_0, best_disparities);
            }
            else //checkerboardGetDispMap == beliefprop::CheckerboardPart::kCheckerboardPart1
            {
              simd_processing::StorePackedDataUnaligned<V, W>(
                num_data_disp_checkerboard + index_output, disparity_checkerboard_0, best_disparities);
            }
          }
        }
      }
    }
  }

  //combine output disparity maps from each checkerboard
  //start with checkerboard 0 in first row since (0, 0) corresponds to (0, 0)
  //in checkerboard 0 and (1, 0) corresponds to (0, 0) in checkerboard 1
#ifdef SET_THREAD_COUNT_INDIVIDUAL_KERNELS_CPU
  int num_threads_kernel{
    (int)opt_cpu_params.OptParamsForKernel({static_cast<unsigned int>(beliefprop::BpKernel::kOutputDisp), 0})[0]};
  #pragma omp parallel for num_threads(num_threads_kernel)
#else
  #pragma omp parallel for
#endif
#ifdef _WIN32
  for (int y=0; y < current_bp_level.height_level_; y++)
#else
  for (unsigned int y=0; y < current_bp_level.height_level_; y++)
#endif //_WIN32
  {
    const bool start_checkerboard_0 = ((y%2) == 0);
    unsigned int checkerboard_index = y * width_disp_checkerboard;
    for (unsigned int x=0; x < (current_bp_level.width_level_); x += 2) {
      if ((y == 0) || (y == (current_bp_level.height_level_ - 1))) {
        disparity_between_images_device[y * current_bp_level.width_level_ + (x + 0)] = 0;
        disparity_between_images_device[y * current_bp_level.width_level_ + (x + 1)] = 0;
      }
      else {
        if (start_checkerboard_0) {
          if ((x == 0) || (x == (current_bp_level.width_level_ - 1))) {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 0)] = 0;
          }
          else {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 0)] =
              (float)disparity_checkerboard_0[checkerboard_index];
          }
          if ((x + 1) == (current_bp_level.width_level_ - 1)) {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 1)] = 0;
          }
          else if ((x + 1) < current_bp_level.width_level_) {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 1)] =
                (float)disparity_checkerboard_0[num_data_disp_checkerboard + checkerboard_index];
          }
        }
        else {
          if ((x == 0) || (x == (current_bp_level.width_level_ - 1))) {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 0)] = 0;
          }
          else {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 0)] =
              (float)disparity_checkerboard_0[num_data_disp_checkerboard + checkerboard_index];
          }
          if ((x + 1) == (current_bp_level.width_level_ - 1)) {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 1)] = 0;
          }
          else if ((x + 1) < current_bp_level.width_level_) {
            disparity_between_images_device[y * current_bp_level.width_level_ + (x + 1)] =
              (float)disparity_checkerboard_0[checkerboard_index];
          }
        }
        //increment checkerboard index for next x-value
        checkerboard_index++;
      }
    }
  }
    
  //delete [] disparity_checkerboard_0;
  free(disparity_checkerboard_0);
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method
//(see "Efficient Belief Propagation for Early Vision")
template<RunDataProcess_t T, RunDataVectProcess_t U, unsigned int DISP_VALS>
void beliefpropCPU::DtStereoSIMD(U f[DISP_VALS])
{
  U prev;
  const U vector_all_one_val = simd_processing::ConvertValToDatatype<U, T>(1.0f);
  for (unsigned int current_disparity = 1; current_disparity < DISP_VALS; current_disparity++)
  {
    //prev = f[current_disparity-1] + (T)1.0;
    prev = simd_processing::AddVals<U, U, U>(f[current_disparity - 1], vector_all_one_val);

    /*if (prev < f[current_disparity])
          f[current_disparity] = prev;*/
    f[current_disparity] = simd_processing::GetMinByElement<U>(prev, f[current_disparity]);
  }

  for (int current_disparity = (int)DISP_VALS-2; current_disparity >= 0; current_disparity--)
  {
    //prev = f[current_disparity+1] + (T)1.0;
    prev = simd_processing::AddVals<U, U, U>(f[current_disparity + 1], vector_all_one_val);

    //if (prev < f[current_disparity])
    //  f[current_disparity] = prev;
    f[current_disparity] = simd_processing::GetMinByElement<U>(prev, f[current_disparity]);
  }
}

//compute current message
template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W, unsigned int DISP_VALS>
void beliefpropCPU::MsgStereoSIMDProcessing(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U messages_neighbor_1[DISP_VALS], const U messages_neighbor_2[DISP_VALS],
  const U messages_neighbor_3[DISP_VALS], const U data_costs[DISP_VALS],
  T* dst_message_array, const U& disc_k_bp, bool data_aligned)
{
  // aggregate and find min
  //T minimum = beliefprop::kInfBp;
  W minimum = simd_processing::ConvertValToDatatype<W, V>(beliefprop::kInfBp);
  W dst[DISP_VALS];

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
  {
    //dst[current_disparity] = messages_neighbor_1[current_disparity] + messages_neighbor_2[current_disparity] +
    //                         messages_neighbor_3[current_disparity] + data_costs[current_disparity];
    dst[current_disparity] =
      simd_processing::AddVals<U, U, W>(messages_neighbor_1[current_disparity], messages_neighbor_2[current_disparity]);
    dst[current_disparity] =
      simd_processing::AddVals<W, U, W>(dst[current_disparity], messages_neighbor_3[current_disparity]);
    dst[current_disparity] =
      simd_processing::AddVals<W, U, W>(dst[current_disparity], data_costs[current_disparity]);

    //if (dst[current_disparity] < minimum)
    //  minimum = dst[current_disparity];
    minimum = simd_processing::GetMinByElement<W>(minimum, dst[current_disparity]);
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereoSIMD<V, W, DISP_VALS>(dst);

  // truncate
  //minimum += disc_k_bp;
  minimum = simd_processing::AddVals<W, U, W>(minimum, disc_k_bp);

  // normalize
  //T val_to_normalize = 0;
  W val_to_normalize = simd_processing::ConvertValToDatatype<W, V>(0.0);

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
  {
    /*if (minimum < dst[current_disparity]) {
      dst[current_disparity] = minimum;
    }*/
    dst[current_disparity] = simd_processing::GetMinByElement<W>(minimum, dst[current_disparity]);

    //val_to_normalize += dst[current_disparity];
    val_to_normalize = simd_processing::AddVals<W, W, W>(val_to_normalize, dst[current_disparity]);
  }

  //val_to_normalize /= DISP_VALS;
  val_to_normalize = simd_processing::divideVals<W, W, W>(
    val_to_normalize, simd_processing::ConvertValToDatatype<W, V>((double)DISP_VALS));

  unsigned int dest_message_array_index = beliefprop::RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_, 0, DISP_VALS);

  for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++)
  {
    //dst[current_disparity] -= val_to_normalize;
    dst[current_disparity] = simd_processing::SubtractVals<W, W, W>(dst[current_disparity], val_to_normalize);

    if (data_aligned) {
      simd_processing::StorePackedDataAligned<T, W>(dest_message_array_index, dst_message_array, dst[current_disparity]);
    }
    else {
      simd_processing::StorePackedDataUnaligned<T, W>(dest_message_array_index, dst_message_array, dst[current_disparity]);
    }

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      dest_message_array_index += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      dest_message_array_index++;
    }
  }
}

//function retrieve the minimum value at each 1-d disparity value in O(n) time using Felzenszwalb's method
//(see "Efficient Belief Propagation for Early Vision")
template<RunDataProcess_t T, RunDataVectProcess_t U>
void beliefpropCPU::DtStereoSIMD(U* f, unsigned int bp_settings_disp_vals)
{
  U prev;
  const U vector_all_one_val = simd_processing::ConvertValToDatatype<U, T>(1.0f);
  for (unsigned int current_disparity = 1; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //prev = f[current_disparity-1] + (T)1.0;
    prev = simd_processing::AddVals<U, U, U>(f[current_disparity - 1], vector_all_one_val);

    /*if (prev < f[current_disparity])
          f[current_disparity] = prev;*/
    f[current_disparity] = simd_processing::GetMinByElement<U>(prev, f[current_disparity]);
  }

  for (int current_disparity = (int)bp_settings_disp_vals-2; current_disparity >= 0; current_disparity--)
  {
    //prev = f[current_disparity+1] + (T)1.0;
    prev = simd_processing::AddVals<U, U, U>(f[current_disparity + 1], vector_all_one_val);

    //if (prev < f[current_disparity])
    //  f[current_disparity] = prev;
    f[current_disparity] = simd_processing::GetMinByElement<U>(prev, f[current_disparity]);
  }
}

// compute current message
template<RunData_t T, RunDataVect_t U, RunDataProcess_t V, RunDataVectProcess_t W>
void beliefpropCPU::MsgStereoSIMDProcessing(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U* messages_neighbor_1, const U* messages_neighbor_2,
  const U* messages_neighbor_3, const U* data_costs,
  T* dst_message_array,
  const U& disc_k_bp, bool data_aligned,
  unsigned int bp_settings_disp_vals)
{
  // aggregate and find min
  //T minimum = beliefprop::kInfBp;
  W minimum = simd_processing::ConvertValToDatatype<W, V>(beliefprop::kInfBp);
  W* dst = new W[bp_settings_disp_vals];

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //dst[current_disparity] = messages_neighbor_1[current_disparity] + messages_neighbor_2[current_disparity] +
    //                         messages_neighbor_3[current_disparity] + data_costs[current_disparity];
    dst[current_disparity] = simd_processing::AddVals<U, U, W>(
      messages_neighbor_1[current_disparity], messages_neighbor_2[current_disparity]);
    dst[current_disparity] = simd_processing::AddVals<W, U, W>(
      dst[current_disparity], messages_neighbor_3[current_disparity]);
    dst[current_disparity] = simd_processing::AddVals<W, U, W>(
      dst[current_disparity], data_costs[current_disparity]);

    //if (dst[current_disparity] < minimum)
    //  minimum = dst[current_disparity];
    minimum = simd_processing::GetMinByElement<W>(minimum, dst[current_disparity]);
  }

  //retrieve the minimum value at each disparity in O(n) time using Felzenszwalb's method
  //(see "Efficient Belief Propagation for Early Vision")
  DtStereoSIMD<V, W>(dst, bp_settings_disp_vals);

  // truncate
  //minimum += disc_k_bp;
  minimum = simd_processing::AddVals<W, U, W>(minimum, disc_k_bp);

  // normalize
  //T val_to_normalize = 0;
  W val_to_normalize = simd_processing::ConvertValToDatatype<W, V>(0.0f);

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //if (minimum < dst[current_disparity]) {
    //  dst[current_disparity] = minimum;
    //}
    dst[current_disparity] = simd_processing::GetMinByElement<W>(minimum, dst[current_disparity]);

    //val_to_normalize += dst[current_disparity];
    val_to_normalize = simd_processing::AddVals<W, W, W>(val_to_normalize, dst[current_disparity]);
  }

  //val_to_normalize /= DISP_VALS;
  val_to_normalize = simd_processing::divideVals<W, W, W>(
    val_to_normalize, simd_processing::ConvertValToDatatype<W, V>((float)bp_settings_disp_vals));

  unsigned int dest_message_array_index = beliefprop::RetrieveIndexInDataAndMessage(x_val, y_val,
    current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_, 0, bp_settings_disp_vals);

  for (unsigned int current_disparity = 0; current_disparity < bp_settings_disp_vals; current_disparity++)
  {
    //dst[current_disparity] -= val_to_normalize;
    dst[current_disparity] = simd_processing::SubtractVals<W, W, W>(dst[current_disparity], val_to_normalize);

    if (data_aligned) {
      simd_processing::StorePackedDataAligned<T, W>(dest_message_array_index, dst_message_array, dst[current_disparity]);
    }
    else {
      simd_processing::StorePackedDataUnaligned<T, W>(dest_message_array_index, dst_message_array, dst[current_disparity]);
    }

    if constexpr (beliefprop::kOptimizedIndexingSetting) {
      dest_message_array_index += current_bp_level.padded_width_checkerboard_level_;
    }
    else {
      dest_message_array_index++;
    }
  }

  delete [] dst;
}

// compute current message
template<RunData_t T, RunDataVect_t U, unsigned int DISP_VALS>
void beliefpropCPU::MsgStereoSIMD(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U messages_neighbor_1[DISP_VALS], const U messages_neighbor_2[DISP_VALS],
  const U messages_neighbor_3[DISP_VALS], const U data_costs[DISP_VALS],
  T* dst_message_array,
  const U& disc_k_bp, bool data_aligned)
{
  MsgStereoSIMDProcessing<T, U, T, U, DISP_VALS>(x_val, y_val,
    current_bp_level, messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs, dst_message_array, disc_k_bp, data_aligned);
}

// compute current message
template<RunData_t T, RunDataVect_t U>
void beliefpropCPU::MsgStereoSIMD(unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const U* messages_neighbor_1, const U* messages_neighbor_2,
  const U* messages_neighbor_3, const U* data_costs,
  T* dst_message_array,
  const U& disc_k_bp, bool data_aligned, unsigned int bp_settings_disp_vals)
{
  MsgStereoSIMDProcessing<T, U, T, U>(
    x_val, y_val, current_bp_level,
    messages_neighbor_1, messages_neighbor_2,
    messages_neighbor_3, data_costs,
    dst_message_array, disc_k_bp, data_aligned,
    bp_settings_disp_vals);
}

template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::PrintDataAndMessageValsAtPointKernel(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1)
{
  if (((x_val + y_val) % 2) == 0) {
    printf("x_val: %u\n", x_val);
    printf("y_val: %u\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %u\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)message_u_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)message_d_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)message_l_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)message_r_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)data_cost_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  } else {
    printf("x_val: %u\n", x_val);
    printf("y_val: %u\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %u\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float)message_u_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float)message_d_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float)message_l_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float)message_r_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float)data_cost_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
}

template<RunData_t T, unsigned int DISP_VALS>
void beliefpropCPU::PrintDataAndMessageValsToPointKernel(
  unsigned int x_val, unsigned int y_val,
  const beliefprop::BpLevelProperties& current_bp_level,
  const T* data_cost_checkerboard_0, const T* data_cost_checkerboard_1,
  const T* message_u_checkerboard_0, const T* message_d_checkerboard_0,
  const T* message_l_checkerboard_0, const T* message_r_checkerboard_0,
  const T* message_u_checkerboard_1, const T* message_d_checkerboard_1,
  const T* message_l_checkerboard_1, const T* message_r_checkerboard_1)
{
  const unsigned int checkerboard_adjustment = (((x_val + y_val) % 2) == 0) ? ((y_val)%2) : ((y_val+1)%2);
  if (((x_val + y_val) % 2) == 0) {
    //TODO: switch use of printf with std::format when it is supported on compiler used for development
    //std::cout << std::format("x_val: {}", x_val) << std::endl;
    printf("x_val: %u\n", x_val);
    printf("y_val: %u\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %u\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float) message_u_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val + 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float) message_d_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val - 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float) message_l_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2 + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float) message_r_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          (x_val / 2 - 1) + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float) data_cost_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
  else {
    printf("x_val: %u\n", x_val);
    printf("y_val: %u\n", y_val);
    for (unsigned int current_disparity = 0; current_disparity < DISP_VALS; current_disparity++) {
      printf("DISP: %u\n", current_disparity);
      printf("messageUPrevStereoCheckerboard: %f \n",
        (float) message_u_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val + 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageDPrevStereoCheckerboard: %f \n",
        (float) message_d_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val - 1, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageLPrevStereoCheckerboard: %f \n",
        (float) message_l_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2 + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("messageRPrevStereoCheckerboard: %f \n",
        (float) message_r_checkerboard_0[beliefprop::RetrieveIndexInDataAndMessage(
          (x_val / 2 - 1) + checkerboard_adjustment, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
      printf("dataCostStereoCheckerboard: %f \n",
        (float) data_cost_checkerboard_1[beliefprop::RetrieveIndexInDataAndMessage(
          x_val / 2, y_val, current_bp_level.padded_width_checkerboard_level_, current_bp_level.height_level_,
          current_disparity, DISP_VALS)]);
    }
  }
}

#endif //KERNEL_BP_STEREO_CPU_H
