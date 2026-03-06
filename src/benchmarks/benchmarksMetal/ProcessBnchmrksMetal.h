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
 * @file ProcessBnchmrksMetal.h
 * @author Scott Grauer-Gray
 * @brief
 * 
 * @copyright Copyright (c) 2026
 */

#ifndef PROCESS_BNCHMRKS_METAL_H_
#define PROCESS_BNCHMRKS_METAL_H_

#include "benchmarksRunProcessing/ProcessBnchmrksDevice.h"
#include "RunEval/RunTypeConstraints.h"
#include "RunEval/RunEvalConstsEnums.h"
#include "KernelBnchmrksMetal.cu"

template<RunData_t T, run_environment::AccSetting ACCELERATION, benchmarks::BenchmarkRun BENCHMARK_RUN>
class ProcessBnchmrksMetal : public ProcessBnchmrksDevice<T, ACCELERATION, BENCHMARK_RUN> {
public:
  explicit ProcessBnchmrksMetal(const ParallelParams& parallel_params) :
    ProcessBnchmrksDevice{parallel_params}
  {
    mDevice = MTL::CreateSystemDefaultDevice();
    NS::Error* error;
    
    auto defaultLibrary = mDevice->newDefaultLibrary();
    
    if (!defaultLibrary) {
        std::cerr << "Failed to find the default library.\n";
        exit(-1);
    }
    
    auto functionName = NS::String::string("TwoDMatricesBnchmrkFloat", NS::ASCIIStringEncoding);
    auto computeFunction = defaultLibrary->newFunction(functionName);
    
    if(!computeFunction){
        std::cerr << "Failed to find the compute function.\n";
    }
    
    mComputeFunctionPSO = mDevice->newComputePipelineState(computeFunction, &error);
    
    if (!mComputeFunctionPSO) {
        std::cerr << "Failed to create the pipeline state object.\n";
        exit(-1);
    }
    
    mCommandQueue = mDevice->newCommandQueue();
    
    if (!mCommandQueue) {
        std::cerr << "Failed to find command queue.\n";
        exit(-1);
    }
  }

private:    
  // The compute pipeline generated from the compute kernel in the .metal shader file.
  MTL::ComputePipelineState* mComputeFunctionPSO;
    
  // The command queue used to pass commands to the device.
  MTL::CommandQueue* mCommandQueue;

  /**
   * @brief Function to run add matrices benchmark on device<br>
   * 
   * @param mat_w_h
   * @param mat_input_0
   * @param mat_input_1
   * @param mat_result
   * @return Status of "no error" if successful, "error" status otherwise
   */
  std::optional<DetailedTimings<benchmarks::Runtime_Type>> TwoDMatricesBnchmrk(
    const unsigned int mat_w_h,
    const T* mat_input_0,
    const T* mat_input_1,
    T* mat_result) const override
  {
    if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return run_eval::Status::kError;
    }

    // Create a command buffer to hold commands.
    MTL::CommandBuffer* commandBuffer = mCommandQueue->commandBuffer();
    assert(commandBuffer != nullptr);
    
    // Start a compute pass.
    MTL::ComputeCommandEncoder* computeEncoder = commandBuffer->computeCommandEncoder();
    assert(computeEncoder != nullptr);
    
    //setup execution parameters
    const auto kernel_thread_block_dims =
      this->parallel_params_.OptParamsForKernel({0, 0});

    const unsigned int mtrx_width = mat_w_h;
    const unsigned int mtrx_height = mat_w_h;

    //process matrix addition on GPU using CUDA
    //encode the pipeline state object and its parameters.
    computeEncoder->setComputePipelineState(mComputeFunctionPSO);
    computeEncoder->setBuffer(&mtrx_width, sizeof(unsigned int), 0);
    computeEncoder->setBuffer(&mtrx_height, sizeof(unsigned int), 1);
    computeEncoder->setBuffer(mat_input_0, 0, 2);
    computeEncoder->setBuffer(mat_input_1, 0, 3);
    computeEncoder->setBuffer(mat_result, 0, 4);
    
    //set grid and threadgroup sizes
    MTL::Size gridSize = MTL::Size(mtrx_width, mtrx_height, 1);
    MTL::Size threadgroupSize =
      MTL::Size(kernel_thread_block_dims[0], kernel_thread_block_dims[1], 1);

    //encode the compute command.
    computeEncoder->dispatchThreads(gridSize, threadgroupSize);
    
    //end the compute pass.
    computeEncoder->endEncoding();
    
    //start timing and run the benchmark
    auto add_mat_start_time = std::chrono::system_clock::now();
    commandBuffer->commit();
    
    //blocks until the calculation is complete.
    commandBuffer->waitUntilCompleted();

    //end timing now that kernel completed
    auto end_mat_start_time = std::chrono::system_clock::now();

    if (ErrorCheck(__FILE__, __LINE__) != run_eval::Status::kNoError) {
      return {};
    }

    DetailedTimings add_mat_timing(benchmarks::kTimingNames);
    add_mat_timing.AddTiming(benchmarks::Runtime_Type::kTotalBnchmrkNoTransfer,
      end_mat_start_time - add_mat_start_time);

    return add_mat_timing;
  }
};

#endif //PROCESS_BNCHMRKS_METAL_H_