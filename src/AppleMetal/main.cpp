/*
See the LICENSE.txt file for this sample’s licensing information.

Abstract:
An app that performs a simple calculation on a GPU.
*/

/*
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import "MetalAdder.h"

// This is the C version of the function that the sample
// implements in Metal Shading Language.
void add_arrays(const float* inA,
                const float* inB,
                float* result,
                int length)
{
    for (int index = 0; index < length ; index++)
    {
        result[index] = inA[index] + inB[index];
    }
}

int main(int argc, const char * argv[]) {
    @autoreleasepool {
        
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();

        // Create the custom object used to encapsulate the Metal code.
        // Initializes objects to communicate with the GPU.
        MetalAdder* adder = [[MetalAdder alloc] initWithDevice:device];
        
        // Create buffers to hold data
        [adder prepareData];
        
        // Send a command to the GPU to perform the calculation.
        [adder sendComputeCommand];

        NSLog(@"Execution finished");
    }
    return 0;
}
*/

#define NS_PRIVATE_IMPLEMENTATION
#define CA_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <iostream>
#include <QuartzCore/QuartzCore.hpp>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>
#include "metalComputeWrapper.hpp"

int main(int argc, const char * argv[]) {
    // insert code here...
    MTL::Device* device = MTL::CreateSystemDefaultDevice();
    std::cout << "Hello, World Metal!\n";
    metalComputeWrapper compWrapper;
    compWrapper.initWithDevice(device);
    compWrapper.prepareData();
    compWrapper.sendComputeCommand();
    return EXIT_SUCCESS;
}


