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

//Declares the functions to evaluate the difference between two input disparity images

#ifndef STEREO_RESULTS_EVAL_HOST_HEADER_CUH
#define STEREO_RESULTS_EVAL_HOST_HEADER_CUH

#include "stereoResultsEvalParameters.cuh"

//include for general utility functions for evaluation
#include "utilityFunctsForEvalHeader.cuh"


//initialize the stereo results
__host__ void initializeStereoResults(stereoEvaluationResults*& currentStereoEvaluation);

//given the corresponding disparity values currentDispVal1 and currentDispVal2 from disparity maps 1 and 2, update the current stereo evaluation
__host__ void updateStereoEvaluation(float unscaledCurrentDispVal1, float unscaledCurrentDispVal2, stereoEvaluationResults*& currentStereoEvaluation);

//retrieve the "final" stereo evaluation after the evaluation was updated with every set of corresponding disparity values in disparity maps 1 and 2
//(not including the border region)
__host__ void retrieveFinalStereoEvaluation(stereoEvaluationResults*& currentStereoEvaluation, unsigned int widthDisparityMap, unsigned int heightDisparityMap);

//retrieve the stereo evaluation results between the unsigned int scaled disparity maps stored in scaledDispMap1Host and scaledDispMap2Host
//(this method is primary for when the input disparity images are being read from a file)
__host__ stereoEvaluationResults* runStereoResultsEvaluationUsedUnsignedIntScaledDispMap(unsigned int* scaledDispMap1Host, unsigned int* scaledDispMap2Host, float scaleFactorDispMap1, float scaleFactorDispMap2, unsigned int widthDisparityMap, unsigned int heightDisparityMap);

//retrieve the stereo evaluation results between the float-valued unscaled disparity maps stored in unscaledDispMap1Host and unscaledDispMap2Host
//(this method is primary for when the disparity data is current stored in main memory)
__host__ stereoEvaluationResults* runStereoResultsEvaluationUseFloatUnscaledDispMap(float* unscaledDispMap1Host, float* unscaledDispMap2Host, unsigned int widthDisparityMap, unsigned int heightDisparityMap);

//retrieve the stereo evaluation results between the float-valued unscaled disparity map 1 stored in unscaledDispMap1Host and the unsigned int disparity 
//map stored in scaledDispMap2Host (this method is primary for when the calculated disparity data in unscaledDispMap1Host is current stored in main memory 
//and the "comparison" disparity (such as the ground truth) is stored in scaledDispMap2
__host__ stereoEvaluationResults* runStereoResultsEvaluationUseFloatUnscaledDispMap(float* unscaledDispMap1Host, unsigned int* scaledDispMap2Host, float scaleFactorDispMap2, unsigned int widthDisparityMap, unsigned int heightDisparityMap);

__host__ void printStereoEvaluationResults(stereoEvaluationResults* evaluationResults, FILE* resultsFile);

__host__ void writeStereoResultsToFile(FILE* currentfp, stereoEvaluationResults* evaluationResults);

#endif //STEREO_RESULTS_EVAL_HOST_HEADER_CUH
