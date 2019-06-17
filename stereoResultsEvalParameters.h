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

//This header file defines the parameters/structures used to evaluate the results of the calculated disparity between two disparity images
//(typically a ground truth and calculated disparity image, but it doesn't have to be)

#ifndef STEREO_RESULTS_EVAL_PARAMETERS_CUH
#define STEREO_RESULTS_EVAL_PARAMETERS_CUH

//use the width/height from bpCudaParameters
#include "bpStereoCudaParameters.cuh"

//define the difference in disparity for it to be considered a "significant difference"
//pixels with difference beyond this are called a "bad pixel" if one of the images is the ground truth
//(when value is SMALL_VAL_BP, then any pixel where disparity values are not the same value are considered "bad pixels")
#define SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_1 SMALL_VAL_BP
#define SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_2 2.01f
#define SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_3 5.01f
#define SIG_DIFF_THRESHOLD_STEREO_EVAL_THRESHOLD_4 10.01f

//don't evaluate the disparities that are within the "border"
//the x-border is from the left/right sides and the
//y-border are from the top/bottom
//initialize to the border of the default ground truth disparity
#define X_BORDER_SIZE_STEREO_EVAL DEFAULT_X_BORDER_GROUND_TRUTH_DISPARITY
#define Y_BORDER_SIZE_STEREO_EVAL DEFAULT_Y_BORDER_GROUND_TRUTH_DISPARITY

//define a "cap" of the maximum difference between the corresponding disparities
//(make this infinity if you don't want to cap this value)
#define MAX_ABS_DIFF_BETWEEN_CORR_DISP INF_BP

//structure holding results of the stereo evaluation
typedef struct
{
	//total value of the absolute difference between the disparity values for all pixels in disparity images 1 and 2 (not including border regions)
	float totalDispAbsDiffNoMax;
	float totalDispAbsDiffWithMax;
	
	//average absolute difference between the disparity values in disparity images 1 and 2 (not including border regions)
	float averageDispAbsDiffNoMax;
	float averageDispAbsDiffWithMax;
	
	//proportion of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	//(not including border regions)
	float propSigDiffPixelsThreshold1;
	float propSigDiffPixelsThreshold2;
	float propSigDiffPixelsThreshold3;
	float propSigDiffPixelsThreshold4;
	
	//stores the number of pixels where the difference between the disparity values in disparity images 1 and 2 is greater than SIG_DIFF_THRESHOLD_STEREO_EVAL
	unsigned int numSigDiffPixelsThreshold1;
	unsigned int numSigDiffPixelsThreshold2;
	unsigned int numSigDiffPixelsThreshold3;
	unsigned int numSigDiffPixelsThreshold4;
} stereoEvaluationResults;

#endif //STEREO_RESULTS_EVAL_PARAMETERS_CUH
